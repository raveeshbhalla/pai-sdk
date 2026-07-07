"""Integrated telemetry: connect plumbing once, every call emits traces."""

import json

import pytest

from pai_sdk import (
    TraceCollector,
    TraceContext,
    configure_telemetry,
    generate_text,
    jsonl_sink,
    load_prompt,
    otel_sink,
    stream_text,
    telemetry,
)
from pai_sdk.integrations.otel import trace_from_otel_spans
from pai_sdk.trace import generate_trace, stream_trace

from conftest import FakeModel, text_step, tool_step


@pytest.fixture(autouse=True)
def _clean_global_telemetry():
    configure_telemetry()
    yield
    configure_telemetry()


PROMPT_DOC = {
    "name": "triage",
    "input": {"company": "string", "ticket": "string"},
    "system": "You triage for {{company}}.",
    "user": "Ticket: {{ticket}}",
}


async def test_generate_text_emits_to_configured_sink():
    collector = TraceCollector()
    configure_telemetry(collector)
    result = await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
    assert result.text == "ok"  # result stays a plain GenerateTextResult
    assert len(collector.traces) == 1
    span = collector.last.spans[0]
    assert [m.role for m in span.messages] == ["user", "assistant"]
    assert span.metadata["input_message_count"] == 1

    configure_telemetry()  # disconnect
    await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
    assert len(collector.traces) == 1


async def test_scoped_and_per_call_telemetry():
    scoped, per_call = TraceCollector(), TraceCollector()
    with telemetry(scoped):
        await generate_text(
            model=FakeModel(steps=[text_step("ok")]),
            prompt="hi",
            telemetry=per_call,
        )
        await generate_text(
            model=FakeModel(steps=[text_step("ok")]), prompt="hi", telemetry=False
        )
    await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
    assert len(scoped.traces) == 1  # telemetry=False skipped, out-of-scope skipped
    assert len(per_call.traces) == 1


async def test_prompt_generate_is_traced_with_prompt_context():
    collector = TraceCollector()
    prompt = load_prompt(PROMPT_DOC)
    with telemetry(collector):
        await prompt.generate(
            {"company": "Acme", "ticket": "It broke"},
            model=FakeModel(steps=[text_step("ok")]),
        )
    span = collector.last.spans[0]
    assert span.inputs == {"company": "Acme", "ticket": "It broke"}
    assert span.metadata["prompt"]["name"] == "triage"
    assert span.metadata["prompt"]["content_hash"] == prompt.content_hash()


async def test_failed_calls_emit_failed_trace_and_attach_it():
    class ExplodingModel(FakeModel):
        async def do_generate(self, options):
            raise RuntimeError("boom")

    collector = TraceCollector()
    with telemetry(collector):
        with pytest.raises(RuntimeError) as excinfo:
            await generate_text(model=ExplodingModel(), prompt="hi")
    span = collector.last.spans[0]
    assert span.metadata["failed"] is True
    assert span.outputs["error"]["message"] == "boom"
    assert excinfo.value.trace is collector.last


async def test_stream_text_emits_on_finish():
    collector = TraceCollector()
    with telemetry(collector):
        result = stream_text(
            model=FakeModel(steps=[text_step("hello")]), prompt="hi"
        )
        assert await result.text == "hello"
    assert len(collector.traces) == 1
    span = collector.last.spans[0]
    assert [m.role for m in span.messages] == ["user", "assistant"]


async def test_sink_failures_never_break_generation():
    def bad_sink(trace):
        raise RuntimeError("sink down")

    good = TraceCollector()
    configure_telemetry(bad_sink, good)
    result = await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
    assert result.text == "ok"
    assert len(good.traces) == 1


async def test_generate_trace_returns_the_same_trace_sinks_receive():
    ambient = TraceCollector()
    with telemetry(ambient):
        traced = await generate_trace(
            load_prompt(PROMPT_DOC),
            {"company": "Acme", "ticket": "x"},
            model=FakeModel(steps=[text_step("ok")]),
        )
    assert len(ambient.traces) == 1
    assert ambient.last is traced.trace  # single build, identical object


async def test_stream_trace_delivers_to_ambient_sinks_once():
    ambient = TraceCollector()
    with telemetry(ambient):
        result = stream_trace(
            load_prompt(PROMPT_DOC),
            {"company": "Acme", "ticket": "x"},
            model=FakeModel(steps=[text_step("ok")]),
        )
        assert await result.text == "ok"
        trace = await result.trace
    assert len(ambient.traces) == 1
    assert ambient.last is trace


async def test_otel_and_jsonl_sinks_round_trip(tmp_path):
    exported = []
    path = tmp_path / "traces.jsonl"
    configure_telemetry(otel_sink(exported.extend), jsonl_sink(path))
    prompt = load_prompt(PROMPT_DOC)
    await prompt.generate(
        {"company": "Acme", "ticket": "It broke"},
        model=FakeModel(
            steps=[tool_step("t", tool_input={}), text_step("done")]
        ),
        tools={},
    )
    # OTEL round trip: production spans -> replayable history
    trace = trace_from_otel_spans(exported)
    roles = [m.role for m in trace.spans[0].messages]
    assert roles[0] == "system" and "assistant" in roles
    # JSONL sink wrote one line of pai.trace.v1
    line = json.loads(path.read_text().splitlines()[0])
    assert line["schemaVersion"] == "pai.trace.v1"


async def test_trace_context_enriches_raw_calls():
    collector = TraceCollector()
    await generate_text(
        model=FakeModel(steps=[text_step("ok")]),
        prompt="hi",
        telemetry=collector,
        trace_context=TraceContext(
            inputs={"case_id": "c_1"},
            metadata={"tenant": "acme"},
            trace_id="trace_123",
        ),
    )
    span = collector.last.spans[0]
    assert collector.last.id == "trace_123"
    assert span.inputs == {"case_id": "c_1"}
    assert span.metadata["tenant"] == "acme"
