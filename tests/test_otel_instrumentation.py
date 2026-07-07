"""Live OTel instrumentation: real spans via the tracer, nested and replayable."""

import asyncio
import time

import pytest

from pai_sdk import (
    configure_telemetry,
    generate_text,
    load_prompt,
    queued_sink,
    step_count_is,
    stream_text,
    tool,
)
from pai_sdk.integrations.otel import instrument, trace_from_otel_spans, uninstrument

from conftest import FakeModel, text_step, tool_step

pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


class Harness:
    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def spans_by_name(self):
        return {span.name: span for span in self.exporter.get_finished_spans()}


@pytest.fixture()
def otel():
    harness = Harness()
    instrument(tracer_provider=harness.provider)
    yield harness
    uninstrument()
    configure_telemetry()


async def test_generate_text_creates_nested_spans(otel):
    model = FakeModel(steps=[tool_step("lookup", tool_input={}), text_step("done")])
    with otel.provider.get_tracer("request").start_as_current_span("http.request"):
        result = await generate_text(
            model=model,
            prompt="weather?",
            tools={"lookup": tool(lambda: "72F", description="d")},
            stop_when=step_count_is(3),
        )
    assert result.text == "done"

    spans = otel.spans_by_name()
    call, request = spans["pai_sdk.generate_text"], spans["http.request"]
    assert call.parent is not None
    assert call.parent.span_id == request.context.span_id  # nests under caller
    for name in ("pai_sdk.step 0", "pai_sdk.step 1"):
        assert spans[name].parent.span_id == call.context.span_id
    assert spans["pai_sdk.step 0"].attributes["gen_ai.usage.output_tokens"] == 8
    assert call.attributes["gen_ai.usage.input_tokens"] == 30  # totals across steps


async def test_call_span_round_trips_to_replayable_history(otel):
    prompt = load_prompt(
        {
            "name": "triage",
            "input": {"company": "string", "ticket": "string"},
            "system": "You triage for {{company}}.",
            "user": "Ticket: {{ticket}}",
        }
    )
    await prompt.generate(
        {"company": "Acme", "ticket": "It broke"},
        model=FakeModel(steps=[text_step("ok")]),
    )
    payload_spans = [
        {"name": s.name, "attributes": dict(s.attributes)}
        for s in otel.exporter.get_finished_spans()
        if "pai.span.id" in (s.attributes or {})
    ]
    trace = trace_from_otel_spans(payload_spans)
    span = trace.spans[0]
    assert [m.role for m in span.messages] == ["system", "user", "assistant"]
    assert span.inputs == {"company": "Acme", "ticket": "It broke"}

    call = otel.spans_by_name()["pai_sdk.generate_text"]
    assert call.attributes["pai.prompt.name"] == "triage"
    assert call.attributes["gen_ai.operation.name"] == "chat"
    assert call.status.status_code.name == "OK"


async def test_failed_calls_mark_span_error(otel):
    class ExplodingModel(FakeModel):
        async def do_generate(self, options):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await generate_text(model=ExplodingModel(), prompt="hi")
    spans = otel.spans_by_name()
    call = spans["pai_sdk.generate_text"]
    assert call.status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in call.events)
    assert "pai_sdk.step 0" in spans  # dangling step span was closed


async def test_stream_text_instrumented(otel):
    result = stream_text(model=FakeModel(steps=[text_step("hello")]), prompt="hi")
    assert await result.text == "hello"
    call = otel.spans_by_name()["pai_sdk.stream_text"]
    assert call.status.status_code.name == "OK"
    assert "pai.span.messages" in call.attributes


async def test_uninstrument_stops_spans(otel):
    uninstrument()
    await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
    assert otel.exporter.get_finished_spans() == ()


async def test_queued_sink_decouples_latency_and_flushes():
    delivered = []

    def slow_sink(trace):
        time.sleep(0.15)  # blocking on purpose — must not stall the call
        delivered.append(trace)

    queued = queued_sink(slow_sink)
    configure_telemetry(queued)
    try:
        started = time.monotonic()
        await generate_text(model=FakeModel(steps=[text_step("ok")]), prompt="hi")
        elapsed = time.monotonic() - started
        assert elapsed < 0.1, f"call blocked on sink ({elapsed:.3f}s)"
        assert delivered == []  # not yet delivered — off the request path
        await asyncio.wait_for(queued.flush(), timeout=5)
        assert len(delivered) == 1
    finally:
        configure_telemetry()
