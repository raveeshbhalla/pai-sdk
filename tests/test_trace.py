from __future__ import annotations

from pai_sdk import (
    dump_messages,
    dump_trace_json,
    generate_trace,
    LanguageModel,
    load_messages,
    load_prompt,
    load_trace,
    ProviderResult,
    replay_span,
    Span,
    span_input_messages,
    span_response_messages,
    step_count_is,
    stream_trace,
    Trace,
    tool,
)
from pai_sdk.messages import ToolModelMessage

from conftest import FakeModel, text_step, tool_step


def _triage_prompt():
    return load_prompt(
        {
            "name": "trace-triage",
            "messages": [
                {
                    "id": "system",
                    "role": "system",
                    "template": "You triage tickets for {{company}}.",
                },
                {
                    "id": "ticket",
                    "role": "user",
                    "template": "Ticket: {{ticket}}",
                },
            ],
            "output": {"urgency": ["low", "high"]},
        }
    )


async def test_prompt_generate_trace_builds_structured_span():
    prompt = _triage_prompt()
    model = FakeModel(steps=[text_step('{"urgency": "high"}')])

    traced = await prompt.generate_trace(
        {"company": "Acme", "ticket": "Login fails"},
        model=model,
        trace_id="trace_fixed",
        span_id="span_fixed",
    )

    assert traced.text == '{"urgency": "high"}'
    assert traced.output == {"urgency": "high"}
    assert traced.trace.id == "trace_fixed"

    [span] = traced.trace.spans
    assert span.id == "span_fixed"
    assert span.inputs == {"company": "Acme", "ticket": "Login fails"}
    assert span.outputs["object"] == {"urgency": "high"}
    assert span.usage.total_tokens == 15
    assert [message.role for message in span.messages] == ["system", "user", "assistant"]
    assert span.metadata["prompt"]["name"] == "trace-triage"
    assert span.metadata["prompt"]["message_ids"] == ["system", "ticket"]

    dumped = span.to_dict()["messages"]
    assert load_messages(dumped)[0].content == "You triage tickets for Acme."
    assert traced.trace.to_dict()["spans"][0]["rootSpanId"] == "trace_fixed"


async def test_generate_trace_helper_captures_tool_messages_in_one_span():
    prompt = load_prompt(
        {
            "name": "trace-with-tool",
            "messages": [{"id": "user", "role": "user", "template": "{{question}}"}],
            "tools": {
                "lookup": {
                    "description": "Look up context.",
                    "input": {"q": "string"},
                }
            },
            "max_steps": 4,
        }
    )
    model = FakeModel(
        steps=[
            tool_step("lookup", tool_input={"q": "account"}),
            text_step("done"),
        ]
    )

    traced = await generate_trace(
        prompt,
        {"question": "Check the account."},
        model=model,
        handlers={"lookup": lambda input: {"found": input["q"]}},
    )

    [span] = traced.trace.spans
    assert traced.text == "done"
    assert [message.role for message in span.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert isinstance(span.messages[2], ToolModelMessage)
    assert span.messages[2].content[0].output.value == {"found": "account"}
    assert span.outputs["tool_calls"][0].tool_name == "lookup"
    assert span.outputs["tool_results"][0].output == {"found": "account"}
    assert span.metadata["step_finish_reasons"] == ["tool-calls", "stop"]

    # The span carries the same replayable ModelMessage[] shape as the serializer.
    assert dump_messages(span.messages)[2]["content"][0]["toolName"] == "lookup"


async def test_trace_dump_load_round_trips_usage_and_message_boundary():
    prompt = _triage_prompt()
    model = FakeModel(steps=[text_step('{"urgency": "low"}')])

    traced = await prompt.generate_trace({"company": "Acme", "ticket": "FYI"}, model=model)

    loaded = load_trace(dump_trace_json(traced.trace))
    [span] = loaded.spans
    assert loaded.id == traced.trace.id
    assert span.usage.total_tokens == 15
    assert span.metadata["input_message_count"] == 2
    assert [message.role for message in span_input_messages(span)] == ["system", "user"]
    assert [message.role for message in span_response_messages(span)] == ["assistant"]


async def test_load_trace_accepts_ai_sdk_camel_usage_shape():
    trace = load_trace(
        {
            "id": "trace_external",
            "spans": [
                {
                    "id": "span_external",
                    "rootSpanId": "span_root",
                    "inputs": {"question": "hi"},
                    "outputs": {"text": "hello"},
                    "messages": [{"role": "user", "content": "hi"}],
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 2,
                        "totalTokens": 3,
                        "inputTokenDetails": {"cacheReadTokens": 1},
                    },
                    "metadata": {"input_message_count": 1},
                }
            ],
        }
    )

    [span] = trace.spans
    assert span.root_span_id == "span_root"
    assert span.usage.total_tokens == 3
    assert span.usage.input_token_details.cache_read_tokens == 1
    assert span_input_messages(span)[0].content == "hi"


def test_multi_span_trace_round_trip_preserves_relationships():
    messages = load_messages([{"role": "user", "content": "hi"}])
    trace = Trace(
        id="span_root",
        spans=[
            Span(
                id="span_root",
                root_span_id="span_root",
                inputs={"question": "hi"},
                outputs={"text": "hello"},
                messages=messages,
                metadata={"input_message_count": 1},
            ),
            Span(
                id="span_child",
                root_span_id="span_root",
                parent_span_id="span_root",
                inputs={"tool": "lookup"},
                outputs={"found": True},
                messages=messages,
                metadata={"input_message_count": 1},
            ),
        ],
    )

    loaded = load_trace(dump_trace_json(trace))

    assert [span.id for span in loaded.spans] == ["span_root", "span_child"]
    assert loaded.spans[1].root_span_id == "span_root"
    assert loaded.spans[1].parent_span_id == "span_root"


async def test_generate_trace_attaches_failed_trace_to_exception():
    class ExplodingModel(FakeModel):
        async def do_generate(self, options):
            self.calls.append(options)
            raise RuntimeError("boom")

    model = ExplodingModel()

    try:
        await generate_trace(model=model, prompt="hi", trace_id="trace_error")
        raised = False
    except RuntimeError as exc:
        raised = True
        trace = exc.trace

    assert raised
    assert trace.id == "trace_error"
    [span] = trace.spans
    assert span.id == "trace_error"
    assert span.root_span_id == "trace_error"
    assert span.outputs["error"]["type"] == "RuntimeError"
    assert span.metadata["failed"] is True
    assert span.metadata["error"]["message"] == "boom"
    assert [message.role for message in span.messages] == ["user"]


async def test_stream_trace_attaches_failed_trace_to_exception():
    class ExplodingStreamModel(LanguageModel):
        provider = "fake"
        model_id = "fake-stream-error"

        async def do_generate(self, options) -> ProviderResult:
            raise NotImplementedError

        async def do_stream(self, options):
            raise RuntimeError("stream boom")
            yield  # pragma: no cover

    traced = stream_trace(
        model=ExplodingStreamModel(),
        prompt="hi",
        trace_id="trace_stream_error",
    )
    parts = [part async for part in traced.full_stream]
    assert parts[-1].type == "error"

    try:
        await traced.trace
        raised = False
    except RuntimeError as exc:
        raised = True
        trace = exc.trace

    assert raised
    assert trace.id == "trace_stream_error"
    [span] = trace.spans
    assert span.root_span_id == "trace_stream_error"
    assert span.outputs["error"]["message"] == "stream boom"
    assert span.metadata["failed"] is True
    assert [message.role for message in span.messages] == ["user"]


async def test_promptless_generate_trace_and_replay_span():
    model = FakeModel(steps=[text_step("first")])

    traced = await generate_trace(
        model=model,
        prompt="Say hello.",
        inputs={"instruction": "Say hello."},
        trace_id="trace_plain",
    )

    [span] = traced.trace.spans
    assert traced.text == "first"
    assert span.inputs == {"instruction": "Say hello."}
    assert [message.role for message in span.messages] == ["user", "assistant"]

    replay_model = FakeModel(steps=[text_step("second")])
    replayed = await replay_span(span, model=replay_model)
    assert replayed.text == "second"
    assert [message.role for message in replay_model.calls[0].prompt] == ["user"]


async def test_stream_trace_captures_completed_stream():
    model = FakeModel(steps=[text_step("streamed")])

    traced = stream_trace(model=model, prompt="stream please", trace_id="trace_stream")
    chunks = [chunk async for chunk in traced.text_stream]
    trace = await traced.trace

    assert "".join(chunks) == "streamed"
    assert trace.id == "trace_stream"
    [span] = trace.spans
    assert span.outputs["text"] == "streamed"
    assert span.usage.total_tokens == 15
    assert span.metadata["input_message_count"] == 1
    assert [message.role for message in span.messages] == ["user", "assistant"]


async def test_prompt_stream_trace_captures_prompt_metadata():
    prompt = _triage_prompt()
    model = FakeModel(steps=[text_step('{"urgency": "high"}')])

    traced = prompt.stream_trace({"company": "Acme", "ticket": "down"}, model=model)
    assert await traced.text == '{"urgency": "high"}'
    trace = await traced.trace

    [span] = trace.spans
    assert span.outputs["object"] == {"urgency": "high"}
    assert span.metadata["prompt"]["name"] == "trace-triage"
