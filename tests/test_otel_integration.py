from __future__ import annotations

from pai_sdk import Span, Trace, Usage, load_messages
from pai_sdk.integrations.otel import trace_from_otel_spans, trace_to_otel_spans


def test_trace_to_otel_spans_round_trips_lossless_payloads():
    trace = Trace(
        id="trace_otel",
        spans=[
            Span(
                id="span_otel",
                root_span_id="trace_otel",
                inputs={"question": "hi"},
                outputs={"text": "hello"},
                messages=load_messages(
                    [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                ),
                usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7),
                metadata={
                    "input_message_count": 1,
                    "response": {"model_id": "provider/model"},
                },
            )
        ],
    )

    otel_spans = trace_to_otel_spans(trace)

    assert otel_spans[0]["context"] == {
        "trace_id": "trace_otel",
        "span_id": "span_otel",
    }
    attrs = otel_spans[0]["attributes"]
    assert attrs["pai.trace.schema_version"] == "pai.trace.v1"
    assert attrs["gen_ai.response.model"] == "provider/model"
    assert attrs["gen_ai.usage.input_tokens"] == 3

    restored = trace_from_otel_spans(otel_spans)

    assert restored.id == trace.id
    [span] = restored.spans
    assert span.id == "span_otel"
    assert span.inputs == {"question": "hi"}
    assert span.outputs == {"text": "hello"}
    assert [message.role for message in span.messages] == ["user", "assistant"]
    assert span.usage.total_tokens == 7
