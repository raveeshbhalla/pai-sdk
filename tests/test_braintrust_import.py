from __future__ import annotations

from pai_sdk import (
    AssistantModelMessage,
    ToolModelMessage,
    braintrust_messages_to_model_messages,
    span_input_messages,
    span_response_messages,
    trace_from_braintrust_rows,
)


def test_braintrust_messages_convert_openai_tool_shapes():
    messages = braintrust_messages_to_model_messages(
        [
            {"role": "system", "content": "You help."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look this up"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"abc"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lookup",
                "content": '{"answer": 42}',
            },
        ]
    )

    assert [message.role for message in messages] == ["system", "user", "assistant", "tool"]
    assistant = messages[2]
    assert isinstance(assistant, AssistantModelMessage)
    assert assistant.content[1].tool_name == "lookup"
    assert assistant.content[1].input == {"q": "abc"}
    tool = messages[3]
    assert isinstance(tool, ToolModelMessage)
    assert tool.content[0].output.value == {"answer": 42}


def test_trace_from_braintrust_rows_imports_messages_usage_and_metadata():
    rows = [
        {
            "id": "span_root",
            "root_span_id": "trace_1",
            "span_attributes": {"name": "streamText"},
            "input": {
                "messages": [
                    {"role": "system", "content": "You triage tickets."},
                    {"role": "user", "content": "Ticket: login is down"},
                ]
            },
            "output": {"text": "High urgency."},
            "metadata": {"product": "support"},
            "scores": {"correctness": 1},
            "metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "tokens": 14,
                "prompt_cached_tokens": 3,
                "completion_reasoning_tokens": 1,
            },
        },
        {
            "id": "span_child",
            "root_span_id": "trace_1",
            "parent_span_id": "span_root",
            "span_attributes": {"name": "lookup"},
            "input": {"query": "login"},
            "output": {"found": True},
            "metadata": {"tool": "lookup"},
            "metrics": {},
        },
    ]

    trace = trace_from_braintrust_rows(rows)

    assert trace.id == "trace_1"
    assert [span.id for span in trace.spans] == ["span_root", "span_child"]
    root = trace.spans[0]
    assert [message.role for message in root.messages] == ["system", "user", "assistant"]
    assert root.usage.total_tokens == 14
    assert root.usage.cached_input_tokens == 3
    assert root.usage.output_token_details.reasoning_tokens == 1
    assert root.metadata["braintrust"]["span_attributes"]["name"] == "streamText"
    assert root.metadata["braintrust"]["scores"] == {"correctness": 1}
    assert [message.role for message in span_input_messages(root)] == ["system", "user"]
    assert [message.role for message in span_response_messages(root)] == ["assistant"]

    child = trace.spans[1]
    assert child.parent_span_id == "span_root"
    assert child.messages == []
    assert child.inputs == {"query": "login"}
    assert child.outputs == {"found": True}
