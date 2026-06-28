"""Import Braintrust-style log rows into pai-sdk traces.

Braintrust project-log exports and SQL rows are provider/application dependent,
but the common useful fields are stable enough to normalize:

- row/span ids (`id`, `root_span_id`, `parent_span_id`)
- `span_attributes.name`
- `input` and `output`
- `metadata`, `scores`, and `metrics`

This module keeps the import best-effort. It preserves the original row data in
span metadata and only reconstructs `ModelMessage[]` when the row carries
message-shaped content.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .messages import (
    AssistantModelMessage,
    JsonOutput,
    ModelMessage,
    TextOutput,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
)
from .results import InputTokenDetails, OutputTokenDetails, Usage
from .serialize import load_messages
from .trace import Span, Trace


def _get(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "".join(parts)
    return str(value)


def _convert_user_content(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return _text_from_content(value)
    converted: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            converted.append({"type": "text", "text": item})
        elif isinstance(item, dict) and item.get("type") == "image_url":
            image_url = _as_dict(item.get("image_url"))
            converted.append({"type": "image", "image": image_url.get("url", "")})
        elif isinstance(item, dict):
            converted.append(item)
    return converted or ""


def _tool_call_part(call: dict[str, Any]) -> ToolCallPart:
    function = _as_dict(call.get("function"))
    tool_name = call.get("toolName") or call.get("tool_name") or function.get("name")
    tool_input = call.get("input")
    if tool_input is None:
        tool_input = function.get("arguments")
    return ToolCallPart(
        tool_call_id=str(call.get("id") or call.get("toolCallId") or call.get("tool_call_id")),
        tool_name=str(tool_name or "tool"),
        input=_parse_json_maybe(tool_input),
    )


def braintrust_message_to_model_message(message: dict[str, Any]) -> ModelMessage:
    """Convert a common Braintrust/OpenAI/AI-SDK message dict to ModelMessage."""

    role = message.get("role")
    if role == "assistant" and message.get("tool_calls"):
        parts: list[Any] = []
        text = _text_from_content(message.get("content"))
        if text:
            parts.append(TextPart(text=text))
        parts.extend(_tool_call_part(call) for call in _as_list(message.get("tool_calls")))
        return AssistantModelMessage(content=parts)

    if role == "tool":
        content = _parse_json_maybe(message.get("content"))
        output = JsonOutput(value=content) if not isinstance(content, str) else TextOutput(value=content)
        return ToolModelMessage(
            content=[
                ToolResultPart(
                    tool_call_id=str(
                        message.get("toolCallId")
                        or message.get("tool_call_id")
                        or message.get("id")
                    ),
                    tool_name=str(message.get("name") or message.get("toolName") or "tool"),
                    output=output,
                )
            ]
        )

    if role == "user":
        return load_messages([{"role": "user", "content": _convert_user_content(message.get("content"))}])[0]

    if role == "assistant":
        return load_messages([{"role": "assistant", "content": _text_from_content(message.get("content"))}])[0]

    if role == "system":
        return load_messages([{"role": "system", "content": _text_from_content(message.get("content"))}])[0]

    return load_messages([message])[0]


def braintrust_messages_to_model_messages(messages: Sequence[Any]) -> list[ModelMessage]:
    """Convert a Braintrust/OpenAI/AI-SDK message list to ModelMessage values."""

    converted: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, dict):
            converted.append(braintrust_message_to_model_message(message))
    return converted


def _messages_from_value(value: Any) -> list[ModelMessage]:
    if isinstance(value, list):
        return braintrust_messages_to_model_messages(value)
    if not isinstance(value, dict):
        return []
    for key in ("messages", "prompt", "input_messages", "inputMessages"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return braintrust_messages_to_model_messages(candidate)
    return []


def _assistant_messages_from_output(output: Any) -> list[ModelMessage]:
    if isinstance(output, dict):
        for key in ("messages", "response_messages", "responseMessages"):
            candidate = output.get(key)
            if isinstance(candidate, list):
                return braintrust_messages_to_model_messages(candidate)
        response = output.get("response")
        if isinstance(response, dict):
            messages = _messages_from_value(response)
            if messages:
                return messages
        text = output.get("text") or output.get("content")
        if text is not None:
            return [AssistantModelMessage(content=_text_from_content(text))]
    elif isinstance(output, str):
        return [AssistantModelMessage(content=output)]
    return []


def _usage_from_metrics(metrics: dict[str, Any]) -> Optional[Usage]:
    if not metrics:
        return None
    input_tokens = _get(metrics, "prompt_tokens", "input_tokens", "inputTokens")
    output_tokens = _get(metrics, "completion_tokens", "output_tokens", "outputTokens")
    total_tokens = _get(metrics, "tokens", "total_tokens", "totalTokens")
    cached = _get(metrics, "prompt_cached_tokens", "cached_input_tokens", "cachedInputTokens")
    cache_write = _get(metrics, "prompt_cache_creation_tokens")
    reasoning = _get(
        metrics,
        "completion_reasoning_tokens",
        "reasoning_tokens",
        "reasoningTokens",
    )
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning,
        cached_input_tokens=cached,
        input_token_details=InputTokenDetails(
            cache_read_tokens=cached,
            cache_write_tokens=cache_write,
        )
        if cached is not None or cache_write is not None
        else None,
        output_token_details=OutputTokenDetails(reasoning_tokens=reasoning)
        if reasoning is not None
        else None,
    )


def span_from_braintrust_row(row: dict[str, Any], *, trace_id: Optional[str] = None) -> Span:
    """Convert one Braintrust project-log row into a pai-sdk Span."""

    row_id = str(_get(row, "id", "span_id", "spanId") or "")
    root_span_id = str(
        _get(row, "root_span_id", "rootSpanId", "root_id", "rootId") or trace_id or row_id
    )
    span_attributes = _as_dict(_get(row, "span_attributes", "spanAttributes"))
    input_value = row.get("input")
    output_value = row.get("output")
    input_messages = _messages_from_value(input_value)
    output_messages = _assistant_messages_from_output(output_value)
    metadata = {
        "braintrust": {
            "id": row_id,
            "span_attributes": span_attributes,
            "metadata": _as_dict(row.get("metadata")),
            "scores": _as_dict(row.get("scores")),
            "metrics": _as_dict(row.get("metrics")),
        },
        "input_message_count": len(input_messages),
    }
    return Span(
        id=row_id,
        root_span_id=root_span_id,
        parent_span_id=_get(row, "parent_span_id", "parentSpanId"),
        inputs=input_value if isinstance(input_value, dict) else {"input": input_value},
        outputs=output_value if isinstance(output_value, dict) else {"output": output_value},
        messages=[*input_messages, *output_messages],
        usage=_usage_from_metrics(_as_dict(row.get("metrics"))),
        metadata=metadata,
    )


def trace_from_braintrust_rows(
    rows: Sequence[dict[str, Any]],
    *,
    trace_id: Optional[str] = None,
) -> Trace:
    """Convert Braintrust project-log rows into a pai-sdk Trace."""

    if not rows:
        raise ValueError("Cannot import an empty Braintrust row set.")
    resolved_trace_id = trace_id or str(
        _get(rows[0], "root_span_id", "rootSpanId", "id", "span_id", "spanId")
    )
    spans = [span_from_braintrust_row(row, trace_id=resolved_trace_id) for row in rows]
    return Trace(id=resolved_trace_id, spans=spans)
