"""OpenTelemetry/OpenLLMetry-style converters for pai-sdk traces.

The functions here use plain dictionaries instead of importing OpenTelemetry.
That keeps pai-sdk dependency-free while still giving exporters and
observability pipelines a stable, vendor-neutral bridge.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from ..trace import TRACE_SCHEMA_VERSION, Trace, dump_trace, load_trace


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _get(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _span_name(span: dict[str, Any]) -> str:
    metadata = span.get("metadata") or {}
    prompt = metadata.get("prompt") or {}
    braintrust = metadata.get("braintrust") or {}
    span_attributes = braintrust.get("span_attributes") or {}
    return (
        prompt.get("name")
        or span_attributes.get("name")
        or metadata.get("name")
        or span["id"]
    )


def _usage_attr(usage: dict[str, Any], *keys: str) -> Any:
    return _get(usage, *keys)


def _usage_from_attributes(attributes: dict[str, Any]) -> Optional[dict[str, Any]]:
    input_tokens = _get(
        attributes,
        "pai.usage.input_tokens",
        "gen_ai.usage.input_tokens",
        "llm.usage.prompt_tokens",
    )
    output_tokens = _get(
        attributes,
        "pai.usage.output_tokens",
        "gen_ai.usage.output_tokens",
        "llm.usage.completion_tokens",
    )
    total_tokens = _get(attributes, "pai.usage.total_tokens", "llm.usage.total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def trace_to_otel_spans(
    trace: Trace | dict[str, Any],
    *,
    include_payloads: bool = True,
) -> list[dict[str, Any]]:
    """Convert a pai-sdk Trace into plain OpenTelemetry-style span dicts."""

    data = dump_trace(trace)
    trace_id = data["id"]
    otel_spans: list[dict[str, Any]] = []
    for span in data.get("spans", []):
        metadata = span.get("metadata") or {}
        response = metadata.get("response") or {}
        usage = span.get("usage") or {}
        attributes: dict[str, Any] = {
            "pai.trace.schema_version": data.get("schemaVersion", TRACE_SCHEMA_VERSION),
            "pai.trace.id": trace_id,
            "pai.span.id": span["id"],
            "pai.span.root_span_id": span.get("rootSpanId"),
            "pai.span.input_message_count": metadata.get("input_message_count"),
        }
        if include_payloads:
            attributes.update(
                {
                    "pai.span.inputs": _json_dumps(span.get("inputs") or {}),
                    "pai.span.outputs": _json_dumps(span.get("outputs") or {}),
                    "pai.span.messages": _json_dumps(span.get("messages") or []),
                    "pai.span.metadata": _json_dumps(metadata),
                }
            )
        if response.get("model_id") is not None:
            attributes["gen_ai.response.model"] = response["model_id"]
        input_tokens = _usage_attr(usage, "input_tokens", "inputTokens")
        output_tokens = _usage_attr(usage, "output_tokens", "outputTokens")
        total_tokens = _usage_attr(usage, "total_tokens", "totalTokens")
        if input_tokens is not None:
            attributes["gen_ai.usage.input_tokens"] = input_tokens
            attributes["pai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            attributes["gen_ai.usage.output_tokens"] = output_tokens
            attributes["pai.usage.output_tokens"] = output_tokens
        if total_tokens is not None:
            attributes["pai.usage.total_tokens"] = total_tokens

        otel_spans.append(
            {
                "name": _span_name(span),
                "kind": "CLIENT",
                "context": {"trace_id": trace_id, "span_id": span["id"]},
                "parent_id": span.get("parentSpanId"),
                "attributes": {key: value for key, value in attributes.items() if value is not None},
            }
        )
    return otel_spans


def trace_from_otel_spans(
    spans: Sequence[dict[str, Any]],
    *,
    trace_id: Optional[str] = None,
) -> Trace:
    """Convert plain OpenTelemetry-style span dicts into a pai-sdk Trace."""

    if not spans:
        raise ValueError("Cannot import an empty OpenTelemetry span set.")

    resolved_trace_id = trace_id
    trace_spans: list[dict[str, Any]] = []
    for item in spans:
        attributes = dict(item.get("attributes") or {})
        context = dict(item.get("context") or {})
        span_id = str(
            attributes.get("pai.span.id")
            or context.get("span_id")
            or item.get("span_id")
            or item.get("id")
        )
        current_trace_id = str(
            attributes.get("pai.trace.id")
            or context.get("trace_id")
            or item.get("trace_id")
            or resolved_trace_id
            or span_id
        )
        resolved_trace_id = resolved_trace_id or current_trace_id
        metadata = _json_loads(attributes.get("pai.span.metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        input_message_count = attributes.get("pai.span.input_message_count")
        if input_message_count is not None and "input_message_count" not in metadata:
            metadata["input_message_count"] = int(input_message_count)
        metadata.setdefault(
            "otel",
            {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "attributes": attributes,
            },
        )

        trace_spans.append(
            {
                "id": span_id,
                "rootSpanId": str(
                    attributes.get("pai.span.root_span_id") or current_trace_id
                ),
                "parentSpanId": item.get("parent_id") or item.get("parent_span_id"),
                "inputs": _json_loads(attributes.get("pai.span.inputs"), {}),
                "outputs": _json_loads(attributes.get("pai.span.outputs"), {}),
                "messages": _json_loads(attributes.get("pai.span.messages"), []),
                "usage": _usage_from_attributes(attributes),
                "metadata": metadata,
            }
        )

    return load_trace(
        {
            "schemaVersion": TRACE_SCHEMA_VERSION,
            "id": resolved_trace_id,
            "spans": trace_spans,
        }
    )
