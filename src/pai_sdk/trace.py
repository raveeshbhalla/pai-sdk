"""Trace-backed structured history helpers.

The trace layer joins structured inputs/outputs with the provider-near
`ModelMessage[]` history produced by a generation call. It is intentionally a
thin wrapper over the existing prompt and generation APIs: inference still runs
through `generate_text`, and replay still uses ordinary `ModelMessage` values.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import BaseModel

from .messages import ModelMessage
from .results import (
    GenerateTextResult,
    InputTokenDetails,
    OutputTokenDetails,
    ResponseMetadata,
    Usage,
)
from .serialize import dump_messages, load_messages

TRACE_SCHEMA_VERSION = "pai.trace.v1"

TRACE_WIRE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "pai-sdk trace",
    "type": "object",
    "required": ["schemaVersion", "id", "spans"],
    "additionalProperties": True,
    "properties": {
        "schemaVersion": {"const": TRACE_SCHEMA_VERSION},
        "id": {"type": "string"},
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "rootSpanId",
                    "inputs",
                    "outputs",
                    "messages",
                    "metadata",
                ],
                "additionalProperties": True,
                "properties": {
                    "id": {"type": "string"},
                    "rootSpanId": {"type": "string"},
                    "parentSpanId": {"type": ["string", "null"]},
                    "inputs": {"type": "object"},
                    "outputs": {"type": "object"},
                    "messages": {"type": "array"},
                    "usage": {"type": ["object", "null"]},
                    "metadata": {"type": "object"},
                },
            },
        },
    },
}

TracePath = tuple[Union[str, int], ...]
TraceRedactor = Callable[[TracePath, Any], Any]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class Span:
    """A structured history row plus its replayable message transcript."""

    id: str
    root_span_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    messages: list[ModelMessage]
    parent_span_id: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rootSpanId": self.root_span_id,
            "parentSpanId": self.parent_span_id,
            "inputs": _jsonable(self.inputs),
            "outputs": _jsonable(self.outputs),
            "messages": dump_messages(self.messages),
            "usage": _jsonable(self.usage),
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class Trace:
    """A replayable trace composed of one or more spans."""

    id: str
    spans: list[Span] = field(default_factory=list)
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "id": self.id,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass
class GenerateTraceResult:
    """A `GenerateTextResult` plus its trace.

    Attribute access falls through to the wrapped generation result so callers
    can continue to use `result.text`, `result.output`, `result.response`, etc.
    """

    result: GenerateTextResult
    trace: Trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)


@dataclass
class StreamTraceResult:
    """A `StreamTextResult` plus an awaitable trace.

    Attribute access falls through to the wrapped stream result, so callers can
    continue to consume `text_stream`, `full_stream`, `text`, `usage`, etc. The
    trace is available once the stream has completed:

        result = stream_trace(model=..., prompt="...")
        async for delta in result.text_stream:
            ...
        trace = await result.trace
    """

    result: Any
    input_messages: list[ModelMessage]
    inputs: dict[str, Any]
    outputs: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    root_span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    sinks: tuple = ()
    _trace: Optional[Trace] = field(default=None, init=False, repr=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    @property
    def trace(self) -> Awaitable[Trace]:
        return self._resolve_trace()

    async def _resolve_trace(self) -> Trace:
        if self._trace is not None:
            return self._trace
        try:
            outputs = self.outputs
            if outputs is None:
                outputs = await outputs_from_stream_result(self.result)
            metadata = {
                **await metadata_from_stream_result(self.result),
                **self.metadata,
            }
            self._trace = build_trace_from_messages(
                inputs=self.inputs,
                input_messages=self.input_messages,
                response_messages=(await self.result.response).messages,
                usage=await self.result.total_usage,
                outputs=outputs,
                metadata=metadata,
                trace_id=self.trace_id,
                span_id=self.span_id,
                root_span_id=self.root_span_id,
                parent_span_id=self.parent_span_id,
            )
        except BaseException as exc:
            self._trace = build_failed_trace(
                inputs=self.inputs,
                input_messages=self.input_messages,
                response_messages=list(getattr(self.result, "_generated_messages", [])),
                usage=getattr(self.result, "_total_usage", None),
                error=exc,
                metadata=self.metadata,
                trace_id=self.trace_id,
                span_id=self.span_id,
                root_span_id=self.root_span_id,
                parent_span_id=self.parent_span_id,
            )
            from .telemetry import emit_trace

            await emit_trace(self._trace, self.sinks)
            raise _attach_trace(exc, self._trace) from exc
        from .telemetry import emit_trace

        await emit_trace(self._trace, self.sinks)
        return self._trace


def outputs_from_result(result: GenerateTextResult) -> dict[str, Any]:
    """Build default span outputs from a generation result."""

    outputs: dict[str, Any] = {
        "text": result.text,
        "finish_reason": result.finish_reason,
    }
    if result.output is not None:
        outputs["object"] = result.output
    tool_calls = [call for step in result.steps for call in step.tool_calls]
    tool_results = [item for step in result.steps for item in step.tool_results]
    if tool_calls:
        outputs["tool_calls"] = tool_calls
    if tool_results:
        outputs["tool_results"] = tool_results
    return outputs


def metadata_from_result(result: GenerateTextResult) -> dict[str, Any]:
    """Build default non-semantic span metadata from a generation result."""

    response = result.response
    return {
        "response": {
            "id": response.id,
            "model_id": response.model_id,
            "timestamp": response.timestamp,
            "headers": response.headers,
        },
        "finish_reason": result.finish_reason,
        "raw_finish_reason": result.raw_finish_reason,
        "step_finish_reasons": [step.finish_reason for step in result.steps],
        "step_request_messages": [
            dump_messages(step.request_messages) for step in result.steps
        ],
        "warnings": list(result.warnings),
        "provider_metadata": result.provider_metadata,
    }


def _error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    for name in ("status_code", "response_body", "is_retryable", "budget", "reason"):
        value = getattr(exc, name, None)
        if value is not None:
            payload[name] = _jsonable(value)
    return payload


async def outputs_from_stream_result(result: Any) -> dict[str, Any]:
    """Build default span outputs from a completed stream result."""

    outputs: dict[str, Any] = {
        "text": await result.text,
        "finish_reason": await result.finish_reason,
    }
    parsed_output = await result.output
    if parsed_output is not None:
        outputs["object"] = parsed_output
    steps = await result.all_steps
    tool_calls = [call for step in steps for call in step.tool_calls]
    tool_results = [item for step in steps for item in step.tool_results]
    if tool_calls:
        outputs["tool_calls"] = tool_calls
    if tool_results:
        outputs["tool_results"] = tool_results
    return outputs


async def metadata_from_stream_result(result: Any) -> dict[str, Any]:
    """Build default non-semantic span metadata from a stream result."""

    response: ResponseMetadata = await result.response
    steps = await result.all_steps
    final = steps[-1] if steps else None
    return {
        "response": {
            "id": response.id,
            "model_id": response.model_id,
            "timestamp": response.timestamp,
            "headers": response.headers,
        },
        "finish_reason": await result.finish_reason,
        "raw_finish_reason": final.raw_finish_reason if final is not None else None,
        "step_finish_reasons": [step.finish_reason for step in steps],
        "step_request_messages": [
            dump_messages(step.request_messages) for step in steps
        ],
        "warnings": [warning for step in steps for warning in step.warnings],
        "provider_metadata": final.provider_metadata if final is not None else None,
    }


def build_trace_from_messages(
    *,
    inputs: dict[str, Any],
    input_messages: list[ModelMessage],
    response_messages: list[ModelMessage],
    usage: Optional[Usage] = None,
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    root_span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Trace:
    """Create a single-span trace from rendered input and generated messages."""

    trace_id = trace_id or root_span_id or _new_id("trace")
    root_span_id = root_span_id or trace_id
    span_id = span_id or (root_span_id if parent_span_id is None else _new_id("span"))
    span = Span(
        id=span_id,
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
        inputs=inputs,
        outputs=outputs or {},
        messages=[*input_messages, *response_messages],
        usage=usage,
        metadata={
            "input_message_count": len(input_messages),
            **(metadata or {}),
        },
    )
    return Trace(id=trace_id, spans=[span])


def build_failed_trace(
    *,
    inputs: dict[str, Any],
    input_messages: list[ModelMessage],
    error: BaseException,
    response_messages: Optional[list[ModelMessage]] = None,
    usage: Optional[Usage] = None,
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    root_span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Trace:
    """Create a single-span trace for a failed generation attempt."""

    error_payload = _error_payload(error)
    return build_trace_from_messages(
        inputs=inputs,
        input_messages=input_messages,
        response_messages=response_messages or [],
        usage=usage,
        outputs=outputs if outputs is not None else {"error": error_payload},
        metadata={
            "failed": True,
            "error": error_payload,
            **(metadata or {}),
        },
        trace_id=trace_id,
        span_id=span_id,
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
    )


def _attach_trace(error: BaseException, trace: Trace) -> BaseException:
    setattr(error, "trace", trace)
    return error


def build_trace(
    *,
    inputs: dict[str, Any],
    result: GenerateTextResult,
    input_messages: list[ModelMessage],
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    root_span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Trace:
    """Create a single-span trace from a completed generation call."""

    return build_trace_from_messages(
        inputs=inputs,
        input_messages=input_messages,
        response_messages=result.response.messages,
        usage=result.total_usage,
        outputs=outputs if outputs is not None else outputs_from_result(result),
        metadata={
            **metadata_from_result(result),
            **(metadata or {}),
        },
        trace_id=trace_id,
        span_id=span_id,
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
    )


def _redact_value(value: Any, redactor: TraceRedactor, path: TracePath) -> Any:
    redacted = redactor(path, value)
    if isinstance(redacted, dict):
        return {
            key: _redact_value(item, redactor, (*path, str(key)))
            for key, item in redacted.items()
        }
    if isinstance(redacted, list):
        return [
            _redact_value(item, redactor, (*path, index))
            for index, item in enumerate(redacted)
        ]
    return redacted


def redact_trace(trace: Union[Trace, dict[str, Any]], redactor: TraceRedactor) -> dict[str, Any]:
    """Return a redacted JSON-able trace dict without mutating the input."""

    return _redact_value(dump_trace(trace), redactor, ())


def redact_trace_content(
    trace: Union[Trace, dict[str, Any]],
    *,
    replacement: str = "[redacted]",
) -> dict[str, Any]:
    """Redact common prompt/response text fields before export."""

    sensitive_keys = {"content", "text", "image", "data", "inputs", "outputs"}

    def redactor(path: TracePath, value: Any) -> Any:
        key = path[-1] if path else None
        if key in ("inputs", "outputs") and isinstance(value, dict):
            return {"redacted": True}
        # Provider response headers can carry request-scoped tokens/cookies.
        if key == "headers" and isinstance(value, dict):
            return {"redacted": True}
        if key in sensitive_keys and isinstance(value, str):
            return replacement
        return value

    return redact_trace(trace, redactor)


def dump_trace(
    trace: Union[Trace, dict[str, Any]],
    *,
    redactor: Optional[TraceRedactor] = None,
) -> dict[str, Any]:
    """Serialize a trace to a JSON-able dict."""

    if isinstance(trace, Trace):
        data = trace.to_dict()
    else:
        data = _jsonable(trace)
        data.setdefault("schemaVersion", TRACE_SCHEMA_VERSION)
    if redactor is not None:
        return _redact_value(data, redactor, ())
    return data


def dump_trace_json(
    trace: Union[Trace, dict[str, Any]],
    *,
    indent: int | None = None,
    redactor: Optional[TraceRedactor] = None,
) -> str:
    """Serialize a trace to a JSON string."""

    return json.dumps(dump_trace(trace, redactor=redactor), indent=indent, ensure_ascii=False)


def _get(data: dict[str, Any], snake: str, camel: str) -> Any:
    if snake in data:
        return data[snake]
    return data.get(camel)


def _load_input_token_details(value: Any) -> Optional[InputTokenDetails]:
    if value is None or isinstance(value, InputTokenDetails):
        return value
    if not isinstance(value, dict):
        return None
    return InputTokenDetails(
        no_cache_tokens=_get(value, "no_cache_tokens", "noCacheTokens"),
        cache_read_tokens=_get(value, "cache_read_tokens", "cacheReadTokens"),
        cache_write_tokens=_get(value, "cache_write_tokens", "cacheWriteTokens"),
    )


def _load_output_token_details(value: Any) -> Optional[OutputTokenDetails]:
    if value is None or isinstance(value, OutputTokenDetails):
        return value
    if not isinstance(value, dict):
        return None
    return OutputTokenDetails(
        text_tokens=_get(value, "text_tokens", "textTokens"),
        reasoning_tokens=_get(value, "reasoning_tokens", "reasoningTokens"),
    )


def _load_usage(value: Any) -> Optional[Usage]:
    if value is None or isinstance(value, Usage):
        return value
    if not isinstance(value, dict):
        return None
    return Usage(
        input_tokens=_get(value, "input_tokens", "inputTokens"),
        output_tokens=_get(value, "output_tokens", "outputTokens"),
        total_tokens=_get(value, "total_tokens", "totalTokens"),
        reasoning_tokens=_get(value, "reasoning_tokens", "reasoningTokens"),
        cached_input_tokens=_get(value, "cached_input_tokens", "cachedInputTokens"),
        input_token_details=_load_input_token_details(
            _get(value, "input_token_details", "inputTokenDetails")
        ),
        output_token_details=_load_output_token_details(
            _get(value, "output_token_details", "outputTokenDetails")
        ),
    )


def _load_span(data: dict[str, Any]) -> Span:
    span_id = data["id"]
    return Span(
        id=span_id,
        root_span_id=_get(data, "root_span_id", "rootSpanId") or span_id,
        parent_span_id=_get(data, "parent_span_id", "parentSpanId"),
        inputs=dict(data.get("inputs") or {}),
        outputs=dict(data.get("outputs") or {}),
        messages=load_messages(data.get("messages") or []),
        usage=_load_usage(data.get("usage")),
        metadata=dict(data.get("metadata") or {}),
    )


def load_trace(data: Union[str, dict[str, Any]]) -> Trace:
    """Load a trace dict or JSON string back into typed Trace/Span objects."""

    if isinstance(data, str):
        data = json.loads(data)
    spans = [_load_span(span) for span in data.get("spans", [])]
    schema_version = _get(data, "schema_version", "schemaVersion") or TRACE_SCHEMA_VERSION
    return Trace(id=data["id"], spans=spans, schema_version=schema_version)


def span_input_messages(span: Span) -> list[ModelMessage]:
    """Return the input prefix for semantic replay when the boundary is known."""

    count = span.metadata.get("input_message_count")
    if not isinstance(count, int) or count < 0 or count > len(span.messages):
        raise ValueError(
            "Span does not include metadata.input_message_count; cannot infer "
            "which messages were inputs versus generated responses."
        )
    return span.messages[:count]


def span_response_messages(span: Span) -> list[ModelMessage]:
    """Return generated response messages when the input boundary is known."""

    count = span.metadata.get("input_message_count")
    if not isinstance(count, int) or count < 0 or count > len(span.messages):
        raise ValueError(
            "Span does not include metadata.input_message_count; cannot infer "
            "which messages were inputs versus generated responses."
        )
    return span.messages[count:]


async def replay_span(span: Span, *, model: Any, **overrides: Any) -> GenerateTextResult:
    """Rerun a span semantically from its recorded input message prefix."""

    from .generate import generate_text

    return await generate_text(
        model=model,
        messages=span_input_messages(span),
        **overrides,
    )


async def replay_trace(
    trace: Trace,
    *,
    model: Any,
    span_id: Optional[str] = None,
    **overrides: Any,
) -> GenerateTextResult:
    """Rerun one span from a trace semantically from its input prefix."""

    if span_id is None:
        if not trace.spans:
            raise ValueError("Trace has no spans to replay.")
        span = trace.spans[-1]
    else:
        span = next((item for item in trace.spans if item.id == span_id), None)
        if span is None:
            raise ValueError(f"No span with id '{span_id}'.")
    return await replay_span(span, model=model, **overrides)


def span_feedback(
    span: Span,
    *,
    include_transcript: bool = True,
    max_transcript_chars: int = 6000,
) -> dict[str, str]:
    """Actionable side information (ASI) for reflective optimizers.

    Turns a span into the `{label: text}` diagnostic dict a GEPA
    `optimize_anything` evaluator returns alongside the score — the textual
    "gradient" a reflective proposer reads to explain WHY a candidate scored
    the way it did: finish reason, errors, tool failures, warnings, the parsed
    output, and (optionally) the provider-near transcript.
    """

    feedback: dict[str, str] = {}
    metadata = span.metadata or {}
    outputs = span.outputs or {}

    finish = metadata.get("finish_reason") or outputs.get("finish_reason")
    if finish is not None:
        feedback["finish_reason"] = str(finish)

    error = metadata.get("error") or outputs.get("error")
    if error is not None:
        feedback["error"] = json.dumps(_jsonable(error), ensure_ascii=False)

    tool_errors: list[str] = []
    for message in dump_messages(span.messages):
        if message.get("role") != "tool":
            continue
        for part in message.get("content") or []:
            output = part.get("output") if isinstance(part, dict) else None
            if isinstance(output, dict) and output.get("type") in (
                "error-text",
                "error-json",
            ):
                tool_errors.append(
                    f"{part.get('toolName', '?')}: "
                    f"{json.dumps(output.get('value'), ensure_ascii=False)}"
                )
    if tool_errors:
        feedback["tool_errors"] = "\n".join(tool_errors)

    warnings = metadata.get("warnings")
    if warnings:
        feedback["warnings"] = json.dumps(_jsonable(warnings), ensure_ascii=False)

    if "object" in outputs:
        feedback["output"] = json.dumps(_jsonable(outputs["object"]), ensure_ascii=False)
    elif outputs.get("text"):
        feedback["output"] = str(outputs["text"])

    if span.usage is not None:
        feedback["usage"] = json.dumps(_jsonable(span.usage), ensure_ascii=False)

    if include_transcript:
        transcript = json.dumps(dump_messages(span.messages), ensure_ascii=False)
        if len(transcript) > max_transcript_chars:
            transcript = transcript[:max_transcript_chars] + "…[truncated]"
        feedback["transcript"] = transcript

    return feedback


def _prompt_metadata(prompt: Any) -> dict[str, Any]:
    input_schema = (
        prompt.input.model_dump(by_alias=True, exclude_none=True)
        if prompt.input
        else None
    )
    output_schema = (
        prompt.output.model_dump(by_alias=True, exclude_none=True)
        if prompt.output
        else None
    )
    return {
        "prompt": {
            "name": prompt.name,
            "version": prompt.version,
            "spec_version": getattr(prompt, "spec_version", None),
            "content_hash": prompt.content_hash(),
            "variables": prompt.variables,
            "input": input_schema,
            "output": output_schema,
            "message_ids": [message.id for message in prompt.messages],
            "tool_names": list(prompt.tools),
            "skill_names": list(getattr(prompt, "skills", {}) or {}),
        }
    }


def _default_inputs(
    call_kwargs: dict[str, Any],
    input_messages: list[ModelMessage],
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if call_kwargs.get("system") is not None:
        data["system"] = call_kwargs["system"]
    if call_kwargs.get("prompt") is not None:
        data["prompt"] = _jsonable(call_kwargs["prompt"])
        return data
    if call_kwargs.get("messages") is not None:
        data["messages"] = dump_messages(input_messages)
    return data


def _plain_call_kwargs(
    prompt_or_config: Any,
    model: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    call_kwargs = dict(overrides)
    if prompt_or_config is not None:
        call_kwargs.setdefault("prompt", prompt_or_config)
    if model is not None:
        call_kwargs["model"] = model
    return call_kwargs


def _is_prompt_config(value: Any) -> bool:
    return hasattr(value, "_call_kwargs") and hasattr(value, "render")


def _standardize_input_messages(call_kwargs: dict[str, Any]) -> list[ModelMessage]:
    from .generate import standardize_prompt

    return list(
        standardize_prompt(
            system=call_kwargs.get("system"),
            prompt=call_kwargs.get("prompt"),
            messages=call_kwargs.get("messages"),
        )
    )


async def generate_trace(
    prompt_or_config: Any = None,
    variables: Optional[dict[str, Any]] = None,
    *,
    model: Any = None,
    handlers: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    root_span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    **overrides: Any,
) -> GenerateTraceResult:
    """Run generation and attach a trace-backed history span.

    When passed a `Prompt`, this renders structured variables first. Otherwise,
    it accepts ordinary `generate_text` kwargs, e.g.
    `generate_trace(model=model, prompt="...")` or
    `generate_trace(model=model, messages=[...], inputs={...})`.
    """

    from .generate import generate_text
    from .telemetry import TraceCollector, TraceContext

    if _is_prompt_config(prompt_or_config):
        prompt = prompt_or_config
        call_kwargs = prompt._call_kwargs(variables, model, handlers, overrides)
        trace_inputs = inputs if inputs is not None else dict(variables or {})
        base_metadata = _prompt_metadata(prompt)
    else:
        if handlers is not None:
            raise TypeError(
                "handlers can only be used when generate_trace receives a Prompt."
            )
        call_kwargs = _plain_call_kwargs(prompt_or_config, model, overrides)
        input_messages = _standardize_input_messages(call_kwargs)
        trace_inputs = (
            inputs if inputs is not None else _default_inputs(call_kwargs, input_messages)
        )
        base_metadata = {}

    collector = TraceCollector()
    existing = call_kwargs.get("telemetry")
    call_kwargs["telemetry"] = (
        [collector]
        if existing in (None, True)
        else [collector, existing]
        if callable(existing)
        else [collector, *existing]
        if existing
        else [collector]
    )
    call_kwargs["trace_context"] = TraceContext(
        inputs=trace_inputs,
        outputs=outputs,
        metadata={**base_metadata, **(metadata or {})},
        trace_id=trace_id,
        span_id=span_id,
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
    )
    # The integrated pipeline builds exactly one trace, delivers it to the
    # collector AND any connected sinks, and attaches it to failures.
    result = await generate_text(**call_kwargs)
    assert collector.last is not None
    return GenerateTraceResult(result=result, trace=collector.last)


def stream_trace(
    prompt_or_config: Any = None,
    variables: Optional[dict[str, Any]] = None,
    *,
    model: Any = None,
    handlers: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    root_span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    **overrides: Any,
) -> StreamTraceResult:
    """Stream generation and expose an awaitable trace after completion."""

    from .generate import stream_text

    if _is_prompt_config(prompt_or_config):
        prompt = prompt_or_config
        call_kwargs = prompt._call_kwargs(variables, model, handlers, overrides)
        input_messages = list(call_kwargs["messages"])
        trace_inputs = inputs if inputs is not None else dict(variables or {})
        base_metadata = _prompt_metadata(prompt)
    else:
        if handlers is not None:
            raise TypeError(
                "handlers can only be used when stream_trace receives a Prompt."
            )
        call_kwargs = _plain_call_kwargs(prompt_or_config, model, overrides)
        input_messages = _standardize_input_messages(call_kwargs)
        trace_inputs = (
            inputs if inputs is not None else _default_inputs(call_kwargs, input_messages)
        )
        base_metadata = {}

    from .telemetry import active_sinks

    call_kwargs["telemetry"] = False  # this wrapper builds the (single) trace
    return StreamTraceResult(
        result=stream_text(**call_kwargs),
        input_messages=input_messages,
        inputs=trace_inputs,
        outputs=outputs,
        metadata={**base_metadata, **(metadata or {})},
        trace_id=trace_id,
        span_id=span_id,
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
        sinks=active_sinks(),
    )
