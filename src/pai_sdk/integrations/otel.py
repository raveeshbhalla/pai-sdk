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


def _span_attribute_dict(
    data: dict[str, Any],
    span: dict[str, Any],
    *,
    include_payloads: bool = True,
) -> dict[str, Any]:
    """The lossless `pai.*` + standard `gen_ai.*` attribute set for one span.

    Shared between the post-hoc exporter (`trace_to_otel_spans`) and live
    instrumentation (`instrument()`), so `trace_from_otel_spans` can recreate
    replayable history from either.
    """
    metadata = span.get("metadata") or {}
    response = metadata.get("response") or {}
    usage = span.get("usage") or {}
    attributes: dict[str, Any] = {
        "pai.trace.schema_version": data.get("schemaVersion", TRACE_SCHEMA_VERSION),
        "pai.trace.id": data["id"],
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
    return {key: value for key, value in attributes.items() if value is not None}


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
        attributes = _span_attribute_dict(data, span, include_payloads=include_payloads)

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


# ---------------------------------------------------------------------------
# Live instrumentation: real OTel spans via the global tracer
# ---------------------------------------------------------------------------


_INSTRUMENT_LOGGER = __import__("logging").getLogger("pai_sdk.telemetry")


def instrument(*, tracer_provider: Any = None, tracer: Any = None) -> None:
    """Create real OpenTelemetry spans for every pai-sdk call.

    One line makes any OTel-based vendor work, with correct nesting:

        from pai_sdk.integrations.otel import instrument
        instrument()          # uses the globally configured tracer provider

    Each `generate_text`/`stream_text` call opens a span (named
    `pai_sdk.generate_text` / `pai_sdk.stream_text`) **during** the call, so
    it nests under whatever span is current (e.g. your web-request span),
    with one child span per provider step. On completion the call span
    carries the same lossless `pai.*` attributes as `trace_to_otel_spans` —
    so `trace_from_otel_spans` can recreate replayable history straight from
    your backend — plus standard `gen_ai.*` mirrors for vendor UIs.

    Spans are exported by YOUR OpenTelemetry SDK pipeline (batching, sampling,
    retries live there — nothing runs on pai-sdk's request path beyond
    attribute assembly). Requires the `otel` extra (`pip install
    "pai-sdk[otel]"`); a tracer provider must be configured for spans to be
    recorded. Instrumentation failures are logged and never break generation.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError as exc:  # pragma: no cover - exercised via extras
        from ..errors import MissingDependencyError

        raise MissingDependencyError("opentelemetry-api", "otel") from exc

    if tracer is None:
        provider = tracer_provider or otel_trace.get_tracer_provider()
        tracer = provider.get_tracer("pai_sdk")

    from ..telemetry import set_live_call_factory

    def factory(name: str, context: Any) -> "_LiveCall":
        return _LiveCall(otel_trace, tracer, name, context)

    set_live_call_factory(factory)


def uninstrument() -> None:
    """Stop creating OTel spans for pai-sdk calls."""
    from ..telemetry import set_live_call_factory

    set_live_call_factory(None)


class _LiveCall:
    """One in-flight instrumented call: an open span + per-step children.

    Every method is exception-isolated; instrumentation can never break or
    fail a generation call.
    """

    def __init__(self, otel_trace: Any, tracer: Any, name: str, context: Any) -> None:
        self._trace_api = otel_trace
        self._tracer = tracer
        self._step_span: Any = None
        attributes = {"gen_ai.operation.name": "chat"}
        prompt_meta = (getattr(context, "metadata", None) or {}).get("prompt") or {}
        if prompt_meta.get("name"):
            attributes["pai.prompt.name"] = prompt_meta["name"]
        if prompt_meta.get("content_hash"):
            attributes["pai.prompt.content_hash"] = prompt_meta["content_hash"]
        self._span = tracer.start_span(name, attributes=attributes)

    # -- per-step children ---------------------------------------------------

    def chain_prepare_step(self, user: Any) -> Any:
        async def hook(*, model: Any, step_number: int, steps: Any, messages: Any) -> Any:
            try:
                self._start_step(step_number, model)
            except Exception:  # noqa: BLE001
                _INSTRUMENT_LOGGER.exception("otel step-span start failed")
            if user is None:
                return None
            outcome = user(
                model=model, step_number=step_number, steps=steps, messages=messages
            )
            if hasattr(outcome, "__await__"):
                return await outcome
            return outcome

        return hook

    def chain_on_step_finish(self, user: Any) -> Any:
        async def hook(step: Any) -> Any:
            try:
                self._finish_step(step)
            except Exception:  # noqa: BLE001
                _INSTRUMENT_LOGGER.exception("otel step-span finish failed")
            if user is None:
                return None
            outcome = user(step)
            if hasattr(outcome, "__await__"):
                return await outcome
            return outcome

        return hook

    def _start_step(self, step_number: int, model: Any) -> None:
        context = self._trace_api.set_span_in_context(self._span)
        attributes = {"pai.step.number": step_number}
        model_id = getattr(model, "model_id", None)
        if model_id:
            attributes["gen_ai.request.model"] = model_id
        self._step_span = self._tracer.start_span(
            f"pai_sdk.step {step_number}", context=context, attributes=attributes
        )

    def _finish_step(self, step: Any) -> None:
        span, self._step_span = self._step_span, None
        if span is None:
            return
        usage = getattr(step, "usage", None)
        for attr, key in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
        ):
            value = getattr(usage, attr, None)
            if value is not None:
                span.set_attribute(key, value)
        finish_reason = getattr(step, "finish_reason", None)
        if finish_reason is not None:
            span.set_attribute("gen_ai.response.finish_reasons", [str(finish_reason)])
        response = getattr(step, "response", None)
        model_id = getattr(response, "model_id", None)
        if model_id:
            span.set_attribute("gen_ai.response.model", model_id)
        span.end()

    # -- completion ------------------------------------------------------------

    def _apply_trace(self, trace: Any) -> None:
        data = dump_trace(trace)
        spans = data.get("spans") or []
        if spans:
            for key, value in _span_attribute_dict(data, spans[0]).items():
                self._span.set_attribute(key, value)

    def _close_dangling_step(self) -> None:
        span, self._step_span = self._step_span, None
        if span is not None:
            span.end()

    def finish(self, trace: Any) -> None:
        try:
            self._close_dangling_step()
            self._apply_trace(trace)
            self._span.set_status(self._trace_api.StatusCode.OK)
            self._span.end()
        except Exception:  # noqa: BLE001
            _INSTRUMENT_LOGGER.exception("otel call-span finish failed")

    def fail(self, error: BaseException, trace: Any = None) -> None:
        try:
            self._close_dangling_step()
            if trace is not None:
                self._apply_trace(trace)
            self._span.record_exception(error)
            self._span.set_status(
                self._trace_api.Status(self._trace_api.StatusCode.ERROR, str(error))
            )
            self._span.end()
        except Exception:  # noqa: BLE001
            _INSTRUMENT_LOGGER.exception("otel call-span fail-path failed")
