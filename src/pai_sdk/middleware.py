"""Language model middleware — a Python port of the AI SDK's
``LanguageModelV3Middleware`` + ``wrapLanguageModel`` + the built-in
middlewares.

A middleware can hook three points around a ``LanguageModel`` call:

- ``transform_params`` — rewrite the :class:`CallOptions` before the call.
- ``wrap_generate`` — wrap ``do_generate`` (non-streaming).
- ``wrap_stream`` — wrap ``do_stream`` (streaming).

Use :func:`wrap_language_model` to apply one or more middlewares to a model,
producing a new :class:`LanguageModel`. Composition matches the AI SDK: the
**first** middleware in the list is the **outermost** layer. ``transform_params``
for every middleware runs (outer-first) before any ``wrap_*`` body executes.

    model = wrap_language_model(
        anthropic("claude-opus-4-8"),
        middleware=[logging_middleware(), extract_reasoning_middleware()],
    )

Built-in factories:

- :func:`default_settings_middleware` — fill ``None`` CallOptions fields.
- :func:`extract_reasoning_middleware` — pull ``<think>...</think>`` spans out
  of text into reasoning (generate + stream).
- :func:`simulate_streaming_middleware` — synthesize a stream from a single
  non-streaming generation.
- :func:`logging_middleware` — log request/response summaries (a copy-me example).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

from .messages import ReasoningPart, TextPart
from .provider import CallOptions, LanguageModel, ProviderResult
from .stream import (
    Finish,
    ProviderStreamPart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    ResponseMetadataPart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallPart,
    ToolInputEnd,
    ToolInputStart,
)

# ---------------------------------------------------------------------------
# Middleware spec
# ---------------------------------------------------------------------------

# transform_params(options, *, type, model) -> CallOptions (sync or async)
TransformParamsFn = Callable[..., Union[CallOptions, Awaitable[CallOptions]]]

# wrap_generate(do_generate, options, model) -> ProviderResult (async)
WrapGenerateFn = Callable[
    [Callable[[], Awaitable[ProviderResult]], CallOptions, LanguageModel],
    Awaitable[ProviderResult],
]

# wrap_stream(do_stream, options, model) -> AsyncIterator[ProviderStreamPart]
WrapStreamFn = Callable[
    [Callable[[], AsyncIterator[ProviderStreamPart]], CallOptions, LanguageModel],
    AsyncIterator[ProviderStreamPart],
]


@dataclass
class LanguageModelMiddleware:
    """Middleware for a :class:`LanguageModel` (AI SDK LanguageModelV3Middleware).

    All hooks are optional. A plain object exposing any of these attribute
    names is also accepted by :func:`wrap_language_model` (duck-typed via
    :func:`_resolve`), so you do not have to use this dataclass.

    - ``transform_params(options, *, type, model)`` — return a (possibly
      modified) :class:`CallOptions`. ``type`` is ``"generate"`` or
      ``"stream"``. May be sync or async. Mutating ``options`` in place is
      allowed, but returning a copy (``dataclasses.replace(options, ...)``) is
      recommended so the caller's object is left untouched.
    - ``wrap_generate(do_generate, options, model)`` — async; ``do_generate``
      is a zero-arg thunk that runs the next layer with the already-transformed
      options.
    - ``wrap_stream(do_stream, options, model)`` — return (or be) an async
      iterator of provider stream parts; ``do_stream`` is a zero-arg thunk
      returning the next layer's stream.
    """

    transform_params: Optional[TransformParamsFn] = None
    wrap_generate: Optional[WrapGenerateFn] = None
    wrap_stream: Optional[WrapStreamFn] = None
    # Optional metadata mirroring the AI SDK (purely informational).
    middleware_version: Optional[str] = None


def _resolve(mw: Any) -> LanguageModelMiddleware:
    """Coerce a middleware into a :class:`LanguageModelMiddleware`.

    Accepts the dataclass itself or any object exposing ``transform_params`` /
    ``wrap_generate`` / ``wrap_stream`` attributes (each optional)."""
    if isinstance(mw, LanguageModelMiddleware):
        return mw
    return LanguageModelMiddleware(
        transform_params=getattr(mw, "transform_params", None),
        wrap_generate=getattr(mw, "wrap_generate", None),
        wrap_stream=getattr(mw, "wrap_stream", None),
        middleware_version=getattr(mw, "middleware_version", None),
    )


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# wrap_language_model
# ---------------------------------------------------------------------------


class _WrappedLanguageModel(LanguageModel):
    """A :class:`LanguageModel` produced by :func:`wrap_language_model`."""

    def __init__(
        self,
        model: LanguageModel,
        middlewares: list[LanguageModelMiddleware],
        *,
        model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        self._model = model
        self._middlewares = middlewares
        self.provider = provider_id if provider_id is not None else model.provider
        self.model_id = model_id if model_id is not None else model.model_id

    async def _transform_params(
        self, options: CallOptions, *, type: Literal["generate", "stream"]
    ) -> CallOptions:
        """Run every middleware's transform_params, outer-first."""
        transformed = options
        for mw in self._middlewares:
            if mw.transform_params is not None:
                transformed = await _maybe_await(
                    mw.transform_params(
                        transformed, type=type, model=self._model
                    )
                )
        return transformed

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        transformed = await self._transform_params(options, type="generate")

        # Innermost: call the wrapped model with the transformed options.
        def base() -> Awaitable[ProviderResult]:
            return self._model.do_generate(transformed)

        thunk: Callable[[], Awaitable[ProviderResult]] = base

        # Wrap inner-to-outer so the FIRST middleware ends up outermost.
        for mw in reversed(self._middlewares):
            if mw.wrap_generate is not None:
                wrap = mw.wrap_generate
                inner = thunk

                def make(wrap=wrap, inner=inner) -> Callable[[], Awaitable[ProviderResult]]:
                    async def call() -> ProviderResult:
                        return await wrap(inner, transformed, self._model)

                    return call

                thunk = make()

        return await thunk()

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        transformed = await self._transform_params(options, type="stream")

        def base() -> AsyncIterator[ProviderStreamPart]:
            return self._model.do_stream(transformed)

        thunk: Callable[[], AsyncIterator[ProviderStreamPart]] = base

        for mw in reversed(self._middlewares):
            if mw.wrap_stream is not None:
                wrap = mw.wrap_stream
                inner = thunk

                def make(
                    wrap=wrap, inner=inner
                ) -> Callable[[], AsyncIterator[ProviderStreamPart]]:
                    def call() -> AsyncIterator[ProviderStreamPart]:
                        # wrap may be an async-generator function (returns an
                        # iterator directly) or an async function returning one.
                        result = wrap(inner, transformed, self._model)
                        return _as_async_iter(result)

                    return call

                thunk = make()

        # Re-yield so do_stream is itself an async generator (matches ABC use).
        async for part in _as_async_iter(thunk()):
            yield part


async def _as_async_iter(value: Any) -> AsyncIterator[ProviderStreamPart]:
    """Normalize a stream thunk's return into an async iterator.

    Accepts an async iterator/generator directly, or an awaitable that resolves
    to one (so wrap_stream may be written as a plain ``async def`` returning a
    stream)."""
    if asyncio.iscoroutine(value):
        value = await value
    async for part in value:
        yield part


def wrap_language_model(
    model: LanguageModel,
    middleware: Union[Any, list[Any]],
    *,
    model_id: Optional[str] = None,
    provider_id: Optional[str] = None,
) -> LanguageModel:
    """Wrap a :class:`LanguageModel` with one or more middlewares (AI SDK
    ``wrapLanguageModel``).

    ``middleware`` is a single middleware or a list. Composition matches the AI
    SDK: the **first** middleware in the list is the **outermost** layer — its
    ``transform_params`` runs first and its ``wrap_generate``/``wrap_stream``
    wraps everything inside it. ``transform_params`` for all middlewares run
    (outer-first) before any ``wrap_*`` body executes.

    The returned model's ``provider`` and ``model_id`` default to the wrapped
    model's; override with ``provider_id`` / ``model_id``. Wrapping an
    already-wrapped model nests correctly.
    """
    mws = middleware if isinstance(middleware, list) else [middleware]
    resolved = [_resolve(mw) for mw in mws]
    return _WrappedLanguageModel(
        model, resolved, model_id=model_id, provider_id=provider_id
    )


# ---------------------------------------------------------------------------
# default_settings_middleware
# ---------------------------------------------------------------------------

# CallOptions fields that default_settings may fill (everything except prompt).
_SETTINGS_FIELDS = {
    f.name for f in dataclasses.fields(CallOptions) if f.name != "prompt"
}


def default_settings_middleware(settings: dict[str, Any]) -> LanguageModelMiddleware:
    """Fill ``None`` :class:`CallOptions` fields from ``settings`` (AI SDK
    ``defaultSettingsMiddleware``).

    Keys are CallOptions field names (e.g. ``"temperature"``,
    ``"max_output_tokens"``). A field is filled only when the incoming value is
    ``None`` (so explicit caller values always win). ``provider_options`` is
    merged per provider key, with existing (caller) values winning over the
    defaults.
    """

    def transform_params(
        options: CallOptions, *, type: str, model: LanguageModel
    ) -> CallOptions:
        updates: dict[str, Any] = {}
        for key, value in settings.items():
            if key == "provider_options":
                updates["provider_options"] = _merge_provider_options(
                    options.provider_options, value
                )
                continue
            if key not in _SETTINGS_FIELDS:
                continue
            if getattr(options, key, None) is None:
                updates[key] = value
        if not updates:
            return options
        return dataclasses.replace(options, **updates)

    return LanguageModelMiddleware(transform_params=transform_params)


def _merge_provider_options(
    existing: dict[str, dict[str, Any]],
    defaults: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge ``defaults`` under ``existing`` per provider key; existing wins."""
    merged: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in (defaults or {}).items()
    }
    for provider, opts in (existing or {}).items():
        if provider in merged:
            combined = dict(merged[provider])
            combined.update(opts)  # existing wins
            merged[provider] = combined
        else:
            merged[provider] = dict(opts)
    return merged


# ---------------------------------------------------------------------------
# extract_reasoning_middleware
# ---------------------------------------------------------------------------


def extract_reasoning_middleware(
    tag_name: str = "think",
    *,
    separator: str = "\n",
    start_with_reasoning: bool = False,
) -> LanguageModelMiddleware:
    """Extract reasoning wrapped in ``<tag_name>...</tag_name>`` tags out of
    text content (AI SDK ``extractReasoningMiddleware``).

    On ``wrap_generate``: each :class:`TextPart` is split — tagged spans become
    :class:`ReasoningPart`s (in order, interleaved with the surrounding text),
    and the remaining text segments are rejoined with ``separator``.

    On ``wrap_stream``: a small streaming tag parser runs over ``text-delta``
    parts, buffering enough to detect tag boundaries that may be split across
    deltas (a possible partial-tag suffix is held back until resolved), and
    re-emits reasoning content as ``reasoning-start/-delta/-end`` (with a
    synthesized id) and ordinary text as ``text-delta``. ``start_with_reasoning``
    makes the parser begin already inside the tag (no opening tag in the stream).

    Single-level, non-nested tags only (like the AI SDK).
    """
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"

    def split_text(text: str) -> list[tuple[str, str]]:
        """Split into ordered ("text"|"reasoning", value) segments."""
        segments: list[tuple[str, str]] = []
        remaining = text
        in_reasoning = start_with_reasoning
        while remaining:
            if in_reasoning:
                idx = remaining.find(close_tag)
                if idx == -1:
                    segments.append(("reasoning", remaining))
                    remaining = ""
                else:
                    if remaining[:idx]:
                        segments.append(("reasoning", remaining[:idx]))
                    remaining = remaining[idx + len(close_tag):]
                    in_reasoning = False
            else:
                idx = remaining.find(open_tag)
                if idx == -1:
                    segments.append(("text", remaining))
                    remaining = ""
                else:
                    if remaining[:idx]:
                        segments.append(("text", remaining[:idx]))
                    remaining = remaining[idx + len(open_tag):]
                    in_reasoning = True
        return segments

    async def wrap_generate(
        do_generate: Callable[[], Awaitable[ProviderResult]],
        options: CallOptions,
        model: LanguageModel,
    ) -> ProviderResult:
        result = await do_generate()
        new_content: list[Any] = []
        for part in result.content:
            if not isinstance(part, TextPart):
                new_content.append(part)
                continue
            segments = split_text(part.text)
            text_pieces: list[str] = []
            ordered: list[Any] = []
            for kind, value in segments:
                if kind == "reasoning":
                    ordered.append(ReasoningPart(text=value))
                else:
                    text_pieces.append(value)
            # Emit reasoning parts in order; collapse text segments into one
            # TextPart joined by the separator (matching AI SDK behavior).
            for seg in ordered:
                new_content.append(seg)
            joined = separator.join(p for p in text_pieces)
            if joined:
                new_content.append(TextPart(text=joined))
        return dataclasses.replace(result, content=new_content)

    async def wrap_stream(
        do_stream: Callable[[], AsyncIterator[ProviderStreamPart]],
        options: CallOptions,
        model: LanguageModel,
    ) -> AsyncIterator[ProviderStreamPart]:
        return _extract_reasoning_stream(
            do_stream(),
            open_tag=open_tag,
            close_tag=close_tag,
            separator=separator,
            start_with_reasoning=start_with_reasoning,
        )

    return LanguageModelMiddleware(
        wrap_generate=wrap_generate, wrap_stream=wrap_stream
    )


def _max_partial_suffix(buffer: str, tag: str) -> int:
    """Length of the longest suffix of ``buffer`` that is a proper prefix of
    ``tag`` (i.e. could be the start of ``tag`` split across deltas)."""
    max_len = min(len(buffer), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if tag.startswith(buffer[-length:]):
            return length
    return 0


async def _extract_reasoning_stream(
    source: AsyncIterator[ProviderStreamPart],
    *,
    open_tag: str,
    close_tag: str,
    separator: str,
    start_with_reasoning: bool,
) -> AsyncIterator[ProviderStreamPart]:
    """Streaming tag parser. Re-emits reasoning spans as reasoning-* parts and
    other text as text deltas. Non-text parts pass through unchanged."""
    reasoning_id = "reasoning-0"
    in_reasoning = start_with_reasoning
    buffer = ""  # text not yet emitted (may contain a partial tag)
    reasoning_open = False  # whether we've emitted reasoning-start
    text_id: Optional[str] = None  # id of the current passed-through text block
    text_started = False
    saw_text_start = False

    async def open_reasoning() -> AsyncIterator[ProviderStreamPart]:
        nonlocal reasoning_open
        if not reasoning_open:
            reasoning_open = True
            yield ReasoningStart(id=reasoning_id)

    async def emit(text: str, *, reasoning: bool) -> AsyncIterator[ProviderStreamPart]:
        nonlocal text_started
        if not text:
            return
        if reasoning:
            async for p in open_reasoning():
                yield p
            yield ReasoningDelta(id=reasoning_id, text=text)
        else:
            if text_id is not None and not text_started:
                text_started = True
                yield TextStart(id=text_id)
            yield TextDelta(id=text_id or "text-0", text=text)

    async for part in source:
        if isinstance(part, TextStart):
            text_id = part.id
            saw_text_start = True
            # Defer emitting TextStart until we actually have text to pass on.
            continue
        if isinstance(part, TextDelta):
            if text_id is None:
                text_id = part.id
            buffer += part.text
            # Process the buffer, holding back any partial-tag suffix.
            while True:
                target = close_tag if in_reasoning else open_tag
                idx = buffer.find(target)
                if idx != -1:
                    before = buffer[:idx]
                    async for p in emit(before, reasoning=in_reasoning):
                        yield p
                    buffer = buffer[idx + len(target):]
                    in_reasoning = not in_reasoning
                    continue
                # No full tag: emit everything except a possible partial suffix.
                hold = _max_partial_suffix(buffer, target)
                emittable = buffer[: len(buffer) - hold] if hold else buffer
                async for p in emit(emittable, reasoning=in_reasoning):
                    yield p
                buffer = buffer[len(buffer) - hold:] if hold else ""
                break
            continue
        if isinstance(part, TextEnd):
            # Flush remaining buffer as-is (it can't be a real tag anymore).
            if buffer:
                async for p in emit(buffer, reasoning=in_reasoning):
                    yield p
                buffer = ""
            if reasoning_open:
                yield ReasoningEnd(id=reasoning_id)
                reasoning_open = False
                in_reasoning = False
            if text_started:
                yield TextEnd(id=text_id or "text-0")
                text_started = False
            continue
        # Non-text parts (reasoning, tool, finish, metadata, ...) pass through.
        yield part


# ---------------------------------------------------------------------------
# simulate_streaming_middleware
# ---------------------------------------------------------------------------


def simulate_streaming_middleware() -> LanguageModelMiddleware:
    """Synthesize a stream from a single non-streaming generation (AI SDK
    ``simulateStreamingMiddleware``).

    ``wrap_stream`` calls the inner ``do_generate`` once and emits, in order: a
    :class:`ResponseMetadataPart`; per content part — :class:`TextPart` →
    text-start / one text-delta / text-end; :class:`ReasoningPart` →
    reasoning-start / delta / end; :class:`ToolCallPart` → tool-input-start /
    tool-input-end then the part itself — and finally a :class:`Finish` carrying
    the result's usage, finish reason, and provider metadata.
    """

    async def wrap_stream(
        do_stream: Callable[[], AsyncIterator[ProviderStreamPart]],
        options: CallOptions,
        model: LanguageModel,
    ) -> AsyncIterator[ProviderStreamPart]:
        # simulate_streaming uses the non-streaming path: call do_generate.
        result = await model.do_generate(options)
        return _simulate_stream(result)

    return LanguageModelMiddleware(wrap_stream=wrap_stream)


async def _simulate_stream(
    result: ProviderResult,
) -> AsyncIterator[ProviderStreamPart]:
    yield ResponseMetadataPart(
        id=result.response.id,
        model_id=result.response.model_id,
        request=result.request,
    )
    for index, part in enumerate(result.content):
        if isinstance(part, TextPart):
            block_id = f"text-{index}"
            yield TextStart(id=block_id)
            yield TextDelta(id=block_id, text=part.text)
            yield TextEnd(id=block_id)
        elif isinstance(part, ReasoningPart):
            block_id = f"reasoning-{index}"
            yield ReasoningStart(id=block_id)
            yield ReasoningDelta(id=block_id, text=part.text)
            yield ReasoningEnd(id=block_id)
        elif isinstance(part, ToolCallPart):
            yield ToolInputStart(
                id=part.tool_call_id,
                tool_name=part.tool_name,
                provider_executed=part.provider_executed,
            )
            yield ToolInputEnd(id=part.tool_call_id)
            yield part
        else:
            # Other assistant parts (files, sources, tool-results) pass through
            # if they are already valid provider stream parts; otherwise skip.
            if hasattr(part, "type"):
                continue
    yield Finish(
        finish_reason=result.finish_reason,
        total_usage=result.usage,
        raw_finish_reason=result.raw_finish_reason,
        provider_metadata=result.provider_metadata,
    )


# ---------------------------------------------------------------------------
# logging_middleware
# ---------------------------------------------------------------------------


def logging_middleware(
    logger: Optional[logging.Logger] = None,
    *,
    level: int = logging.DEBUG,
) -> LanguageModelMiddleware:
    """A minimal example middleware that logs request options and response
    summaries via the stdlib ``logging`` module — the template users copy.

    Logs the model id and the standard call parameters on the way in, and the
    finish reason + usage on the way out. For streams it counts the parts and
    logs the total on completion.
    """
    log = logger if logger is not None else logging.getLogger("pai_sdk.middleware")

    def _describe_options(options: CallOptions, model: LanguageModel) -> str:
        return (
            f"model={model.model_id} "
            f"max_output_tokens={options.max_output_tokens} "
            f"temperature={options.temperature} "
            f"tools={len(options.tools)} "
            f"tool_choice={options.tool_choice}"
        )

    async def wrap_generate(
        do_generate: Callable[[], Awaitable[ProviderResult]],
        options: CallOptions,
        model: LanguageModel,
    ) -> ProviderResult:
        log.log(level, "generate request: %s", _describe_options(options, model))
        result = await do_generate()
        log.log(
            level,
            "generate response: finish_reason=%s usage=%s",
            result.finish_reason,
            result.usage,
        )
        return result

    async def wrap_stream(
        do_stream: Callable[[], AsyncIterator[ProviderStreamPart]],
        options: CallOptions,
        model: LanguageModel,
    ) -> AsyncIterator[ProviderStreamPart]:
        log.log(level, "stream request: %s", _describe_options(options, model))
        return _logging_stream(do_stream(), log, level)

    return LanguageModelMiddleware(
        wrap_generate=wrap_generate, wrap_stream=wrap_stream
    )


async def _logging_stream(
    source: AsyncIterator[ProviderStreamPart],
    log: logging.Logger,
    level: int,
) -> AsyncIterator[ProviderStreamPart]:
    count = 0
    finish_reason: Any = None
    async for part in source:
        count += 1
        if isinstance(part, Finish):
            finish_reason = part.finish_reason
        yield part
    log.log(
        level,
        "stream response: parts=%d finish_reason=%s",
        count,
        finish_reason,
    )


__all__ = [
    "LanguageModelMiddleware",
    "wrap_language_model",
    "default_settings_middleware",
    "extract_reasoning_middleware",
    "simulate_streaming_middleware",
    "logging_middleware",
]
