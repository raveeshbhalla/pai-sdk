"""Trace telemetry — connect plumbing once, every call emits.

Tracing is not a separate API. Connect a sink (an OTEL exporter, an
observability vendor, a JSONL file, an in-memory collector) and every
`generate_text`/`stream_text` call — including calls made through
`Prompt.generate` and bound `PromptSpec` prompts — produces a `Trace` as a
side effect, exactly like the AI SDK's telemetry integrations:

    from pai_sdk import configure_telemetry, otel_sink
    configure_telemetry(otel_sink(my_exporter))     # once, at startup

    result = await generate_text(model=..., prompt="...")   # traced
    result = await prompt.generate({...})                    # traced, with
                                                             # prompt metadata

Sinks are fire-and-forget: a raising sink is logged and never breaks (or
slows the error path of) generation. Failed calls emit a failed-trace span
and the exception carries it as `exc.trace`.

Scoping:

- `configure_telemetry(*sinks)` sets the process-wide sinks (call with no
  arguments to disconnect).
- `telemetry(*sinks)` is a context manager adding sinks for a block/task —
  handy for tests and optimizer evaluators.
- Per call, `generate_text(..., telemetry=sink_or_list)` adds sinks for that
  call only, and `telemetry=False` disables emission for that call.

`generate_trace()`/`stream_trace()` remain as in-process conveniences that
return the trace directly (they ride this same pipeline — the trace they
return is the same object connected sinks receive).
"""

from __future__ import annotations

import contextvars
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence, Union

from .trace import Trace

logger = logging.getLogger("pai_sdk.telemetry")

# A sink is any callable taking a Trace; async callables are awaited.
TraceSink = Callable[[Trace], Any]

TelemetryArg = Union[None, bool, TraceSink, Sequence[TraceSink]]

_GLOBAL_SINKS: tuple[TraceSink, ...] = ()
_SCOPED_SINKS: contextvars.ContextVar[tuple[TraceSink, ...]] = contextvars.ContextVar(
    "pai_sdk_scoped_trace_sinks", default=()
)


def configure_telemetry(*sinks: TraceSink) -> None:
    """Set the process-wide trace sinks (replaces the previous set).

    Call with no arguments to disconnect telemetry.
    """
    global _GLOBAL_SINKS
    _GLOBAL_SINKS = tuple(sinks)


@contextmanager
def telemetry(*sinks: TraceSink) -> Iterator[None]:
    """Add sinks for the current block/task (on top of the global set)."""
    token = _SCOPED_SINKS.set(_SCOPED_SINKS.get() + tuple(sinks))
    try:
        yield
    finally:
        _SCOPED_SINKS.reset(token)


def active_sinks() -> tuple[TraceSink, ...]:
    """The sinks a call made right now would emit to."""
    return _GLOBAL_SINKS + _SCOPED_SINKS.get()


def resolve_sinks(telemetry: TelemetryArg) -> tuple[TraceSink, ...]:
    """Resolve a per-call `telemetry=` argument against the active sinks."""
    if telemetry is False:
        return ()
    base = active_sinks()
    if telemetry is None or telemetry is True:
        return base
    if callable(telemetry):
        return base + (telemetry,)
    return base + tuple(telemetry)


async def emit_trace(trace: Trace, sinks: Sequence[TraceSink]) -> None:
    """Deliver a trace to sinks; failures are logged, never raised."""
    for sink in sinks:
        try:
            outcome = sink(trace)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:  # noqa: BLE001 — telemetry must never break calls
            logger.exception("pai-sdk trace sink failed; generation unaffected")


@dataclass
class TraceContext:
    """Semantic context attached to integrated traces.

    `Prompt.generate` sets this automatically (variables as `inputs`, prompt
    name/hash/ids as `metadata`); pass one explicitly to enrich raw
    `generate_text` calls or to thread span relationships.
    """

    inputs: Optional[dict[str, Any]] = None
    outputs: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    root_span_id: Optional[str] = None
    parent_span_id: Optional[str] = None


class TraceCollector:
    """In-memory sink — for optimizer evaluators and tests.

        collector = TraceCollector()
        with telemetry(collector):
            await prompt.generate({...})
        collector.last.spans[0].messages
    """

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def __call__(self, trace: Trace) -> None:
        self.traces.append(trace)

    @property
    def last(self) -> Optional[Trace]:
        return self.traces[-1] if self.traces else None

    def clear(self) -> None:
        self.traces.clear()


def otel_sink(export: Callable[[list[dict[str, Any]]], Any]) -> TraceSink:
    """Adapt an exporter of OpenTelemetry span dicts into a trace sink.

    `export` receives the spans produced by `trace_to_otel_spans` (lossless
    `pai.*` attributes plus standard `gen_ai.*` mirrors) — hand them to your
    collector/vendor exporter. `trace_from_otel_spans` recreates replayable
    history from the same spans later.
    """
    from .integrations.otel import trace_to_otel_spans

    def sink(trace: Trace) -> Any:
        return export(trace_to_otel_spans(trace))

    return sink


# ---------------------------------------------------------------------------
# Live-call instrumentation hook (registered by integrations.otel.instrument())
# ---------------------------------------------------------------------------

# A factory taking (call_name, TraceContext|None) and returning a live-call
# handle with chain_prepare_step/chain_on_step_finish/finish/fail — or None.
_LIVE_CALL_FACTORY: Optional[Callable[[str, Any], Any]] = None


def set_live_call_factory(factory: Optional[Callable[[str, Any], Any]]) -> None:
    """Internal: registered by `pai_sdk.integrations.otel.instrument()`."""
    global _LIVE_CALL_FACTORY
    _LIVE_CALL_FACTORY = factory


def start_live_call(name: str, context: Any) -> Any:
    """Internal: open a live instrumentation span for a call, if enabled."""
    if _LIVE_CALL_FACTORY is None:
        return None
    try:
        return _LIVE_CALL_FACTORY(name, context)
    except Exception:  # noqa: BLE001 — instrumentation must never break calls
        logger.exception("pai-sdk live-call instrumentation failed to start")
        return None


class QueuedSink:
    """Decouple a sink from the request path: enqueue and return immediately.

    A background worker drains the queue; sync inner sinks run in a thread
    (so blocking I/O — files, HTTP clients — never stalls the event loop).
    When the queue is full the oldest trace is dropped (and logged). Call
    `await queued.flush()` before shutdown to drain outstanding traces.
    """

    def __init__(self, sink: TraceSink, *, max_queue: int = 1024) -> None:
        self._sink = sink
        self._max_queue = max_queue
        self._queue: Optional[Any] = None  # asyncio.Queue, created lazily
        self._worker: Optional[Any] = None
        self._dropped = 0

    def _ensure_worker(self) -> None:
        import asyncio

        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_queue)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        import asyncio

        assert self._queue is not None
        while True:
            trace = await self._queue.get()
            try:
                if inspect.iscoroutinefunction(self._sink):
                    await self._sink(trace)
                else:
                    outcome = await asyncio.to_thread(self._sink, trace)
                    if inspect.isawaitable(outcome):
                        await outcome
            except Exception:  # noqa: BLE001
                logger.exception("queued trace sink failed; trace dropped")
            finally:
                self._queue.task_done()

    def __call__(self, trace: Trace) -> None:
        self._ensure_worker()
        assert self._queue is not None
        try:
            self._queue.put_nowait(trace)
        except Exception:  # queue full
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(trace)
                self._dropped += 1
                logger.warning(
                    "queued trace sink overflow; dropped oldest trace "
                    "(%d dropped so far)",
                    self._dropped,
                )
            except Exception:  # noqa: BLE001 — never break the caller
                logger.exception("queued trace sink could not enqueue")

    async def flush(self) -> None:
        """Wait until every enqueued trace has been delivered."""
        if self._queue is not None:
            await self._queue.join()


def queued_sink(sink: TraceSink, *, max_queue: int = 1024) -> QueuedSink:
    """Wrap any sink so delivery happens off the request path (see QueuedSink)."""
    return QueuedSink(sink, max_queue=max_queue)


def jsonl_sink(path: Any) -> TraceSink:
    """Append each trace as one JSON line — the simplest durable sink."""
    from .trace import dump_trace_json

    def sink(trace: Trace) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(dump_trace_json(trace) + "\n")

    return sink
