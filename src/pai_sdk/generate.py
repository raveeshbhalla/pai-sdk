"""generate_text() and stream_text() — the core AI SDK functions in Python."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Sequence,
    Union,
)

from ._prompt import Prompt, standardize_prompt
from .errors import (
    AbortError,
    APICallError,
    GenerationTimeoutError,
    InvalidToolInputError,
    NoSuchToolError,
)
from .messages import (
    AssistantContentPart,
    AssistantModelMessage,
    DocumentSourcePart,
    ErrorTextOutput,
    FilePart,
    ModelMessage,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    UrlSourcePart,
)
from .provider import CallOptions, FunctionToolSpec, LanguageModel, ProviderResult
from .results import (
    CallWarning,
    FinishReason,
    GeneratedFile,
    GenerateTextResult,
    ResponseMetadata,
    StepResult,
    ToolChoice,
    ToolResult,
    Usage,
    coerce_warnings,
)
from .stream import (
    AbortPart,
    ErrorPart,
    Finish,
    FinishStep,
    FilePartEvent,
    RawPart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    ResponseMetadataPart,
    SourceStreamPart,
    StartStep,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    TextStreamPart,
    ToolErrorEvent,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputStart,
    ToolResultEvent,
)

_SOURCE_TYPES = (UrlSourcePart, DocumentSourcePart)
from .tools import Tool, ToolCallOptions, ToolSet, output_to_model_output
from .transforms import Transform, compose_transforms
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .telemetry import TelemetryArg, TraceContext

# ---------------------------------------------------------------------------
# Stop conditions (AI SDK stopWhen)
# ---------------------------------------------------------------------------

StopCondition = Callable[[list[StepResult]], Union[bool, Awaitable[bool]]]


def step_count_is(count: int) -> StopCondition:
    """Stop after `count` steps (AI SDK stepCountIs)."""

    def condition(steps: list[StepResult]) -> bool:
        return len(steps) >= count

    return condition


def has_tool_call(tool_name: str) -> StopCondition:
    """Stop when the last step called the given tool (AI SDK hasToolCall)."""

    def condition(steps: list[StepResult]) -> bool:
        return any(tc.tool_name == tool_name for tc in steps[-1].tool_calls)

    return condition


async def _is_stopped(
    stop_when: Union[StopCondition, Sequence[StopCondition]],
    steps: list[StepResult],
) -> bool:
    conditions = stop_when if isinstance(stop_when, Sequence) else [stop_when]
    for condition in conditions:
        result = condition(steps)
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return True
    return False


# ---------------------------------------------------------------------------
# Loop control: prepare_step, repair, abort, timeouts (AI SDK parity)
# ---------------------------------------------------------------------------


@dataclass
class PrepareStepResult:
    """Per-step overrides returned by a `prepare_step` callback (AI SDK
    prepareStep). Any field left None keeps the loop's current value. All
    overrides are per-step only and do not mutate canonical loop state, except
    `messages`, which fully replaces the working message list for that step.
    """

    model: Optional[Union[str, "LanguageModel"]] = None
    tools: Optional[ToolSet] = None
    active_tools: Optional[Sequence[str]] = None
    tool_choice: Optional[ToolChoice] = None
    system: Optional[str] = None
    messages: Optional[Sequence[ModelMessage]] = None


# prepare_step signature: called before each step (sync or async).
PrepareStepFn = Callable[..., Union[Optional[PrepareStepResult], Awaitable[Optional[PrepareStepResult]]]]

# repair_tool_call signature (sync or async) -> fixed ToolCallPart or None.
RepairToolCallFn = Callable[..., Union[Optional[ToolCallPart], Awaitable[Optional[ToolCallPart]]]]


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _normalize_messages(messages: Sequence[Any]) -> list[ModelMessage]:
    """Coerce a prepare_step `messages` override (ModelMessage instances or
    plain dicts) into a list of ModelMessage, mirroring standardize_prompt."""
    from ._prompt import _MESSAGE_TYPES
    from .messages import pai_sdk_adapter

    out: list[ModelMessage] = []
    for item in messages:
        if isinstance(item, _MESSAGE_TYPES):
            out.append(item)
        else:
            out.append(pai_sdk_adapter.validate_python(item))
    return out


async def _call_prepare_step(
    prepare_step: Optional[PrepareStepFn],
    *,
    model: LanguageModel,
    step_number: int,
    steps: list[StepResult],
    messages: list[ModelMessage],
) -> Optional[PrepareStepResult]:
    if prepare_step is None:
        return None
    result = prepare_step(
        model=model,
        step_number=step_number,
        steps=steps,
        messages=messages,
    )
    return await _maybe_await(result)


@dataclass
class _StepConfig:
    """Effective per-step configuration after applying prepare_step overrides."""

    model: LanguageModel
    tools: Optional[ToolSet]
    specs: list[FunctionToolSpec]
    tool_choice: Optional[ToolChoice]
    messages: list[ModelMessage]


def _resolve_step_config(
    prep: Optional[PrepareStepResult],
    *,
    base_model: LanguageModel,
    base_tools: Optional[ToolSet],
    base_active_tools: Optional[Sequence[str]],
    base_tool_choice: Optional[ToolChoice],
    working_messages: list[ModelMessage],
) -> _StepConfig:
    """Apply a PrepareStepResult to derive the effective per-step config.

    model/tools/tool_choice/active_tools are per-step only; `messages` and
    `system` build the working message list for this step only (subsequent
    steps resume from canonical history)."""
    model = base_model
    tools = base_tools
    active_tools = base_active_tools
    tool_choice = base_tool_choice
    messages = working_messages

    if prep is not None:
        if prep.model is not None:
            model = _resolve_model(prep.model)
        if prep.tools is not None:
            tools = prep.tools
            active_tools = None  # a replacement tool set ignores the base filter
        if prep.active_tools is not None:
            active_tools = prep.active_tools
        if prep.tool_choice is not None:
            tool_choice = prep.tool_choice
        if prep.messages is not None:
            messages = _normalize_messages(prep.messages)
        if prep.system is not None:
            # Replace any leading system messages for this step only.
            from .messages import SystemModelMessage

            rest = [m for m in messages if not isinstance(m, SystemModelMessage)]
            messages = [SystemModelMessage(content=prep.system), *rest]

    specs = _tool_specs(tools, active_tools)
    return _StepConfig(
        model=model,
        tools=tools,
        specs=specs,
        tool_choice=tool_choice,
        messages=list(messages),
    )


async def _repair_tool_calls(
    content: list[AssistantContentPart],
    tool_calls: list[ToolCallPart],
    tools: Optional[ToolSet],
    repair_tool_call: Optional[RepairToolCallFn],
    messages: list[ModelMessage],
) -> list[ToolCallPart]:
    """For each tool call, detect NoSuchToolError / InvalidToolInputError during
    prep. If `repair_tool_call` is configured, attempt a single repair and
    replace the call in `content` and the returned list so history stays
    consistent. Returns the (possibly repaired) list of tool calls.

    When `repair_tool_call` is None, this is a no-op (unknown tools and invalid
    inputs flow to the normal error-result path)."""
    if repair_tool_call is None or not tools:
        return tool_calls

    available = list(tools.keys())
    repaired: list[ToolCallPart] = []
    for call in tool_calls:
        if call.provider_executed:
            repaired.append(call)
            continue

        error: Optional[Exception] = None
        tool_def = tools.get(call.tool_name)
        if tool_def is None:
            error = NoSuchToolError(call.tool_name, available)
        else:
            try:
                tool_def.parse_input(call.input)
            except InvalidToolInputError as exc:
                error = exc

        if error is None:
            repaired.append(call)
            continue

        fixed = await _maybe_await(
            repair_tool_call(
                tool_call=call,
                tools=tools,
                error=error,
                messages=messages,
            )
        )
        if fixed is None:
            repaired.append(call)  # fall back to error-result behavior
            continue

        # Replace the call in assistant content so history is consistent.
        for i, p in enumerate(content):
            if p is call:
                content[i] = fixed
                break
        repaired.append(fixed)

    return repaired


def _check_abort(abort_signal: Optional[asyncio.Event]) -> bool:
    return abort_signal is not None and abort_signal.is_set()


def _normalize_timeout(
    timeout: Union[float, int, dict, None],
) -> tuple[Optional[float], Optional[float]]:
    """Return (total_seconds, step_seconds) from a float or
    {"total_ms": int, "step_ms": int} dict (either key optional)."""
    if timeout is None:
        return None, None
    if isinstance(timeout, (int, float)):
        return float(timeout), None
    total_ms = timeout.get("total_ms")
    step_ms = timeout.get("step_ms")
    total = total_ms / 1000.0 if total_ms is not None else None
    step = step_ms / 1000.0 if step_ms is not None else None
    return total, step


def _step_budget(
    step_timeout: Optional[float], total_deadline: Optional[float]
) -> Optional[float]:
    """Effective timeout for the next provider call: min(step, remaining total).
    Raises GenerationTimeoutError if the total budget is already exhausted."""
    budgets: list[float] = []
    if step_timeout is not None:
        budgets.append(step_timeout)
    if total_deadline is not None:
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            raise GenerationTimeoutError("total")
        budgets.append(remaining)
    if not budgets:
        return None
    return min(budgets)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_model(model: Union[str, LanguageModel]) -> LanguageModel:
    if isinstance(model, LanguageModel):
        return model
    from .providers import resolve_model_string

    return resolve_model_string(model)


def _tool_specs(
    tools: Optional[ToolSet], active_tools: Optional[Sequence[str]]
) -> list[FunctionToolSpec]:
    if not tools:
        return []
    specs = []
    for name, tool_def in tools.items():
        if active_tools is not None and name not in active_tools:
            continue
        tool_def.name = name
        specs.append(
            FunctionToolSpec(
                name=name,
                description=tool_def.description,
                input_schema=tool_def.json_schema(),
                strict=tool_def.strict,
                provider_options=tool_def.provider_options,
            )
        )
    return specs


async def _execute_tool_calls(
    tool_calls: list[ToolCallPart],
    tools: Optional[ToolSet],
    messages: list[ModelMessage],
    *,
    strict_unknown: bool = False,
) -> list[ToolResult]:
    """Run all executable tool calls concurrently. Errors become error results
    (sent back to the model as error-text), mirroring AI SDK behavior.

    When `strict_unknown` is True (repair_tool_call configured), a call naming a
    tool genuinely absent from the ToolSet becomes a NoSuchToolError result
    instead of being treated as a client-side call. Tools present but without
    `execute` remain client-side regardless."""
    if not tools:
        return []

    available = list(tools.keys())

    async def run_one(call: ToolCallPart) -> Optional[ToolResult]:
        if call.provider_executed:
            return None  # already executed by the provider
        tool_def = tools.get(call.tool_name)
        if tool_def is None:
            if strict_unknown:
                exc = NoSuchToolError(call.tool_name, available)
                return ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    input=call.input,
                    output=exc,
                    model_output=ErrorTextOutput(value=str(exc)),
                    is_error=True,
                )
            return None  # client-side tool: caller handles it
        if tool_def.execute is None:
            return None  # client-side tool: caller handles it
        try:
            parsed = tool_def.parse_input(call.input)
            output = await tool_def.run(
                parsed, ToolCallOptions(tool_call_id=call.tool_call_id, messages=messages)
            )
            return ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                input=call.input,
                output=output,
                model_output=output_to_model_output(tool_def, output),
            )
        except Exception as exc:  # noqa: BLE001 — error feeds back to the model
            return ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                input=call.input,
                output=exc,
                model_output=ErrorTextOutput(value=str(exc)),
                is_error=True,
            )

    results = await asyncio.gather(*(run_one(c) for c in tool_calls))
    return [r for r in results if r is not None]


def _step_messages(
    content: list[AssistantContentPart], tool_results: list[ToolResult]
) -> list[ModelMessage]:
    """Build the assistant (and tool) messages produced by one step."""
    new_messages: list[ModelMessage] = [AssistantModelMessage(content=content)]
    if tool_results:
        new_messages.append(
            ToolModelMessage(
                content=[
                    ToolResultPart(
                        tool_call_id=r.tool_call_id,
                        tool_name=r.tool_name,
                        output=r.model_output or ErrorTextOutput(value="missing output"),
                    )
                    for r in tool_results
                ]
            )
        )
    return new_messages


def _provider_executed_results(
    content: list[AssistantContentPart],
) -> list[ToolResult]:
    """Surface provider-executed ToolResultPart entries (in assistant content)
    as ToolResult entries with provider_executed=True."""
    results: list[ToolResult] = []
    for p in content:
        if isinstance(p, ToolResultPart) and p.provider_executed:
            is_error = getattr(p.output, "type", None) in (
                "error-text",
                "error-json",
            )
            results.append(
                ToolResult(
                    tool_call_id=p.tool_call_id,
                    tool_name=p.tool_name,
                    input=None,
                    output=getattr(p.output, "value", None),
                    model_output=p.output,
                    is_error=is_error,
                    provider_executed=True,
                )
            )
    return results


def _collect_sources(content: list[AssistantContentPart]) -> list:
    return [p for p in content if isinstance(p, _SOURCE_TYPES)]


def _build_step_result(
    content: list[AssistantContentPart],
    tool_results: list[ToolResult],
    finish_reason: FinishReason,
    raw_finish_reason: Optional[str],
    usage: Usage,
    response: ResponseMetadata,
    warnings: list[Any],
    provider_metadata: Optional[dict[str, dict[str, Any]]],
    files: Optional[list[GeneratedFile]] = None,
    request: Any = None,
    request_messages: Optional[list[ModelMessage]] = None,
) -> StepResult:
    text = "".join(p.text for p in content if isinstance(p, TextPart))
    reasoning = [p for p in content if isinstance(p, ReasoningPart)]
    # Provider-executed tool results live in assistant content; surface them.
    all_results = _provider_executed_results(content) + list(tool_results)
    collected_files = files if files is not None else _files_from_content(content)
    return StepResult(
        content=content,
        text=text,
        reasoning=reasoning,
        reasoning_text="".join(p.text for p in reasoning) or None,
        tool_calls=[p for p in content if isinstance(p, ToolCallPart)],
        tool_results=all_results,
        finish_reason=finish_reason,
        raw_finish_reason=raw_finish_reason,
        usage=usage,
        warnings=coerce_warnings(warnings),
        response=response,
        provider_metadata=provider_metadata,
        sources=_collect_sources(content),
        files=collected_files,
        request=request,
        request_messages=list(request_messages or []),
    )


def _files_from_content(content: list[AssistantContentPart]) -> list[GeneratedFile]:
    """Convert FilePart entries in content into GeneratedFile (result.files).

    Mirrors AI SDK: content keeps file parts; result.files exposes GeneratedFile.
    Only inline-byte file parts become GeneratedFile.
    """
    files: list[GeneratedFile] = []
    for p in content:
        if isinstance(p, FilePart) and isinstance(p.data, (bytes, bytearray)):
            files.append(
                GeneratedFile(data=bytes(p.data), media_type=p.media_type)
            )
    return files


def _should_continue(step: StepResult) -> bool:
    """Continue the loop only if the model asked for tools and every
    locally-executed call got a result we can send back. Provider-executed
    calls already have their results in assistant content and do not require a
    local result."""
    if step.finish_reason != "tool-calls" or not step.tool_calls:
        return False
    local_calls = [c for c in step.tool_calls if not c.provider_executed]
    if not local_calls:
        return False
    result_ids = {
        r.tool_call_id for r in step.tool_results if not r.provider_executed
    }
    return all(c.tool_call_id in result_ids for c in local_calls)


async def _with_retry(
    fn: Callable[[], Awaitable[Any]], max_retries: int
) -> Any:
    attempt = 0
    while True:
        try:
            return await fn()
        except APICallError as exc:
            if not exc.is_retryable or attempt >= max_retries:
                raise
            await asyncio.sleep(min(2.0**attempt, 16.0))
            attempt += 1


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------


async def _generate_text_impl(
    *,
    model: Union[str, LanguageModel],
    system: Optional[str] = None,
    prompt: Prompt = None,
    messages: Optional[Sequence[Any]] = None,
    tools: Optional[ToolSet] = None,
    tool_choice: Optional[ToolChoice] = None,
    active_tools: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop_sequences: Optional[list[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    stop_when: Union[StopCondition, Sequence[StopCondition], None] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
    output: Optional[Any] = None,
    on_step_finish: Optional[Callable[[StepResult], Any]] = None,
    prepare_step: Optional[PrepareStepFn] = None,
    repair_tool_call: Optional[RepairToolCallFn] = None,
    abort_signal: Optional[asyncio.Event] = None,
    timeout: Union[float, int, dict, None] = None,
) -> GenerateTextResult:
    """Generate text (and tool calls) — the AI SDK generateText().

    With tools that have `execute`, runs the multi-step tool loop until
    `stop_when` is met (default: a single step, like the AI SDK).

    Pass `output=Output.object(...)` for structured output: it sets
    CallOptions.response_format and parses the final text into `result.output`.

    Loop control (AI SDK parity):
    - `prepare_step(model, step_number, steps, messages)` (sync/async) runs
      before each step and may return a `PrepareStepResult` overriding model,
      tools, active_tools, tool_choice, system, or messages for that step only.
    - `repair_tool_call(tool_call, tools, error, messages)` (sync/async) is
      invoked on NoSuchToolError / InvalidToolInputError and may return a fixed
      ToolCallPart (retried once) or None (fall back to error result).
    - `abort_signal` (asyncio.Event): when set, the loop stops; raises AbortError.
    - `timeout`: float seconds (total) or {"total_ms", "step_ms"} dict.
    """
    resolved = _resolve_model(model)
    working_messages = standardize_prompt(system=system, prompt=prompt, messages=messages)
    stop = stop_when if stop_when is not None else step_count_is(1)
    response_format = output.response_format() if output is not None else None
    total_timeout, step_timeout = _normalize_timeout(timeout)
    total_deadline = (
        time.monotonic() + total_timeout if total_timeout is not None else None
    )

    steps: list[StepResult] = []
    generated_messages: list[ModelMessage] = []
    total_usage = Usage()

    while True:
        if _check_abort(abort_signal):
            raise AbortError()

        prep = await _call_prepare_step(
            prepare_step,
            model=resolved,
            step_number=len(steps),
            steps=steps,
            messages=working_messages,
        )
        cfg = _resolve_step_config(
            prep,
            base_model=resolved,
            base_tools=tools,
            base_active_tools=active_tools,
            base_tool_choice=tool_choice,
            working_messages=working_messages,
        )

        options = CallOptions(
            prompt=cfg.messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop_sequences=stop_sequences,
            seed=seed,
            tools=cfg.specs,
            tool_choice=cfg.tool_choice,
            response_format=response_format,
            headers=headers,
            provider_options=provider_options or {},
        )

        budget = _step_budget(step_timeout, total_deadline)

        async def _do_call(_opts: CallOptions = options, _model: LanguageModel = cfg.model) -> ProviderResult:
            return await _with_retry(lambda: _model.do_generate(_opts), max_retries)

        try:
            if budget is not None:
                result: ProviderResult = await asyncio.wait_for(_do_call(), budget)
            else:
                result = await _do_call()
        except asyncio.TimeoutError:
            budget_kind = (
                "total"
                if total_deadline is not None and time.monotonic() >= total_deadline
                else "step"
            )
            raise GenerationTimeoutError(budget_kind) from None

        tool_calls = [p for p in result.content if isinstance(p, ToolCallPart)]
        tool_calls = await _repair_tool_calls(
            result.content, tool_calls, cfg.tools, repair_tool_call, cfg.messages
        )
        tool_results = await _execute_tool_calls(
            tool_calls,
            cfg.tools,
            cfg.messages,
            strict_unknown=repair_tool_call is not None,
        )

        step = _build_step_result(
            content=result.content,
            tool_results=tool_results,
            finish_reason=result.finish_reason,
            raw_finish_reason=result.raw_finish_reason,
            usage=result.usage,
            response=result.response,
            warnings=result.warnings,
            provider_metadata=result.provider_metadata,
            request=result.request,
            request_messages=cfg.messages,
        )
        steps.append(step)
        total_usage = total_usage + step.usage

        new_messages = _step_messages(result.content, tool_results)
        generated_messages.extend(new_messages)
        working_messages.extend(new_messages)

        if _check_abort(abort_signal):
            raise AbortError()

        if on_step_finish is not None:
            cb = on_step_finish(step)
            if asyncio.iscoroutine(cb):
                await cb

        if await _is_stopped(stop, steps) or not _should_continue(step):
            break

    final = steps[-1]
    response = ResponseMetadata(
        id=final.response.id,
        model_id=final.response.model_id,
        timestamp=final.response.timestamp,
        headers=final.response.headers,
        body=final.response.body,
        messages=generated_messages,
    )

    parsed_output: Any = None
    if output is not None and getattr(output, "type", None) == "object":
        from .errors import NoObjectGeneratedError

        try:
            parsed_output = output.parse(final.text)
        except NoObjectGeneratedError as exc:
            exc.finish_reason = final.finish_reason
            exc.usage = final.usage
            raise

    all_sources: list = []
    all_files: list[GeneratedFile] = []
    for s in steps:
        all_sources.extend(s.sources)
        all_files.extend(s.files)

    return GenerateTextResult(
        text=final.text,
        content=final.content,
        reasoning=final.reasoning,
        reasoning_text=final.reasoning_text,
        tool_calls=final.tool_calls,
        tool_results=final.tool_results,
        finish_reason=final.finish_reason,
        raw_finish_reason=final.raw_finish_reason,
        usage=final.usage,
        total_usage=total_usage,
        steps=steps,
        response=response,
        warnings=final.warnings,
        provider_metadata=final.provider_metadata,
        output=parsed_output,
        sources=all_sources,
        files=all_files,
        request=final.request,
    )


# ---------------------------------------------------------------------------
# stream_text
# ---------------------------------------------------------------------------


class StreamTextResult:
    """The result of stream_text().

    Iterate `text_stream` (str deltas) or `full_stream` (TextStreamPart),
    both can be consumed multiple times/concurrently. Aggregate values are
    awaitable properties that resolve when the stream finishes:

        result = stream_text(model=..., prompt=...)
        async for delta in result.text_stream:
            print(delta, end="")
        print(await result.usage)
    """

    def __init__(
        self,
        run_step_loop: Callable[["StreamTextResult"], Awaitable[None]],
        on_chunk: Optional[Callable[[TextStreamPart], Any]] = None,
        on_error: Optional[Callable[[Any], Any]] = None,
        on_step_finish: Optional[Callable[[StepResult], Any]] = None,
        on_finish: Optional[Callable[["StreamTextResult"], Any]] = None,
        on_abort: Optional[Callable[["StreamTextResult"], Any]] = None,
        output: Optional[Any] = None,
        transform: Optional[Transform] = None,
        abort_signal: Optional[asyncio.Event] = None,
    ) -> None:
        self._run_step_loop = run_step_loop
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._on_step_finish_cb = on_step_finish
        self._on_finish = on_finish
        self._on_abort = on_abort
        self._output = output
        self._transform = transform
        # The abort event the drive loop checks; .abort() also sets it. Accept a
        # caller-supplied event so a pre-set signal aborts the run immediately.
        self._abort_event = abort_signal if abort_signal is not None else asyncio.Event()

        self._parts: list[TextStreamPart] = []
        self._cond: Optional[asyncio.Condition] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._chunk_task: Optional[asyncio.Task[None]] = None
        self._error: Optional[BaseException] = None
        self._aborted = False

        # Filled by the driver:
        self.steps: list[StepResult] = []
        self._generated_messages: list[ModelMessage] = []
        self._total_usage = Usage()
        self._finish_reason: FinishReason = "unknown"
        self._raw_finish_reason: Optional[str] = None

    # -- public abort -------------------------------------------------------

    def abort(self, reason: Optional[str] = None) -> None:
        """Abort the run (AI SDK abort). The drive loop stops between provider
        parts/steps, emits an AbortPart, and aggregates raise AbortError.

        Note: a blocked provider SDK read cannot be interrupted; the abort takes
        effect at the next part/step boundary."""
        self._abort_reason = reason
        self._abort_event.set()

    _abort_reason: Optional[str] = None

    @property
    def aborted(self) -> Awaitable[bool]:
        async def get() -> bool:
            await self._wait_done()
            return self._aborted

        return get()

    # -- plumbing -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._task is None:
            self._cond = asyncio.Condition()
            self._task = asyncio.get_running_loop().create_task(self._drive())
            if self._on_chunk is not None:
                self._chunk_task = asyncio.get_running_loop().create_task(
                    self._run_on_chunk()
                )

    async def _drive(self) -> None:
        try:
            await self._run_step_loop(self)
        except AbortError as exc:
            self._aborted = True
            self._error = exc
            await self._emit(AbortPart(reason=getattr(exc, "reason", None)))
            if self._on_abort is not None:
                cb = self._on_abort(self)
                if asyncio.iscoroutine(cb):
                    await cb
        except BaseException as exc:  # noqa: BLE001 — surfaced as error part
            self._error = exc
            await self._emit(ErrorPart(error=exc))
            if self._on_error is not None:
                cb = self._on_error(exc)
                if asyncio.iscoroutine(cb):
                    await cb
        finally:
            assert self._cond is not None
            async with self._cond:
                self._parts.append(None)  # type: ignore[arg-type] — sentinel
                self._cond.notify_all()
            if self._on_finish is not None and self._error is None:
                cb = self._on_finish(self)
                if asyncio.iscoroutine(cb):
                    await cb

    async def _emit(self, part: TextStreamPart) -> None:
        assert self._cond is not None
        async with self._cond:
            self._parts.append(part)
            self._cond.notify_all()

    async def _run_on_chunk(self) -> None:
        """Drive on_chunk from the transformed stream so it observes the same
        parts subscribers do (AI SDK applies transforms before on_chunk)."""
        async for part in self._subscribe():
            assert self._on_chunk is not None
            cb = self._on_chunk(part)
            if asyncio.iscoroutine(cb):
                await cb

    async def _subscribe_raw(self) -> AsyncIterator[TextStreamPart]:
        self._ensure_started()
        assert self._cond is not None
        index = 0
        while True:
            async with self._cond:
                while index >= len(self._parts):
                    await self._cond.wait()
                part = self._parts[index]
            index += 1
            if part is None:  # sentinel: stream complete
                return
            yield part

    def _subscribe(self) -> AsyncIterator[TextStreamPart]:
        """Subscribe to the (transformed) stream — all public stream consumers
        and on_chunk observe transformed parts. The per-step content and the
        awaitable aggregates are built by the drive loop from raw provider parts
        and are NOT affected by transforms (see stream_text docstring)."""
        if self._transform is None:
            return self._subscribe_raw()
        return self._transform(self._subscribe_raw())

    async def _wait_done(self) -> None:
        self._ensure_started()
        assert self._task is not None
        await asyncio.shield(self._task)
        if self._chunk_task is not None:
            await asyncio.shield(self._chunk_task)
        if self._error is not None:
            raise self._error

    # -- streams ------------------------------------------------------------

    @property
    def full_stream(self) -> AsyncIterator[TextStreamPart]:
        return self._subscribe()

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            async for part in self._subscribe():
                if isinstance(part, TextDelta):
                    yield part.text

        return gen()

    async def consume_stream(self) -> None:
        """Drain the stream without processing parts."""
        await self._wait_done()

    # -- awaitable aggregates ------------------------------------------------

    async def _final_step(self) -> StepResult:
        await self._wait_done()
        return self.steps[-1]

    @property
    def text(self) -> Awaitable[str]:
        async def get() -> str:
            return (await self._final_step()).text

        return get()

    @property
    def reasoning_text(self) -> Awaitable[Optional[str]]:
        async def get() -> Optional[str]:
            return (await self._final_step()).reasoning_text

        return get()

    @property
    def content(self) -> Awaitable[list[AssistantContentPart]]:
        async def get() -> list[AssistantContentPart]:
            return (await self._final_step()).content

        return get()

    @property
    def tool_calls(self) -> Awaitable[list[ToolCallPart]]:
        async def get() -> list[ToolCallPart]:
            return (await self._final_step()).tool_calls

        return get()

    @property
    def tool_results(self) -> Awaitable[list[ToolResult]]:
        async def get() -> list[ToolResult]:
            return (await self._final_step()).tool_results

        return get()

    @property
    def sources(self) -> Awaitable[list]:
        async def get() -> list:
            await self._wait_done()
            out: list = []
            for s in self.steps:
                out.extend(s.sources)
            return out

        return get()

    @property
    def files(self) -> Awaitable[list[GeneratedFile]]:
        async def get() -> list[GeneratedFile]:
            await self._wait_done()
            out: list[GeneratedFile] = []
            for s in self.steps:
                out.extend(s.files)
            return out

        return get()

    @property
    def finish_reason(self) -> Awaitable[FinishReason]:
        async def get() -> FinishReason:
            await self._wait_done()
            return self._finish_reason

        return get()

    @property
    def usage(self) -> Awaitable[Usage]:
        async def get() -> Usage:
            return (await self._final_step()).usage

        return get()

    @property
    def total_usage(self) -> Awaitable[Usage]:
        async def get() -> Usage:
            await self._wait_done()
            return self._total_usage

        return get()

    @property
    def response(self) -> Awaitable[ResponseMetadata]:
        async def get() -> ResponseMetadata:
            final = (await self._final_step()).response
            return ResponseMetadata(
                id=final.id,
                model_id=final.model_id,
                timestamp=final.timestamp,
                headers=final.headers,
                messages=self._generated_messages,
            )

        return get()

    @property
    def all_steps(self) -> Awaitable[list[StepResult]]:
        async def get() -> list[StepResult]:
            await self._wait_done()
            return self.steps

        return get()

    # -- structured output ---------------------------------------------------

    @property
    def output(self) -> Awaitable[Any]:
        """Parse the final text into the structured object. Raises
        NoObjectGeneratedError on parse/validation failure."""

        async def get() -> Any:
            from .errors import NoObjectGeneratedError

            final = await self._final_step()
            if self._output is None or getattr(self._output, "type", None) != "object":
                return None
            try:
                return self._output.parse(final.text)
            except NoObjectGeneratedError as exc:
                exc.finish_reason = final.finish_reason
                exc.usage = final.usage
                raise

        return get()

    @property
    def partial_output_stream(self) -> AsyncIterator[Any]:
        """Yield successively-parsed partial objects as text accumulates.

        Accumulates text deltas, runs parse_partial_json, and yields the parsed
        value whenever it changes (raw dicts — partials are not validated
        against the schema)."""

        async def gen() -> AsyncIterator[Any]:
            from .output import parse_partial_json

            accumulated = ""
            previous: Any = None
            has_previous = False
            async for part in self._subscribe():
                if not isinstance(part, TextDelta):
                    continue
                accumulated += part.text
                value, state = parse_partial_json(accumulated)
                if state == "failed-parse" or value is None:
                    continue
                if not has_previous or value != previous:
                    previous = value
                    has_previous = True
                    yield value

        return gen()


def _stream_text_impl(
    *,
    model: Union[str, LanguageModel],
    system: Optional[str] = None,
    prompt: Prompt = None,
    messages: Optional[Sequence[Any]] = None,
    tools: Optional[ToolSet] = None,
    tool_choice: Optional[ToolChoice] = None,
    active_tools: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop_sequences: Optional[list[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    stop_when: Union[StopCondition, Sequence[StopCondition], None] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
    output: Optional[Any] = None,
    include_raw_chunks: bool = False,
    on_chunk: Optional[Callable[[TextStreamPart], Any]] = None,
    on_error: Optional[Callable[[Any], Any]] = None,
    on_step_finish: Optional[Callable[[StepResult], Any]] = None,
    on_finish: Optional[Callable[[StreamTextResult], Any]] = None,
    on_abort: Optional[Callable[[StreamTextResult], Any]] = None,
    prepare_step: Optional[PrepareStepFn] = None,
    repair_tool_call: Optional[RepairToolCallFn] = None,
    abort_signal: Optional[asyncio.Event] = None,
    timeout: Union[float, int, dict, None] = None,
    transform: Union[Transform, list, None] = None,
) -> StreamTextResult:
    """Stream text (and tool calls) — the AI SDK streamText().

    Returns immediately; work starts on first consumption/await. Errors are
    emitted as `error` parts on full_stream (and raised when awaiting
    aggregate properties).

    Pass `output=Output.object(...)` for structured output: it sets
    CallOptions.response_format and enables `partial_output_stream` and the
    awaitable `output`.

    Loop control (AI SDK parity):
    - `prepare_step` / `repair_tool_call` — see generate_text.
    - `abort_signal` (asyncio.Event) or `StreamTextResult.abort()`: stops the
      run between provider parts/steps, emits an AbortPart, ends WITHOUT a finish
      part; aggregates raise AbortError and `on_abort` fires. A blocked provider
      read cannot be interrupted — abort takes effect at the next boundary.
    - `timeout`: float seconds (total) or {"total_ms", "step_ms"} dict; on expiry
      an ErrorPart is emitted and aggregates re-raise GenerationTimeoutError.
    - `transform`: one transform or a list (composed in order). Transforms are
      applied at the subscription layer; full_stream/text_stream/
      partial_output_stream and on_chunk observe transformed parts. The per-step
      content and awaitable aggregates are computed from raw provider parts and
      are NOT transformed.
    """
    resolved = _resolve_model(model)
    initial_messages = standardize_prompt(system=system, prompt=prompt, messages=messages)
    stop = stop_when if stop_when is not None else step_count_is(1)
    response_format = output.response_format() if output is not None else None
    total_timeout, step_timeout = _normalize_timeout(timeout)
    composed_transform = compose_transforms(transform)

    async def run_step_loop(result: StreamTextResult) -> None:
        working_messages = list(initial_messages)
        await result._emit(StreamStart())

        total_deadline = (
            time.monotonic() + total_timeout if total_timeout is not None else None
        )
        abort_event = result._abort_event

        while True:
            if abort_event.is_set():
                raise AbortError(reason=result._abort_reason)

            prep = await _call_prepare_step(
                prepare_step,
                model=resolved,
                step_number=len(result.steps),
                steps=result.steps,
                messages=working_messages,
            )
            cfg = _resolve_step_config(
                prep,
                base_model=resolved,
                base_tools=tools,
                base_active_tools=active_tools,
                base_tool_choice=tool_choice,
                working_messages=working_messages,
            )
            step_tools = cfg.tools

            options = CallOptions(
                prompt=cfg.messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop_sequences=stop_sequences,
                seed=seed,
                tools=cfg.specs,
                tool_choice=cfg.tool_choice,
                response_format=response_format,
                headers=headers,
                provider_options=provider_options or {},
                include_raw_chunks=include_raw_chunks,
            )
            await result._emit(StartStep())

            budget = _step_budget(step_timeout, total_deadline)
            step_deadline = (
                time.monotonic() + budget if budget is not None else None
            )

            def _timed_out() -> bool:
                return step_deadline is not None and time.monotonic() >= step_deadline

            # -- consume one provider stream -------------------------------
            content: list[AssistantContentPart] = []
            open_blocks: dict[str, AssistantContentPart] = {}
            finish_reason: FinishReason = "unknown"
            raw_finish_reason: Optional[str] = None
            usage = Usage()
            response_meta = ResponseMetadata()
            provider_metadata: Optional[dict[str, dict[str, Any]]] = None
            step_files: list[GeneratedFile] = []
            step_request: Any = None

            async for part in cfg.model.do_stream(options):
                # Abort/timeout are checked between provider parts; a blocked
                # SDK read cannot be interrupted (documented in the docstring).
                if abort_event.is_set():
                    raise AbortError(reason=result._abort_reason)
                if _timed_out():
                    budget_kind = (
                        "total"
                        if total_deadline is not None
                        and time.monotonic() >= total_deadline
                        else "step"
                    )
                    raise GenerationTimeoutError(budget_kind)
                if isinstance(part, ResponseMetadataPart):
                    response_meta.id = part.id or response_meta.id
                    response_meta.model_id = part.model_id or response_meta.model_id
                    if part.request is not None:
                        step_request = part.request
                    continue
                if isinstance(part, Finish):
                    finish_reason = part.finish_reason
                    raw_finish_reason = part.raw_finish_reason
                    usage = part.total_usage
                    if part.provider_metadata is not None:
                        provider_metadata = part.provider_metadata
                    continue
                if isinstance(part, ErrorPart):
                    raise part.error if isinstance(
                        part.error, BaseException
                    ) else APICallError(str(part.error))

                if isinstance(part, TextStart):
                    block = TextPart(text="")
                    open_blocks[f"text:{part.id}"] = block
                    content.append(block)
                elif isinstance(part, TextDelta):
                    block = open_blocks.get(f"text:{part.id}")
                    if block is None:
                        block = TextPart(text="")
                        open_blocks[f"text:{part.id}"] = block
                        content.append(block)
                    block.text += part.text  # type: ignore[union-attr]
                elif isinstance(part, ReasoningStart):
                    block = ReasoningPart(text="")
                    open_blocks[f"reasoning:{part.id}"] = block
                    content.append(block)
                elif isinstance(part, ReasoningDelta):
                    block = open_blocks.get(f"reasoning:{part.id}")
                    if block is None:
                        block = ReasoningPart(text="")
                        open_blocks[f"reasoning:{part.id}"] = block
                        content.append(block)
                    block.text += part.text  # type: ignore[union-attr]
                elif isinstance(part, ReasoningEnd) and part.provider_metadata:
                    block = open_blocks.get(f"reasoning:{part.id}")
                    if block is not None:
                        block.provider_options = part.provider_metadata
                elif isinstance(part, ToolCallPart):
                    content.append(part)
                elif isinstance(part, FilePartEvent):
                    content.append(FilePart(data=part.data, media_type=part.media_type))
                    step_files.append(
                        GeneratedFile(data=part.data, media_type=part.media_type)
                    )
                elif isinstance(part, SourceStreamPart):
                    content.append(part.source)
                elif isinstance(part, ToolResultEvent):
                    # Provider-executed tool result yielded directly from the
                    # provider stream: surface it as assistant content so it is
                    # recorded on the step (via _provider_executed_results) and
                    # replayed in history, but not re-sent as a tool message.
                    content.append(
                        ToolResultPart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            output=part.model_output
                            or ErrorTextOutput(value="missing output"),
                            provider_executed=True,
                        )
                    )

                await result._emit(part)

            # -- execute tools, finish the step ----------------------------
            tool_calls = [p for p in content if isinstance(p, ToolCallPart)]
            tool_calls = await _repair_tool_calls(
                content, tool_calls, step_tools, repair_tool_call, cfg.messages
            )
            tool_results = await _execute_tool_calls(
                tool_calls,
                step_tools,
                cfg.messages,
                strict_unknown=repair_tool_call is not None,
            )
            for tr in tool_results:
                event: TextStreamPart
                if tr.is_error:
                    event = ToolErrorEvent(
                        tool_call_id=tr.tool_call_id,
                        tool_name=tr.tool_name,
                        input=tr.input,
                        error=tr.output,
                    )
                else:
                    event = ToolResultEvent(
                        tool_call_id=tr.tool_call_id,
                        tool_name=tr.tool_name,
                        input=tr.input,
                        output=tr.output,
                        model_output=tr.model_output,
                    )
                await result._emit(event)

            step = _build_step_result(
                content=content,
                tool_results=tool_results,
                finish_reason=finish_reason,
                raw_finish_reason=raw_finish_reason,
                usage=usage,
                response=response_meta,
                warnings=[],
                provider_metadata=provider_metadata,
                files=step_files,
                request=step_request,
                request_messages=cfg.messages,
            )
            result.steps.append(step)
            result._total_usage = result._total_usage + usage

            new_messages = _step_messages(content, tool_results)
            result._generated_messages.extend(new_messages)
            working_messages.extend(new_messages)

            await result._emit(
                FinishStep(
                    response=response_meta,
                    usage=usage,
                    finish_reason=finish_reason,
                    raw_finish_reason=raw_finish_reason,
                    provider_metadata=provider_metadata,
                    request=step_request,
                )
            )
            if on_step_finish is not None:
                cb = on_step_finish(step)
                if asyncio.iscoroutine(cb):
                    await cb

            if abort_event.is_set():
                raise AbortError(reason=result._abort_reason)

            if await _is_stopped(stop, result.steps) or not _should_continue(step):
                break

        final = result.steps[-1]
        result._finish_reason = final.finish_reason
        result._raw_finish_reason = final.raw_finish_reason
        await result._emit(
            Finish(
                finish_reason=final.finish_reason,
                raw_finish_reason=final.raw_finish_reason,
                total_usage=result._total_usage,
            )
        )

    return StreamTextResult(
        run_step_loop,
        on_chunk=on_chunk,
        on_error=on_error,
        on_step_finish=on_step_finish,
        on_finish=on_finish,
        on_abort=on_abort,
        output=output,
        transform=composed_transform,
        abort_signal=abort_signal,
    )


def generate_id() -> str:
    """Generate a unique id (for tool calls etc.)."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Public entry points: telemetry-integrated wrappers
# ---------------------------------------------------------------------------


def _telemetry_inputs(ctx, system, prompt, input_messages):
    if ctx.inputs is not None:
        return ctx.inputs
    from .trace import _default_inputs

    return _default_inputs(
        {"system": system, "prompt": prompt, "messages": input_messages},
        input_messages,
    )


async def generate_text(
    *,
    model: Union[str, LanguageModel],
    system: Optional[str] = None,
    prompt: Prompt = None,
    messages: Optional[Sequence[Any]] = None,
    tools: Optional[ToolSet] = None,
    tool_choice: Optional[ToolChoice] = None,
    active_tools: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop_sequences: Optional[list[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    stop_when: Union[StopCondition, Sequence[StopCondition], None] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
    output: Optional[Any] = None,
    on_step_finish: Optional[Callable[[StepResult], Any]] = None,
    prepare_step: Optional[PrepareStepFn] = None,
    repair_tool_call: Optional[RepairToolCallFn] = None,
    abort_signal: Optional[asyncio.Event] = None,
    timeout: Union[float, int, dict, None] = None,
    telemetry: "TelemetryArg" = None,
    trace_context: Optional["TraceContext"] = None,
) -> GenerateTextResult:
    """Generate text (and tool calls) — the AI SDK generateText().

    See `_generate_text_impl` for the full loop-control docs. Telemetry: when
    trace sinks are connected (`configure_telemetry(...)`, the `telemetry()`
    context manager, or a per-call `telemetry=` argument), the call emits a
    replayable `Trace` as a side effect — success or failure (failed calls
    also carry it as `exc.trace`). `telemetry=False` disables emission for
    this call; `trace_context=` enriches the span with semantic
    inputs/metadata and span relationships (Prompt.generate sets it
    automatically).
    """
    from .telemetry import TraceContext, emit_trace, resolve_sinks

    impl_kwargs = dict(
        model=model,
        system=system,
        prompt=prompt,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        active_tools=active_tools,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop_sequences=stop_sequences,
        seed=seed,
        max_retries=max_retries,
        headers=headers,
        stop_when=stop_when,
        provider_options=provider_options,
        output=output,
        on_step_finish=on_step_finish,
        prepare_step=prepare_step,
        repair_tool_call=repair_tool_call,
        abort_signal=abort_signal,
        timeout=timeout,
    )
    sinks = resolve_sinks(telemetry)
    if not sinks:
        return await _generate_text_impl(**impl_kwargs)

    from .trace import build_failed_trace, build_trace

    ctx = trace_context or TraceContext()
    input_messages = list(
        standardize_prompt(system=system, prompt=prompt, messages=messages)
    )
    inputs = _telemetry_inputs(ctx, system, prompt, input_messages)
    try:
        result = await _generate_text_impl(**impl_kwargs)
    except BaseException as exc:
        failed = build_failed_trace(
            inputs=inputs,
            input_messages=input_messages,
            error=exc,
            metadata=ctx.metadata,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            root_span_id=ctx.root_span_id,
            parent_span_id=ctx.parent_span_id,
        )
        await emit_trace(failed, sinks)
        setattr(exc, "trace", failed)
        raise
    trace = build_trace(
        inputs=inputs,
        result=result,
        input_messages=input_messages,
        outputs=ctx.outputs,
        metadata=ctx.metadata,
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        root_span_id=ctx.root_span_id,
        parent_span_id=ctx.parent_span_id,
    )
    await emit_trace(trace, sinks)
    return result


def stream_text(
    *,
    model: Union[str, LanguageModel],
    system: Optional[str] = None,
    prompt: Prompt = None,
    messages: Optional[Sequence[Any]] = None,
    tools: Optional[ToolSet] = None,
    tool_choice: Optional[ToolChoice] = None,
    active_tools: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop_sequences: Optional[list[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    stop_when: Union[StopCondition, Sequence[StopCondition], None] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
    output: Optional[Any] = None,
    include_raw_chunks: bool = False,
    on_chunk: Optional[Callable[[TextStreamPart], Any]] = None,
    on_error: Optional[Callable[[Any], Any]] = None,
    on_step_finish: Optional[Callable[[StepResult], Any]] = None,
    on_finish: Optional[Callable[[StreamTextResult], Any]] = None,
    on_abort: Optional[Callable[[StreamTextResult], Any]] = None,
    prepare_step: Optional[PrepareStepFn] = None,
    repair_tool_call: Optional[RepairToolCallFn] = None,
    abort_signal: Optional[asyncio.Event] = None,
    timeout: Union[float, int, dict, None] = None,
    transform: Union[Transform, list, None] = None,
    telemetry: "TelemetryArg" = None,
    trace_context: Optional["TraceContext"] = None,
) -> StreamTextResult:
    """Stream text (and tool calls) — the AI SDK streamText().

    See `_stream_text_impl` for the full streaming docs. Telemetry works like
    generate_text: with sinks connected, a replayable `Trace` is emitted when
    the stream finishes (or fails/aborts, as a failed-trace span).
    """
    from .telemetry import TraceContext, emit_trace, resolve_sinks

    impl_kwargs = dict(
        model=model,
        system=system,
        prompt=prompt,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        active_tools=active_tools,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop_sequences=stop_sequences,
        seed=seed,
        max_retries=max_retries,
        headers=headers,
        stop_when=stop_when,
        provider_options=provider_options,
        output=output,
        include_raw_chunks=include_raw_chunks,
        on_chunk=on_chunk,
        on_error=on_error,
        on_step_finish=on_step_finish,
        on_finish=on_finish,
        on_abort=on_abort,
        prepare_step=prepare_step,
        repair_tool_call=repair_tool_call,
        abort_signal=abort_signal,
        timeout=timeout,
        transform=transform,
    )
    sinks = resolve_sinks(telemetry)
    if not sinks:
        return _stream_text_impl(**impl_kwargs)

    from .serialize import dump_messages
    from .trace import build_failed_trace, build_trace_from_messages

    ctx = trace_context or TraceContext()
    input_messages = list(
        standardize_prompt(system=system, prompt=prompt, messages=messages)
    )
    inputs = _telemetry_inputs(ctx, system, prompt, input_messages)
    holder: dict[str, StreamTextResult] = {}
    user_on_finish = on_finish
    user_on_error = on_error
    user_on_abort = on_abort

    async def _emit_success(result: StreamTextResult) -> None:
        # Runs inside the stream's finish hook, i.e. within the drive task —
        # the awaitable aggregates would deadlock here (they await that very
        # task), so read the settled per-step state directly.
        steps = result.steps
        final = steps[-1] if steps else None
        outputs = ctx.outputs
        if outputs is None:
            outputs = {
                "text": final.text if final is not None else "",
                "finish_reason": result._finish_reason,
            }
            if output is not None and final is not None:
                try:
                    outputs["object"] = output.parse(final.text)
                except Exception:  # noqa: BLE001 — parse failure is not a
                    pass           # telemetry failure; the caller sees it.
            step_tool_calls = [c for st in steps for c in st.tool_calls]
            step_tool_results = [r for st in steps for r in st.tool_results]
            if step_tool_calls:
                outputs["tool_calls"] = step_tool_calls
            if step_tool_results:
                outputs["tool_results"] = step_tool_results
        response = final.response if final is not None else None
        metadata = {
            "response": {
                "id": response.id if response is not None else None,
                "model_id": response.model_id if response is not None else None,
                "timestamp": response.timestamp if response is not None else None,
                "headers": response.headers if response is not None else None,
            },
            "finish_reason": result._finish_reason,
            "raw_finish_reason": final.raw_finish_reason if final is not None else None,
            "step_finish_reasons": [st.finish_reason for st in steps],
            "step_request_messages": [
                dump_messages(st.request_messages) for st in steps
            ],
            "warnings": [w for st in steps for w in st.warnings],
            "provider_metadata": final.provider_metadata if final is not None else None,
            **ctx.metadata,
        }
        trace = build_trace_from_messages(
            inputs=inputs,
            input_messages=input_messages,
            response_messages=list(result._generated_messages),
            usage=result._total_usage,
            outputs=outputs,
            metadata=metadata,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            root_span_id=ctx.root_span_id,
            parent_span_id=ctx.parent_span_id,
        )
        await emit_trace(trace, sinks)

    async def _emit_failure(error: BaseException) -> None:
        result = holder.get("result")
        failed = build_failed_trace(
            inputs=inputs,
            input_messages=input_messages,
            error=error,
            response_messages=list(getattr(result, "_generated_messages", []) or []),
            usage=getattr(result, "_total_usage", None),
            metadata=ctx.metadata,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            root_span_id=ctx.root_span_id,
            parent_span_id=ctx.parent_span_id,
        )
        await emit_trace(failed, sinks)

    async def _finish_hook(result: StreamTextResult) -> None:
        if user_on_finish is not None:
            outcome = user_on_finish(result)
            if asyncio.iscoroutine(outcome):
                await outcome
        await _emit_success(result)

    async def _error_hook(error: Any) -> None:
        if user_on_error is not None:
            outcome = user_on_error(error)
            if asyncio.iscoroutine(outcome):
                await outcome
        if isinstance(error, BaseException):
            await _emit_failure(error)

    async def _abort_hook(result: StreamTextResult) -> None:
        if user_on_abort is not None:
            outcome = user_on_abort(result)
            if asyncio.iscoroutine(outcome):
                await outcome
        await _emit_failure(AbortError())

    impl_kwargs["on_finish"] = _finish_hook
    impl_kwargs["on_error"] = _error_hook
    impl_kwargs["on_abort"] = _abort_hook
    result = _stream_text_impl(**impl_kwargs)
    holder["result"] = result
    return result
