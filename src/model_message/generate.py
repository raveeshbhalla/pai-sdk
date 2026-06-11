"""generate_text() and stream_text() — the core AI SDK functions in Python."""

from __future__ import annotations

import asyncio
import uuid
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
from .errors import APICallError
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
) -> list[ToolResult]:
    """Run all executable tool calls concurrently. Errors become error results
    (sent back to the model as error-text), mirroring AI SDK behavior."""
    if not tools:
        return []

    async def run_one(call: ToolCallPart) -> Optional[ToolResult]:
        if call.provider_executed:
            return None  # already executed by the provider
        tool_def = tools.get(call.tool_name)
        if tool_def is None or tool_def.execute is None:
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
) -> GenerateTextResult:
    """Generate text (and tool calls) — the AI SDK generateText().

    With tools that have `execute`, runs the multi-step tool loop until
    `stop_when` is met (default: a single step, like the AI SDK).

    Pass `output=Output.object(...)` for structured output: it sets
    CallOptions.response_format and parses the final text into `result.output`.
    """
    resolved = _resolve_model(model)
    working_messages = standardize_prompt(system=system, prompt=prompt, messages=messages)
    stop = stop_when if stop_when is not None else step_count_is(1)
    specs = _tool_specs(tools, active_tools)
    response_format = output.response_format() if output is not None else None

    steps: list[StepResult] = []
    generated_messages: list[ModelMessage] = []
    total_usage = Usage()

    while True:
        options = CallOptions(
            prompt=list(working_messages),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop_sequences=stop_sequences,
            seed=seed,
            tools=specs,
            tool_choice=tool_choice,
            response_format=response_format,
            headers=headers,
            provider_options=provider_options or {},
        )
        result: ProviderResult = await _with_retry(
            lambda: resolved.do_generate(options), max_retries
        )

        tool_calls = [p for p in result.content if isinstance(p, ToolCallPart)]
        tool_results = await _execute_tool_calls(tool_calls, tools, working_messages)

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
        )
        steps.append(step)
        total_usage = total_usage + step.usage

        new_messages = _step_messages(result.content, tool_results)
        generated_messages.extend(new_messages)
        working_messages.extend(new_messages)

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
        output: Optional[Any] = None,
    ) -> None:
        self._run_step_loop = run_step_loop
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._on_step_finish_cb = on_step_finish
        self._on_finish = on_finish
        self._output = output

        self._parts: list[TextStreamPart] = []
        self._cond: Optional[asyncio.Condition] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._error: Optional[BaseException] = None

        # Filled by the driver:
        self.steps: list[StepResult] = []
        self._generated_messages: list[ModelMessage] = []
        self._total_usage = Usage()
        self._finish_reason: FinishReason = "unknown"
        self._raw_finish_reason: Optional[str] = None

    # -- plumbing -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._task is None:
            self._cond = asyncio.Condition()
            self._task = asyncio.get_running_loop().create_task(self._drive())

    async def _drive(self) -> None:
        try:
            await self._run_step_loop(self)
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
        if self._on_chunk is not None:
            cb = self._on_chunk(part)
            if asyncio.iscoroutine(cb):
                await cb

    async def _subscribe(self) -> AsyncIterator[TextStreamPart]:
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

    async def _wait_done(self) -> None:
        self._ensure_started()
        assert self._task is not None
        await asyncio.shield(self._task)
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
) -> StreamTextResult:
    """Stream text (and tool calls) — the AI SDK streamText().

    Returns immediately; work starts on first consumption/await. Errors are
    emitted as `error` parts on full_stream (and raised when awaiting
    aggregate properties).

    Pass `output=Output.object(...)` for structured output: it sets
    CallOptions.response_format and enables `partial_output_stream` and the
    awaitable `output`.
    """
    resolved = _resolve_model(model)
    initial_messages = standardize_prompt(system=system, prompt=prompt, messages=messages)
    stop = stop_when if stop_when is not None else step_count_is(1)
    specs = _tool_specs(tools, active_tools)
    response_format = output.response_format() if output is not None else None

    async def run_step_loop(result: StreamTextResult) -> None:
        working_messages = list(initial_messages)
        await result._emit(StreamStart())

        while True:
            options = CallOptions(
                prompt=list(working_messages),
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop_sequences=stop_sequences,
                seed=seed,
                tools=specs,
                tool_choice=tool_choice,
                response_format=response_format,
                headers=headers,
                provider_options=provider_options or {},
                include_raw_chunks=include_raw_chunks,
            )
            await result._emit(StartStep())

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

            async for part in resolved.do_stream(options):
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
            tool_results = await _execute_tool_calls(tool_calls, tools, working_messages)
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
                    request=step_request,
                )
            )
            if on_step_finish is not None:
                cb = on_step_finish(step)
                if asyncio.iscoroutine(cb):
                    await cb

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
        output=output,
    )


def generate_id() -> str:
    """Generate a unique id (for tool calls etc.)."""
    return uuid.uuid4().hex
