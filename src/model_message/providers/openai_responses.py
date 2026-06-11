"""OpenAI Responses API provider (the default for openai(...) models).

System messages map to `instructions`; tool calls/results map to
function_call / function_call_output input items. providerOptions under the
"openai" key are merged into the request body (e.g.
{"openai": {"reasoning": {"effort": "high", "summary": "auto"}}}).

Reasoning replay
----------------
When the adapter parses a Responses output it captures, per ReasoningPart,
``provider_options={"openai": {"item_id", "encrypted_content"}}``. On request
conversion these are replayed as a ``{"type": "reasoning", ...}`` input item
emitted *before* the rest of that assistant message's items, so multi-turn
reasoning context is preserved. To receive encrypted reasoning content back,
the caller sets ``provider_options={"openai": {"store": False, "include":
["reasoning.encrypted_content"]}}`` — ``store`` and ``include`` flow through
as top-level request fields like any other ``openai`` provider option.

previous_response_id
--------------------
``provider_options={"openai": {"previous_response_id": "resp_..."}}`` is
promoted to a first-class top-level request field (an extra_body path also
works for backward compatibility).

Built-in (provider-executed) tools
----------------------------------
``CallOptions.tools`` are function tools. To request OpenAI's server-side
built-in tools (web search, file search, code interpreter), pass them via
``provider_options={"openai": {"tools": [{"type": "web_search"}, ...]}}``;
those entries are merged into the request ``tools`` array alongside the
converted function tools. Built-in tool *output* items
(``web_search_call`` / ``file_search_call`` / ``code_interpreter_call``)
parse into provider-executed ToolCallPart/ToolResultPart, and url_citation
annotations on output text become UrlSourcePart sources. Provider-executed
parts are server-side items and are *skipped* on history replay (they are not
re-sendable input items).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ..errors import MissingDependencyError
from ..messages import (
    AssistantContentPart,
    AssistantModelMessage,
    FileIdData,
    FilePart,
    ImagePart,
    JsonOutput,
    ModelMessage,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    UrlSourcePart,
    UserModelMessage,
)
from ..provider import CallOptions, LanguageModel, ProviderResult
from ..results import (
    CallWarning,
    FinishReason,
    InputTokenDetails,
    OutputTokenDetails,
    ResponseMetadata,
    Usage,
)
from ..stream import (
    ErrorPart,
    Finish,
    ProviderStreamPart,
    RawPart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    ResponseMetadataPart,
    SourceStreamPart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputStart,
)
from ._util import (
    as_data_url,
    file_id_value,
    raw_event_value,
    request_echo,
    system_and_rest,
    wrap_provider_error,
)

from .openai_chat import _map_usage as _map_chat_usage  # noqa: F401 (unused; kept for symmetry)
from .openai_chat import _tool_result_text

# Built-in (provider-executed) tool output item types → tool names.
_BUILTIN_TOOL_NAMES = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
}


def _user_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "input_text", "text": part.text}
    if isinstance(part, ImagePart):
        if isinstance(part.image, FileIdData):
            item: dict[str, Any] = {
                "type": "input_image",
                "file_id": file_id_value(part.image),
            }
        else:
            item = {
                "type": "input_image",
                "image_url": as_data_url(part.image, part.media_type),
            }
        detail = ((part.provider_options or {}).get("openai") or {}).get("image_detail")
        if detail:
            item["detail"] = detail
        return item
    if isinstance(part, FilePart):
        if isinstance(part.data, FileIdData):
            item = {"type": "input_file", "file_id": file_id_value(part.data)}
            if part.filename:
                item["filename"] = part.filename
            return item
        if part.media_type.startswith("image/"):
            return {
                "type": "input_image",
                "image_url": as_data_url(part.data, part.media_type),
            }
        item = {"type": "input_file", "file_data": as_data_url(part.data, part.media_type)}
        if part.filename:
            item["filename"] = part.filename
        return item
    raise ValueError(f"Unsupported user content part: {part!r}")


def _reasoning_replay_item(part: ReasoningPart) -> Optional[dict[str, Any]]:
    """Build a Responses ``reasoning`` input item from a ReasoningPart that
    carries openai item_id/encrypted_content provider options, else None.

    Wire shape (verified against openai.types.responses
    ResponseReasoningItemParam): {"type": "reasoning", "id": item_id,
    "summary": [{"type": "summary_text", "text": ...}],
    "encrypted_content": ... (only when present)}.
    """
    opts = (part.provider_options or {}).get("openai") or {}
    item_id = opts.get("item_id")
    encrypted = opts.get("encrypted_content")
    if item_id is None and encrypted is None:
        return None
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "summary": (
            [{"type": "summary_text", "text": part.text}] if part.text else []
        ),
    }
    if encrypted is not None:
        item["encrypted_content"] = encrypted
    return item


def convert_to_responses_input(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserModelMessage):
            if isinstance(message.content, str):
                items.append({"role": "user", "content": message.content})
            else:
                items.append(
                    {
                        "role": "user",
                        "content": [_user_content_part(p) for p in message.content],
                    }
                )

        elif isinstance(message, AssistantModelMessage):
            if isinstance(message.content, str):
                items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": message.content}],
                    }
                )
                continue
            # Replay reasoning items (captured on a prior parse) BEFORE the
            # other items of this assistant message.
            for p in message.content:
                if isinstance(p, ReasoningPart):
                    reasoning_item = _reasoning_replay_item(p)
                    if reasoning_item is not None:
                        items.append(reasoning_item)
            text_parts = [
                {"type": "output_text", "text": p.text}
                for p in message.content
                if isinstance(p, TextPart)
            ]
            if text_parts:
                items.append({"role": "assistant", "content": text_parts})
            for p in message.content:
                if isinstance(p, ToolCallPart):
                    # Provider-executed tool calls are server-side items, not
                    # re-sendable function_call input; skip on replay.
                    if p.provider_executed:
                        continue
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": p.tool_call_id,
                            "name": p.tool_name,
                            "arguments": json.dumps(p.input or {}),
                        }
                    )
                # ToolResultPart with provider_executed (built-in tool results)
                # and source parts are server-side; skipped on replay.

        elif isinstance(message, ToolModelMessage):
            for part in message.content:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": part.tool_call_id,
                        "output": _tool_result_text(part),
                    }
                )
    return items


def _map_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    output_details = getattr(usage, "output_tokens_details", None)
    input_details = getattr(usage, "input_tokens_details", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    reasoning_tokens = (
        getattr(output_details, "reasoning_tokens", None) if output_details else None
    )
    cached_tokens = (
        getattr(input_details, "cached_tokens", None) if input_details else None
    )

    input_token_details: Optional[InputTokenDetails] = None
    if cached_tokens is not None:
        no_cache = input_tokens - cached_tokens if input_tokens is not None else None
        input_token_details = InputTokenDetails(
            cache_read_tokens=cached_tokens, no_cache_tokens=no_cache
        )
    output_token_details: Optional[OutputTokenDetails] = None
    if reasoning_tokens is not None:
        text = output_tokens - reasoning_tokens if output_tokens is not None else None
        output_token_details = OutputTokenDetails(
            reasoning_tokens=reasoning_tokens, text_tokens=text
        )

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_tokens,
        input_token_details=input_token_details,
        output_token_details=output_token_details,
    )


def _finish_reason(response: Any, has_tool_calls: bool) -> tuple[FinishReason, Optional[str]]:
    status = getattr(response, "status", None)
    if status == "completed":
        return ("tool-calls" if has_tool_calls else "stop"), status
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        if reason == "max_output_tokens":
            return "length", reason
        if reason == "content_filter":
            return "content-filter", reason
        return "other", reason or status
    if status == "failed":
        return "error", status
    return "unknown", status


def _builtin_tool_input(item: Any) -> Any:
    """Extract a JSON-able input for a built-in tool call output item."""
    action = getattr(item, "action", None)
    if action is not None:
        dump = getattr(action, "model_dump", None)
        if callable(dump):
            try:
                return dump(exclude_none=True)
            except Exception:  # noqa: BLE001
                pass
        return action
    queries = getattr(item, "queries", None)
    if queries is not None:
        return {"queries": list(queries)}
    code = getattr(item, "code", None)
    if code is not None:
        return {"code": code}
    return None


def _builtin_tool_output(item: Any) -> Any:
    """Extract a JSON-able output (results) for a built-in tool call, or None."""
    results = getattr(item, "results", None)
    if results is not None:
        out = []
        for r in results:
            dump = getattr(r, "model_dump", None)
            out.append(dump(exclude_none=True) if callable(dump) else r)
        return out
    outputs = getattr(item, "outputs", None)
    if outputs is not None:
        out = []
        for o in outputs:
            dump = getattr(o, "model_dump", None)
            out.append(dump(exclude_none=True) if callable(dump) else o)
        return out
    return None


def _annotation_source(ann: Any) -> Optional[UrlSourcePart]:
    """Build a UrlSourcePart from a streaming url_citation annotation.

    The streaming event types this field as an untyped object, so it arrives
    as a plain dict; handle both dict and attribute access defensively."""
    if ann is None:
        return None
    if isinstance(ann, dict):
        ann_type = ann.get("type")
        url = ann.get("url")
        title = ann.get("title")
    else:
        ann_type = getattr(ann, "type", None)
        url = getattr(ann, "url", None)
        title = getattr(ann, "title", None)
    if ann_type == "url_citation" and url:
        return UrlSourcePart(id=url, url=url, title=title)
    return None


def _url_citation_sources(item: Any) -> list[UrlSourcePart]:
    """Collect url_citation annotations from a message item's output_text
    parts into UrlSourcePart sources."""
    sources: list[UrlSourcePart] = []
    for part in getattr(item, "content", None) or []:
        if getattr(part, "type", None) != "output_text":
            continue
        for ann in getattr(part, "annotations", None) or []:
            if getattr(ann, "type", None) == "url_citation" and getattr(ann, "url", None):
                sources.append(
                    UrlSourcePart(
                        id=ann.url,
                        url=ann.url,
                        title=getattr(ann, "title", None),
                    )
                )
    return sources


@dataclass
class OpenAIResponsesLanguageModel(LanguageModel):
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: dict[str, str] = field(default_factory=dict)
    provider: str = "openai.responses"
    provider_options_keys: tuple[str, ...] = ("openai",)
    _client_cache: Any = field(default=None, repr=False, compare=False)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import openai
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai") from exc
        kwargs: dict[str, Any] = {"max_retries": 0}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        self._client_cache = openai.AsyncOpenAI(**kwargs)
        return self._client_cache

    def _request(self, options: CallOptions) -> dict[str, Any]:
        system_texts, rest = system_and_rest(options.prompt)
        request: dict[str, Any] = {
            "model": self.model_id,
            "input": convert_to_responses_input(rest),
        }
        if system_texts:
            request["instructions"] = "\n\n".join(system_texts)
        if options.max_output_tokens is not None:
            request["max_output_tokens"] = options.max_output_tokens
        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.top_p is not None:
            request["top_p"] = options.top_p
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description or "",
                "parameters": spec.input_schema,
                **({"strict": spec.strict} if spec.strict is not None else {}),
            }
            for spec in options.tools
        ]
        # Merge any built-in (server-side) tools supplied via provider options.
        for key in self.provider_options_keys:
            builtin = (options.provider_options.get(key) or {}).get("tools")
            if isinstance(builtin, list):
                tools.extend(builtin)
        if tools:
            request["tools"] = tools
        if options.tool_choice is not None:
            choice = options.tool_choice
            if choice in ("auto", "none", "required"):
                request["tool_choice"] = choice
            elif isinstance(choice, dict):
                request["tool_choice"] = {
                    "type": "function",
                    "name": choice.get("tool_name") or choice.get("toolName"),
                }
        if options.response_format and options.response_format.get("type") == "json":
            fmt: dict[str, Any] = {"type": "json_object"}
            if options.response_format.get("schema"):
                fmt = {
                    "type": "json_schema",
                    "name": options.response_format.get("name", "response"),
                    "schema": options.response_format["schema"],
                    "strict": True,
                }
            request["text"] = {"format": fmt}

        # Provider options under "openai" merge into the request. "tools" is
        # handled above; "previous_response_id" (and other known SDK params
        # like store/include) are promoted to top-level request fields, while
        # anything else falls through extra_body.
        _TOP_LEVEL = {"previous_response_id", "store", "include"}
        extra_body: dict[str, Any] = {}
        for key in self.provider_options_keys:
            for name, value in (options.provider_options.get(key) or {}).items():
                if name == "tools":
                    continue
                if name in _TOP_LEVEL:
                    request.setdefault(name, value)
                else:
                    extra_body.setdefault(name, value)
        if extra_body:
            request["extra_body"] = extra_body
        if options.headers:
            request["extra_headers"] = options.headers
        return request

    @staticmethod
    def _warnings(options: CallOptions) -> list[CallWarning]:
        warnings: list[CallWarning] = []
        for name in ("presence_penalty", "frequency_penalty", "seed", "stop_sequences", "top_k"):
            if getattr(options, name) is not None:
                warnings.append(CallWarning(type="unsupported-setting", setting=name))
        return warnings

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        request = self._request(options)
        try:
            response = await client.responses.create(**request)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        content: list[AssistantContentPart] = []
        for item in response.output or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in item.content or []:
                    if getattr(part, "type", None) == "output_text":
                        content.append(TextPart(text=part.text))
                content.extend(_url_citation_sources(item))
            elif item_type == "function_call":
                try:
                    parsed = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                content.append(
                    ToolCallPart(
                        tool_call_id=item.call_id, tool_name=item.name, input=parsed
                    )
                )
            elif item_type in _BUILTIN_TOOL_NAMES:
                tool_name = _BUILTIN_TOOL_NAMES[item_type]
                call_id = getattr(item, "id", None)
                content.append(
                    ToolCallPart(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        input=_builtin_tool_input(item),
                        provider_executed=True,
                    )
                )
                output = _builtin_tool_output(item)
                if output is not None:
                    content.append(
                        ToolResultPart(
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            output=JsonOutput(value=output),
                            provider_executed=True,
                        )
                    )
            elif item_type == "reasoning":
                summary = "".join(
                    getattr(s, "text", "") for s in (item.summary or [])
                )
                content.append(
                    ReasoningPart(
                        text=summary,
                        provider_options={
                            "openai": {
                                "item_id": getattr(item, "id", None),
                                "encrypted_content": getattr(item, "encrypted_content", None),
                            }
                        },
                    )
                )

        # Only function tool calls drive the agent loop; provider-executed
        # built-in calls are server-side and do not stop for local execution.
        has_tool_calls = any(
            isinstance(p, ToolCallPart) and not p.provider_executed for p in content
        )
        finish, raw_finish = _finish_reason(response, has_tool_calls)
        return ProviderResult(
            content=content,
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            usage=_map_usage(response.usage),
            response=ResponseMetadata(id=response.id, model_id=response.model),
            warnings=self._warnings(options),
            request=request_echo(request),
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        request = self._request(options)
        try:
            stream = await client.responses.create(**request, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        echoed_request = request_echo(request)
        usage = Usage()
        finish: FinishReason = "unknown"
        raw_finish: Optional[str] = None
        has_tool_calls = False
        open_text: set[str] = set()
        open_reasoning: set[str] = set()
        # item_id -> {"call_id", "name", "arguments"}
        calls: dict[str, dict[str, Any]] = {}

        try:
            async for event in stream:
                etype = getattr(event, "type", "")
                if options.include_raw_chunks:
                    yield RawPart(raw_value=raw_event_value(event))
                if etype == "response.created":
                    yield ResponseMetadataPart(
                        id=event.response.id,
                        model_id=event.response.model,
                        request=echoed_request,
                    )
                elif etype == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        has_tool_calls = True
                        calls[item.id] = {
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": "",
                        }
                        yield ToolInputStart(id=item.call_id, tool_name=item.name)
                    elif getattr(item, "type", None) in _BUILTIN_TOOL_NAMES:
                        # Built-in tool call (web_search/file_search/code_int.):
                        # announce the provider-executed call; the complete
                        # ToolCallPart is emitted on output_item.done.
                        tool_name = _BUILTIN_TOOL_NAMES[item.type]
                        yield ToolInputStart(
                            id=item.id,
                            tool_name=tool_name,
                            provider_executed=True,
                        )
                elif etype == "response.output_item.done":
                    item = event.item
                    if getattr(item, "type", None) in _BUILTIN_TOOL_NAMES:
                        tool_name = _BUILTIN_TOOL_NAMES[item.type]
                        yield ToolInputEnd(id=item.id)
                        yield ToolCallPart(
                            tool_call_id=item.id,
                            tool_name=tool_name,
                            input=_builtin_tool_input(item),
                            provider_executed=True,
                        )
                elif etype == "response.output_text.annotation.added":
                    ann = getattr(event, "annotation", None)
                    src = _annotation_source(ann)
                    if src is not None:
                        yield SourceStreamPart(source=src)
                elif etype == "response.output_text.delta":
                    if event.item_id not in open_text:
                        open_text.add(event.item_id)
                        yield TextStart(id=event.item_id)
                    yield TextDelta(id=event.item_id, text=event.delta)
                elif etype == "response.output_text.done":
                    if event.item_id in open_text:
                        open_text.discard(event.item_id)
                        yield TextEnd(id=event.item_id)
                elif etype in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    if event.item_id not in open_reasoning:
                        open_reasoning.add(event.item_id)
                        yield ReasoningStart(id=event.item_id)
                    yield ReasoningDelta(id=event.item_id, text=event.delta)
                elif etype in (
                    "response.reasoning_summary_text.done",
                    "response.reasoning_text.done",
                ):
                    if event.item_id in open_reasoning:
                        open_reasoning.discard(event.item_id)
                        yield ReasoningEnd(id=event.item_id)
                elif etype == "response.function_call_arguments.delta":
                    state = calls.get(event.item_id)
                    if state is not None:
                        state["arguments"] += event.delta
                        yield ToolInputDelta(id=state["call_id"], delta=event.delta)
                elif etype == "response.function_call_arguments.done":
                    state = calls.get(event.item_id)
                    if state is not None:
                        arguments = getattr(event, "arguments", None) or state["arguments"]
                        yield ToolInputEnd(id=state["call_id"])
                        try:
                            parsed = json.loads(arguments or "{}")
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallPart(
                            tool_call_id=state["call_id"],
                            tool_name=state["name"],
                            input=parsed,
                        )
                elif etype in ("response.completed", "response.incomplete", "response.failed"):
                    usage = _map_usage(getattr(event.response, "usage", None))
                    finish, raw_finish = _finish_reason(event.response, has_tool_calls)
                elif etype == "error":
                    yield ErrorPart(error=getattr(event, "message", "stream error"))
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        yield Finish(finish_reason=finish, raw_finish_reason=raw_finish, total_usage=usage)
