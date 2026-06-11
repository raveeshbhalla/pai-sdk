"""Anthropic provider — maps ModelMessage onto the Messages API.

Notes:
- Thinking blocks are surfaced as ReasoningPart; the signature is preserved in
  provider_options["anthropic"]["signature"] and replayed on the next turn.
- providerOptions under the "anthropic" key are merged into the request body
  (e.g. {"anthropic": {"thinking": {"type": "adaptive"}}}).
- Server-side (provider-executed) tools — web_search / web_fetch — surface as
  provider-executed ToolCallPart/ToolResultPart in assistant content, plus
  UrlSourcePart sources. The raw block is stashed in
  provider_options["anthropic"]["raw_block"] so the exact wire shape can be
  echoed back on a follow-up turn (required for pause_turn continuations).
- Text-block `citations` (url / web-search-result locations) surface as
  UrlSourcePart sources appended to the content.
- FilePart/ImagePart carrying FileIdData map to Files-API source blocks
  ({"type": "file", "file_id": ...}). The Files API requires the
  `anthropic-beta: files-api-2025-04-14` header — callers pass it via
  headers= (e.g. generate_text(..., headers={"anthropic-beta": "files-api-..."})).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ..errors import MissingDependencyError
from ..messages import (
    AssistantContentPart,
    AssistantModelMessage,
    ContentOutput,
    ErrorJsonOutput,
    ErrorTextOutput,
    FileIdData,
    FilePart,
    ImagePart,
    JsonOutput,
    ModelMessage,
    ReasoningPart,
    TextOutput,
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
    ResponseMetadata,
    Usage,
)
from ..stream import (
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
    ToolResultEvent,
)
from ._util import (
    file_id_value,
    merge_provider_options,
    raw_event_value,
    request_echo,
    split_data_content,
    system_and_rest,
    to_bytes,
    wrap_provider_error,
)

_DEFAULT_MAX_TOKENS = 4096

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool-calls",
    "refusal": "content-filter",
    "pause_turn": "other",
}


def _raw_block(part: Any) -> Optional[dict[str, Any]]:
    """Return the stashed raw Anthropic block dict for a provider-executed part,
    if one was captured at parse time."""
    opts = getattr(part, "provider_options", None) or {}
    raw = (opts.get("anthropic") or {}).get("raw_block")
    return raw if isinstance(raw, dict) else None


def _image_block(part: ImagePart) -> dict[str, Any]:
    if isinstance(part.image, FileIdData):
        return {
            "type": "image",
            "source": {"type": "file", "file_id": file_id_value(part.image)},
        }
    kind, payload, media = split_data_content(part.image, part.media_type)
    if kind == "url":
        return {"type": "image", "source": {"type": "url", "url": payload}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media or "image/png",
            "data": payload,
        },
    }


def _file_block(part: FilePart) -> dict[str, Any]:
    if isinstance(part.data, FileIdData):
        source_type = "image" if part.media_type.startswith("image/") else "document"
        block: dict[str, Any] = {
            "type": source_type,
            "source": {"type": "file", "file_id": file_id_value(part.data)},
        }
        if source_type == "document" and part.filename:
            block["title"] = part.filename
        return block
    if part.media_type.startswith("image/"):
        return _image_block(ImagePart(image=part.data, media_type=part.media_type))
    if part.media_type == "application/pdf":
        kind, payload, _ = split_data_content(part.data, part.media_type)
        source = (
            {"type": "url", "url": payload}
            if kind == "url"
            else {"type": "base64", "media_type": "application/pdf", "data": payload}
        )
        block: dict[str, Any] = {"type": "document", "source": source}
        if part.filename:
            block["title"] = part.filename
        return block
    if part.media_type.startswith("text/"):
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": to_bytes(part.data).decode("utf-8", errors="replace"),
            },
        }
    raise ValueError(
        f"Anthropic does not support media type '{part.media_type}' as input."
    )


def _tool_result_content(part: ToolResultPart) -> tuple[Any, bool]:
    output = part.output
    if isinstance(output, TextOutput):
        return output.value, False
    if isinstance(output, JsonOutput):
        return json.dumps(output.value), False
    if isinstance(output, ErrorTextOutput):
        return output.value, True
    if isinstance(output, ErrorJsonOutput):
        return json.dumps(output.value), True
    if isinstance(output, ContentOutput):
        blocks: list[dict[str, Any]] = []
        for item in output.value:
            if item.type == "text":
                blocks.append({"type": "text", "text": item.text})
            else:  # media
                kind, payload, media = split_data_content(item.data, item.media_type)
                source = (
                    {"type": "url", "url": payload}
                    if kind == "url"
                    else {"type": "base64", "media_type": media or "image/png", "data": payload}
                )
                blocks.append({"type": "image", "source": source})
        return blocks, False
    return str(output), False


def _convert_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserModelMessage):
            if isinstance(message.content, str):
                converted.append({"role": "user", "content": message.content})
                continue
            blocks: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    blocks.append(_image_block(part))
                elif isinstance(part, FilePart):
                    blocks.append(_file_block(part))
            converted.append({"role": "user", "content": blocks})

        elif isinstance(message, AssistantModelMessage):
            if isinstance(message.content, str):
                converted.append({"role": "assistant", "content": message.content})
                continue
            blocks = []
            for part in message.content:
                if isinstance(part, TextPart):
                    blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ReasoningPart):
                    anthropic_opts = (part.provider_options or {}).get("anthropic", {})
                    signature = anthropic_opts.get("signature")
                    redacted = anthropic_opts.get("redacted_data")
                    if redacted is not None:
                        blocks.append({"type": "redacted_thinking", "data": redacted})
                    elif signature is not None:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": part.text,
                                "signature": signature,
                            }
                        )
                    # Unsigned reasoning can't be replayed — drop it.
                elif isinstance(part, ToolCallPart):
                    raw = _raw_block(part)
                    if part.provider_executed and raw is not None:
                        # Echo the original server_tool_use block verbatim so
                        # pause_turn continuations resolve correctly.
                        blocks.append(raw)
                    elif part.provider_executed:
                        # Best-effort reconstruction of a server_tool_use block.
                        blocks.append(
                            {
                                "type": "server_tool_use",
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "input": part.input or {},
                            }
                        )
                    else:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "input": part.input or {},
                            }
                        )
                elif isinstance(part, ToolResultPart) and part.provider_executed:
                    # Provider-executed tool results live in assistant content
                    # (e.g. web_search_tool_result). Echo the original block.
                    raw = _raw_block(part)
                    if raw is not None:
                        blocks.append(raw)
                elif isinstance(part, UrlSourcePart):
                    # Sources are model-generated citations, not request input.
                    continue
            if blocks:
                converted.append({"role": "assistant", "content": blocks})

        elif isinstance(message, ToolModelMessage):
            blocks = []
            for part in message.content:
                content, is_error = _tool_result_content(part)
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id,
                    "content": content,
                }
                if is_error:
                    block["is_error"] = True
                blocks.append(block)
            converted.append({"role": "user", "content": blocks})
    return converted


def _map_usage(usage: Any) -> Usage:
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cached = getattr(usage, "cache_read_input_tokens", None)
    cache_write = getattr(usage, "cache_creation_input_tokens", None)
    total = None
    if input_tokens is not None or output_tokens is not None:
        total = (input_tokens or 0) + (output_tokens or 0) + (cached or 0)
    details: Optional[InputTokenDetails] = None
    if input_tokens is not None or cached is not None or cache_write is not None:
        details = InputTokenDetails(
            no_cache_tokens=input_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=cache_write,
        )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cached_input_tokens=cached,
        input_token_details=details,
    )


def _provider_metadata(response: Any) -> dict[str, dict[str, Any]]:
    """Build provider_metadata["anthropic"] from a non-streaming Message."""
    usage = getattr(response, "usage", None)
    meta: dict[str, Any] = {
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "stop_sequence": getattr(response, "stop_sequence", None),
    }
    container = getattr(response, "container", None)
    if container is not None:
        meta["container_id"] = getattr(container, "id", None)
    return {"anthropic": meta}


def _block_dump(block: Any) -> dict[str, Any]:
    """JSON-able dict for an SDK response block (for raw_block stashing)."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return dict(block) if isinstance(block, dict) else {}


def _new_source_id() -> str:
    return str(uuid.uuid4())


def _web_search_sources(content: Any) -> list[UrlSourcePart]:
    """UrlSourceParts for each web_search_result entry in a tool-result block."""
    sources: list[UrlSourcePart] = []
    if not isinstance(content, list):
        return sources
    for entry in content:
        if getattr(entry, "type", None) == "web_search_result":
            url = getattr(entry, "url", None)
            if url:
                sources.append(
                    UrlSourcePart(
                        id=_new_source_id(),
                        url=url,
                        title=getattr(entry, "title", None),
                    )
                )
    return sources


def _serialize_tool_result_content(content: Any) -> Any:
    """JSON-able value for a provider-executed tool-result block's content."""
    if isinstance(content, list):
        return [_block_dump(item) for item in content]
    dump = getattr(content, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return content


def _citation_sources(block: Any) -> list[UrlSourcePart]:
    """UrlSourceParts for url/web-search citations on a text block."""
    sources: list[UrlSourcePart] = []
    citations = getattr(block, "citations", None) or []
    for cite in citations:
        ctype = getattr(cite, "type", None)
        url = getattr(cite, "url", None)
        if ctype in ("web_search_result_location", "url") and url:
            sources.append(
                UrlSourcePart(
                    id=_new_source_id(),
                    url=url,
                    title=getattr(cite, "title", None),
                )
            )
    return sources


def _unsupported_setting_warnings(options: CallOptions) -> list[CallWarning]:
    """Anthropic's Messages API has no seed/presence_penalty/frequency_penalty;
    warn when set rather than silently dropping them."""
    warnings: list[CallWarning] = []
    for setting, value in (
        ("seed", options.seed),
        ("presence_penalty", options.presence_penalty),
        ("frequency_penalty", options.frequency_penalty),
    ):
        if value is not None:
            warnings.append(
                CallWarning(
                    type="unsupported-setting",
                    setting=setting,
                    message=f"Anthropic does not support the '{setting}' setting.",
                )
            )
    return warnings


@dataclass
class AnthropicLanguageModel(LanguageModel):
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: dict[str, str] = field(default_factory=dict)
    provider: str = "anthropic"
    _client_cache: Any = field(default=None, repr=False, compare=False)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "anthropic") from exc
        kwargs: dict[str, Any] = {"max_retries": 0}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client_cache = anthropic.AsyncAnthropic(**kwargs)
        return self._client_cache

    def _request(self, options: CallOptions) -> dict[str, Any]:
        system_texts, rest = system_and_rest(options.prompt)
        request: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": options.max_output_tokens or _DEFAULT_MAX_TOKENS,
            "messages": _convert_messages(rest),
        }
        if system_texts:
            request["system"] = "\n\n".join(system_texts)
        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.top_p is not None:
            request["top_p"] = options.top_p
        if options.top_k is not None:
            request["top_k"] = options.top_k
        if options.stop_sequences:
            request["stop_sequences"] = options.stop_sequences
        if options.tools:
            request["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description or "",
                    "input_schema": spec.input_schema,
                    **({"strict": spec.strict} if spec.strict is not None else {}),
                }
                for spec in options.tools
            ]
        if options.tool_choice is not None:
            choice = options.tool_choice
            if choice == "auto":
                request["tool_choice"] = {"type": "auto"}
            elif choice == "required":
                request["tool_choice"] = {"type": "any"}
            elif choice == "none":
                request["tool_choice"] = {"type": "none"}
            elif isinstance(choice, dict):
                request["tool_choice"] = {
                    "type": "tool",
                    "name": choice.get("tool_name") or choice.get("toolName"),
                }
        if options.response_format and options.response_format.get("type") == "json":
            output_format: dict[str, Any] = {"type": "json_schema"}
            if options.response_format.get("schema"):
                output_format["schema"] = options.response_format["schema"]
            request["output_config"] = {"format": output_format}
        # Provider options ride in extra_body so they merge into the JSON
        # body even when the SDK's typed create() doesn't know the param.
        extra_body: dict[str, Any] = {}
        merge_provider_options(extra_body, options.provider_options, "anthropic")
        if extra_body:
            request["extra_body"] = extra_body
        if options.headers or self.default_headers:
            request["extra_headers"] = {**self.default_headers, **(options.headers or {})}
        return request

    @staticmethod
    def _content_parts(blocks: Any) -> list[AssistantContentPart]:
        parts: list[AssistantContentPart] = []
        for block in blocks:
            if block.type == "text":
                parts.append(TextPart(text=block.text))
            elif block.type == "thinking":
                parts.append(
                    ReasoningPart(
                        text=block.thinking,
                        provider_options={"anthropic": {"signature": block.signature}},
                    )
                )
            elif block.type == "redacted_thinking":
                parts.append(
                    ReasoningPart(
                        text="",
                        provider_options={"anthropic": {"redacted_data": block.data}},
                    )
                )
            elif block.type == "tool_use":
                parts.append(
                    ToolCallPart(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        input=block.input,
                    )
                )
            elif block.type == "server_tool_use":
                parts.append(
                    ToolCallPart(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        input=block.input,
                        provider_executed=True,
                        provider_options={"anthropic": {"raw_block": _block_dump(block)}},
                    )
                )
            elif block.type in ("web_search_tool_result", "web_fetch_tool_result"):
                parts.append(
                    ToolResultPart(
                        tool_call_id=block.tool_use_id,
                        tool_name="web_search"
                        if block.type == "web_search_tool_result"
                        else "web_fetch",
                        output=JsonOutput(
                            value=_serialize_tool_result_content(block.content)
                        ),
                        provider_executed=True,
                        provider_options={"anthropic": {"raw_block": _block_dump(block)}},
                    )
                )
                if block.type == "web_search_tool_result":
                    parts.extend(_web_search_sources(block.content))

        # Append url citation sources after the content parts they cite.
        for block in blocks:
            if getattr(block, "type", None) == "text":
                parts.extend(_citation_sources(block))
        return parts

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        request = self._request(options)
        try:
            response = await client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        raw_finish = response.stop_reason
        return ProviderResult(
            content=self._content_parts(response.content),
            finish_reason=_FINISH_REASONS.get(raw_finish or "", "unknown"),
            raw_finish_reason=raw_finish,
            usage=_map_usage(response.usage),
            response=ResponseMetadata(id=response.id, model_id=response.model),
            warnings=_unsupported_setting_warnings(options),
            provider_metadata=_provider_metadata(response),
            request=request_echo(request),
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        request = self._request(options)
        request_body = request_echo(request)
        try:
            stream = await client.messages.create(**request, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        finish: FinishReason = "unknown"
        raw_finish: Optional[str] = None
        emit_raw = options.include_raw_chunks
        stream_metadata: Optional[dict[str, Any]] = None

        try:
            async for event in stream:
                if emit_raw:
                    yield RawPart(raw_value=raw_event_value(event))
                etype = event.type
                if etype == "message_start":
                    usage = _map_usage(event.message.usage)
                    message_usage = event.message.usage
                    stream_metadata = {
                        "anthropic": {
                            "cache_creation_input_tokens": getattr(
                                message_usage, "cache_creation_input_tokens", None
                            ),
                            "cache_read_input_tokens": getattr(
                                message_usage, "cache_read_input_tokens", None
                            ),
                        }
                    }
                    yield ResponseMetadataPart(
                        id=event.message.id,
                        model_id=event.message.model,
                        request=request_body,
                    )
                elif etype == "content_block_start":
                    block = event.content_block
                    index = event.index
                    if block.type == "text":
                        blocks[index] = {"kind": "text"}
                        yield TextStart(id=str(index))
                    elif block.type == "thinking":
                        blocks[index] = {"kind": "reasoning", "signature": None}
                        yield ReasoningStart(id=str(index))
                    elif block.type == "redacted_thinking":
                        blocks[index] = {"kind": "redacted", "data": block.data}
                    elif block.type == "tool_use":
                        blocks[index] = {
                            "kind": "tool",
                            "id": block.id,
                            "name": block.name,
                            "json": "",
                        }
                        yield ToolInputStart(id=block.id, tool_name=block.name)
                    elif block.type == "server_tool_use":
                        blocks[index] = {
                            "kind": "server_tool",
                            "id": block.id,
                            "name": block.name,
                            "json": "",
                            "raw": _block_dump(block),
                        }
                        yield ToolInputStart(
                            id=block.id,
                            tool_name=block.name,
                            provider_executed=True,
                        )
                    elif block.type in (
                        "web_search_tool_result",
                        "web_fetch_tool_result",
                    ):
                        tool_name = (
                            "web_search"
                            if block.type == "web_search_tool_result"
                            else "web_fetch"
                        )
                        yield ToolResultEvent(
                            tool_call_id=block.tool_use_id,
                            tool_name=tool_name,
                            input=None,
                            output=_serialize_tool_result_content(block.content),
                            model_output=JsonOutput(
                                value=_serialize_tool_result_content(block.content)
                            ),
                            provider_executed=True,
                        )
                        if block.type == "web_search_tool_result":
                            for source in _web_search_sources(block.content):
                                yield SourceStreamPart(source=source)
                elif etype == "content_block_delta":
                    state = blocks.get(event.index)
                    if state is None:
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextDelta(id=str(event.index), text=delta.text)
                    elif delta.type == "thinking_delta":
                        yield ReasoningDelta(id=str(event.index), text=delta.thinking)
                    elif delta.type == "signature_delta":
                        state["signature"] = (state.get("signature") or "") + delta.signature
                    elif delta.type == "input_json_delta":
                        state["json"] += delta.partial_json
                        yield ToolInputDelta(id=state["id"], delta=delta.partial_json)
                    elif delta.type == "citations_delta":
                        cite = getattr(delta, "citation", None)
                        url = getattr(cite, "url", None)
                        ctype = getattr(cite, "type", None)
                        if url and ctype in ("web_search_result_location", "url"):
                            yield SourceStreamPart(
                                source=UrlSourcePart(
                                    id=_new_source_id(),
                                    url=url,
                                    title=getattr(cite, "title", None),
                                )
                            )
                elif etype == "content_block_stop":
                    state = blocks.get(event.index)
                    if state is None:
                        continue
                    if state["kind"] == "text":
                        yield TextEnd(id=str(event.index))
                    elif state["kind"] == "reasoning":
                        yield ReasoningEnd(
                            id=str(event.index),
                            provider_metadata={
                                "anthropic": {"signature": state.get("signature")}
                            },
                        )
                    elif state["kind"] in ("tool", "server_tool"):
                        yield ToolInputEnd(id=state["id"])
                        try:
                            parsed = json.loads(state["json"]) if state["json"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        provider_executed = state["kind"] == "server_tool"
                        kwargs: dict[str, Any] = {
                            "tool_call_id": state["id"],
                            "tool_name": state["name"],
                            "input": parsed,
                        }
                        if provider_executed:
                            kwargs["provider_executed"] = True
                            kwargs["provider_options"] = {
                                "anthropic": {"raw_block": state.get("raw")}
                            }
                        yield ToolCallPart(**kwargs)
                elif etype == "message_delta":
                    raw_finish = event.delta.stop_reason
                    finish = _FINISH_REASONS.get(raw_finish or "", "unknown")
                    stop_sequence = getattr(event.delta, "stop_sequence", None)
                    if stream_metadata is not None and stop_sequence is not None:
                        stream_metadata["anthropic"]["stop_sequence"] = stop_sequence
                    delta_usage = getattr(event, "usage", None)
                    if delta_usage is not None:
                        usage.output_tokens = getattr(delta_usage, "output_tokens", None)
                        if usage.input_tokens is not None:
                            usage.total_tokens = (usage.input_tokens or 0) + (
                                usage.output_tokens or 0
                            ) + (usage.cached_input_tokens or 0)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        yield Finish(
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            total_usage=usage,
            provider_metadata=stream_metadata,
        )
