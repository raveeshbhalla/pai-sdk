"""Anthropic provider — maps ModelMessage onto the Messages API.

Notes:
- Thinking blocks are surfaced as ReasoningPart; the signature is preserved in
  provider_options["anthropic"]["signature"] and replayed on the next turn.
- providerOptions under the "anthropic" key are merged into the request body
  (e.g. {"anthropic": {"thinking": {"type": "adaptive"}}}).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ..errors import MissingDependencyError
from ..messages import (
    AssistantContentPart,
    AssistantModelMessage,
    ContentOutput,
    ErrorJsonOutput,
    ErrorTextOutput,
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
    UserModelMessage,
)
from ..provider import CallOptions, LanguageModel, ProviderResult
from ..results import FinishReason, ResponseMetadata, Usage
from ..stream import (
    Finish,
    ProviderStreamPart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    ResponseMetadataPart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputStart,
)
from ._util import (
    merge_provider_options,
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


def _image_block(part: ImagePart) -> dict[str, Any]:
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
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": part.input or {},
                        }
                    )
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
    total = None
    if input_tokens is not None or output_tokens is not None:
        total = (input_tokens or 0) + (output_tokens or 0) + (cached or 0)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cached_input_tokens=cached,
    )


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
        return parts

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        try:
            response = await client.messages.create(**self._request(options))
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        raw_finish = response.stop_reason
        return ProviderResult(
            content=self._content_parts(response.content),
            finish_reason=_FINISH_REASONS.get(raw_finish or "", "unknown"),
            raw_finish_reason=raw_finish,
            usage=_map_usage(response.usage),
            response=ResponseMetadata(id=response.id, model_id=response.model),
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        try:
            stream = await client.messages.create(**self._request(options), stream=True)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        finish: FinishReason = "unknown"
        raw_finish: Optional[str] = None

        try:
            async for event in stream:
                etype = event.type
                if etype == "message_start":
                    usage = _map_usage(event.message.usage)
                    yield ResponseMetadataPart(
                        id=event.message.id, model_id=event.message.model
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
                    elif state["kind"] == "tool":
                        yield ToolInputEnd(id=state["id"])
                        try:
                            parsed = json.loads(state["json"]) if state["json"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallPart(
                            tool_call_id=state["id"],
                            tool_name=state["name"],
                            input=parsed,
                        )
                elif etype == "message_delta":
                    raw_finish = event.delta.stop_reason
                    finish = _FINISH_REASONS.get(raw_finish or "", "unknown")
                    delta_usage = getattr(event, "usage", None)
                    if delta_usage is not None:
                        usage.output_tokens = getattr(delta_usage, "output_tokens", None)
                        if usage.input_tokens is not None:
                            usage.total_tokens = (usage.input_tokens or 0) + (
                                usage.output_tokens or 0
                            ) + (usage.cached_input_tokens or 0)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Anthropic") from exc

        yield Finish(finish_reason=finish, raw_finish_reason=raw_finish, total_usage=usage)
