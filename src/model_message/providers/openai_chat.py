"""OpenAI Chat Completions provider (also the base for OpenRouter).

providerOptions under the "openai" key are merged into the request body via
extra_body (e.g. {"openai": {"reasoning_effort": "high"}}).
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
    SystemModelMessage,
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
    ResponseMetadataPart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputStart,
)
from ._util import as_data_url, split_data_content, wrap_provider_error

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool-calls",
    "function_call": "tool-calls",
    "content_filter": "content-filter",
    "error": "error",
}

_AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}


def _user_part(part: Any) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        image_url: dict[str, Any] = {"url": as_data_url(part.image, part.media_type)}
        detail = ((part.provider_options or {}).get("openai") or {}).get("image_detail")
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}
    if isinstance(part, FilePart):
        if part.media_type.startswith("image/"):
            return {
                "type": "image_url",
                "image_url": {"url": as_data_url(part.data, part.media_type)},
            }
        if part.media_type in _AUDIO_FORMATS:
            kind, payload, _ = split_data_content(part.data, part.media_type)
            if kind == "url":
                raise ValueError("OpenAI chat audio input requires base64 data, not URLs.")
            return {
                "type": "input_audio",
                "input_audio": {"data": payload, "format": _AUDIO_FORMATS[part.media_type]},
            }
        file_obj: dict[str, Any] = {"file_data": as_data_url(part.data, part.media_type)}
        if part.filename:
            file_obj["filename"] = part.filename
        return {"type": "file", "file": file_obj}
    raise ValueError(f"Unsupported user content part: {part!r}")


def _tool_result_text(part: ToolResultPart) -> str:
    output = part.output
    if isinstance(output, TextOutput) or isinstance(output, ErrorTextOutput):
        return output.value
    if isinstance(output, (JsonOutput, ErrorJsonOutput)):
        return json.dumps(output.value)
    if isinstance(output, ContentOutput):
        texts = [item.text for item in output.value if item.type == "text"]
        return "\n".join(texts) if texts else "(non-text tool output)"
    return str(output)


def convert_to_chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemModelMessage):
            converted.append({"role": "system", "content": message.content})

        elif isinstance(message, UserModelMessage):
            if isinstance(message.content, str):
                converted.append({"role": "user", "content": message.content})
            else:
                converted.append(
                    {
                        "role": "user",
                        "content": [_user_part(p) for p in message.content],
                    }
                )

        elif isinstance(message, AssistantModelMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if isinstance(message.content, str):
                entry["content"] = message.content
            else:
                texts = [p.text for p in message.content if isinstance(p, TextPart)]
                entry["content"] = "".join(texts) or None
                tool_calls = [
                    {
                        "id": p.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": p.tool_name,
                            "arguments": json.dumps(p.input or {}),
                        },
                    }
                    for p in message.content
                    if isinstance(p, ToolCallPart)
                ]
                if tool_calls:
                    entry["tool_calls"] = tool_calls
            converted.append(entry)

        elif isinstance(message, ToolModelMessage):
            for part in message.content:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": _tool_result_text(part),
                    }
                )
    return converted


def _map_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", None)
        if completion_details
        else None,
        cached_input_tokens=getattr(prompt_details, "cached_tokens", None)
        if prompt_details
        else None,
    )


@dataclass
class OpenAIChatLanguageModel(LanguageModel):
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: dict[str, str] = field(default_factory=dict)
    provider: str = "openai.chat"
    # providerOptions keys merged into the request body, in order.
    provider_options_keys: tuple[str, ...] = ("openai",)
    api_key_env: str = "OPENAI_API_KEY"
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

    def _request(self, options: CallOptions, stream: bool) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": convert_to_chat_messages(options.prompt),
        }
        if options.max_output_tokens is not None:
            request["max_completion_tokens"] = options.max_output_tokens
        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.top_p is not None:
            request["top_p"] = options.top_p
        if options.presence_penalty is not None:
            request["presence_penalty"] = options.presence_penalty
        if options.frequency_penalty is not None:
            request["frequency_penalty"] = options.frequency_penalty
        if options.stop_sequences:
            request["stop"] = options.stop_sequences
        if options.seed is not None:
            request["seed"] = options.seed
        if options.tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description or "",
                        "parameters": spec.input_schema,
                        **({"strict": spec.strict} if spec.strict is not None else {}),
                    },
                }
                for spec in options.tools
            ]
        if options.tool_choice is not None:
            choice = options.tool_choice
            if choice in ("auto", "none", "required"):
                request["tool_choice"] = choice
            elif isinstance(choice, dict):
                request["tool_choice"] = {
                    "type": "function",
                    "function": {
                        "name": choice.get("tool_name") or choice.get("toolName")
                    },
                }
        if options.response_format:
            fmt = options.response_format
            if fmt.get("type") == "json" and fmt.get("schema"):
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.get("name", "response"),
                        "schema": fmt["schema"],
                        "strict": True,
                    },
                }
            elif fmt.get("type") == "json":
                request["response_format"] = {"type": "json_object"}
        if stream:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}

        extra_body: dict[str, Any] = {}
        for key in self.provider_options_keys:
            for name, value in (options.provider_options.get(key) or {}).items():
                extra_body.setdefault(name, value)
        if extra_body:
            request["extra_body"] = extra_body
        if options.headers:
            request["extra_headers"] = options.headers
        return request

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        try:
            response = await client.chat.completions.create(**self._request(options, stream=False))
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        choice = response.choices[0]
        message = choice.message
        content: list[AssistantContentPart] = []
        reasoning_text = getattr(message, "reasoning", None)  # OpenRouter extension
        if reasoning_text:
            from ..messages import ReasoningPart

            content.append(ReasoningPart(text=reasoning_text))
        if message.content:
            content.append(TextPart(text=message.content))
        for call in message.tool_calls or []:
            try:
                parsed = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                parsed = {}
            content.append(
                ToolCallPart(
                    tool_call_id=call.id, tool_name=call.function.name, input=parsed
                )
            )

        raw_finish = choice.finish_reason
        return ProviderResult(
            content=content,
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
            stream = await client.chat.completions.create(**self._request(options, stream=True))
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        usage = Usage()
        raw_finish: Optional[str] = None
        text_open = False
        sent_metadata = False
        # tool call accumulation by index: {index: {id, name, arguments}}
        tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in stream:
                if not sent_metadata and getattr(chunk, "id", None):
                    sent_metadata = True
                    yield ResponseMetadataPart(id=chunk.id, model_id=chunk.model)
                if getattr(chunk, "usage", None):
                    usage = _map_usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None and delta.content:
                    if not text_open:
                        text_open = True
                        yield TextStart(id="0")
                    yield TextDelta(id="0", text=delta.content)
                for tc in (delta.tool_calls if delta is not None else None) or []:
                    state = tool_calls.get(tc.index)
                    if state is None:
                        state = {"id": tc.id, "name": "", "arguments": ""}
                        tool_calls[tc.index] = state
                        if tc.function and tc.function.name:
                            state["name"] = tc.function.name
                        yield ToolInputStart(
                            id=state["id"] or f"call_{tc.index}",
                            tool_name=state["name"],
                        )
                    if tc.id and not state["id"]:
                        state["id"] = tc.id
                    if tc.function:
                        if tc.function.name and not state["name"]:
                            state["name"] = tc.function.name
                        if tc.function.arguments:
                            state["arguments"] += tc.function.arguments
                            yield ToolInputDelta(
                                id=state["id"] or f"call_{tc.index}",
                                delta=tc.function.arguments,
                            )
                if choice.finish_reason:
                    raw_finish = choice.finish_reason
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        if text_open:
            yield TextEnd(id="0")
        for index in sorted(tool_calls):
            state = tool_calls[index]
            call_id = state["id"] or f"call_{index}"
            yield ToolInputEnd(id=call_id)
            try:
                parsed = json.loads(state["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed = {}
            yield ToolCallPart(tool_call_id=call_id, tool_name=state["name"], input=parsed)

        yield Finish(
            finish_reason=_FINISH_REASONS.get(raw_finish or "", "unknown"),
            raw_finish_reason=raw_finish,
            total_usage=usage,
        )
