"""OpenAI Responses API provider (the default for openai(...) models).

System messages map to `instructions`; tool calls/results map to
function_call / function_call_output input items. providerOptions under the
"openai" key are merged into the request body (e.g.
{"openai": {"reasoning": {"effort": "high", "summary": "auto"}}}).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ..errors import MissingDependencyError
from ..messages import (
    AssistantContentPart,
    AssistantModelMessage,
    FilePart,
    ImagePart,
    ModelMessage,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    UserModelMessage,
)
from ..provider import CallOptions, LanguageModel, ProviderResult
from ..results import FinishReason, ResponseMetadata, Usage
from ..stream import (
    ErrorPart,
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
from ._util import as_data_url, system_and_rest, wrap_provider_error

from .openai_chat import _map_usage as _map_chat_usage  # noqa: F401 (unused; kept for symmetry)
from .openai_chat import _tool_result_text


def _user_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "input_text", "text": part.text}
    if isinstance(part, ImagePart):
        item: dict[str, Any] = {
            "type": "input_image",
            "image_url": as_data_url(part.image, part.media_type),
        }
        detail = ((part.provider_options or {}).get("openai") or {}).get("image_detail")
        if detail:
            item["detail"] = detail
        return item
    if isinstance(part, FilePart):
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
            text_parts = [
                {"type": "output_text", "text": p.text}
                for p in message.content
                if isinstance(p, TextPart)
            ]
            if text_parts:
                items.append({"role": "assistant", "content": text_parts})
            for p in message.content:
                if isinstance(p, ToolCallPart):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": p.tool_call_id,
                            "name": p.tool_name,
                            "arguments": json.dumps(p.input or {}),
                        }
                    )

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
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None)
        if output_details
        else None,
        cached_input_tokens=getattr(input_details, "cached_tokens", None)
        if input_details
        else None,
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
        if options.tools:
            request["tools"] = [
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description or "",
                    "parameters": spec.input_schema,
                    **({"strict": spec.strict} if spec.strict is not None else {}),
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

        extra_body: dict[str, Any] = {}
        for key in self.provider_options_keys:
            for name, value in (options.provider_options.get(key) or {}).items():
                extra_body.setdefault(name, value)
        if extra_body:
            request["extra_body"] = extra_body
        if options.headers:
            request["extra_headers"] = options.headers
        return request

    @staticmethod
    def _warnings(options: CallOptions) -> list[str]:
        warnings = []
        for name in ("presence_penalty", "frequency_penalty", "seed", "stop_sequences", "top_k"):
            if getattr(options, name) is not None:
                warnings.append(f"{name} is not supported by the OpenAI Responses API.")
        return warnings

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        try:
            response = await client.responses.create(**self._request(options))
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        content: list[AssistantContentPart] = []
        for item in response.output or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in item.content or []:
                    if getattr(part, "type", None) == "output_text":
                        content.append(TextPart(text=part.text))
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

        has_tool_calls = any(isinstance(p, ToolCallPart) for p in content)
        finish, raw_finish = _finish_reason(response, has_tool_calls)
        return ProviderResult(
            content=content,
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            usage=_map_usage(response.usage),
            response=ResponseMetadata(id=response.id, model_id=response.model),
            warnings=self._warnings(options),
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        try:
            stream = await client.responses.create(**self._request(options), stream=True)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

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
                if etype == "response.created":
                    yield ResponseMetadataPart(
                        id=event.response.id, model_id=event.response.model
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
