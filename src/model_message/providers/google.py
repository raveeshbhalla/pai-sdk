"""Google Gemini provider via the google-genai SDK.

Notes:
- System messages map to config.system_instruction.
- http(s) image/file URLs are downloaded (Gemini only takes inline bytes or
  Files API URIs); Files API URIs can be passed via FilePart with
  provider_options={"google": {"file_uri": ...}}.
- Gemini 3 thought signatures are preserved in
  provider_options["google"]["thought_signature"] and echoed on replay.
- providerOptions under the "google" key are merged into the generation
  config (e.g. {"google": {"thinking_config": {"thinking_level": "high"}}}).
"""

from __future__ import annotations

import base64
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
    FilePartEvent,
    ProviderStreamPart,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    ResponseMetadataPart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputEnd,
    ToolInputStart,
)
from ._util import is_url, to_bytes, wrap_provider_error

_FINISH_REASONS: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content-filter",
    "RECITATION": "content-filter",
    "BLOCKLIST": "content-filter",
    "PROHIBITED_CONTENT": "content-filter",
    "SPII": "content-filter",
    "IMAGE_SAFETY": "content-filter",
    "LANGUAGE": "other",
    "MALFORMED_FUNCTION_CALL": "error",
    "UNEXPECTED_TOOL_CALL": "error",
    "OTHER": "other",
}


async def _download(url: str) -> tuple[bytes, Optional[str]]:
    import httpx

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content, response.headers.get("content-type")


async def _media_part(data: Any, media_type: Optional[str], provider_options) -> dict[str, Any]:
    file_uri = ((provider_options or {}).get("google") or {}).get("file_uri")
    if file_uri:
        return {"file_data": {"file_uri": file_uri, "mime_type": media_type}}
    if is_url(data):
        raw, content_type = await _download(data)
        media_type = media_type or (content_type.split(";")[0] if content_type else None)
    else:
        raw = to_bytes(data)
    if media_type is None:
        from ._util import detect_media_type

        media_type = detect_media_type(raw)
    return {
        "inline_data": {
            "mime_type": media_type or "application/octet-stream",
            "data": raw,
        }
    }


def _function_response_value(part: ToolResultPart) -> dict[str, Any]:
    output = part.output
    if isinstance(output, TextOutput):
        return {"result": output.value}
    if isinstance(output, JsonOutput):
        value = output.value
        return value if isinstance(value, dict) else {"result": value}
    if isinstance(output, ErrorTextOutput):
        return {"error": output.value}
    if isinstance(output, ErrorJsonOutput):
        value = output.value
        return {"error": value if not isinstance(value, dict) else json.dumps(value)}
    if isinstance(output, ContentOutput):
        texts = [item.text for item in output.value if item.type == "text"]
        return {"result": "\n".join(texts)}
    return {"result": str(output)}


async def convert_to_gemini_contents(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserModelMessage):
            parts: list[dict[str, Any]] = []
            if isinstance(message.content, str):
                parts.append({"text": message.content})
            else:
                for part in message.content:
                    if isinstance(part, TextPart):
                        parts.append({"text": part.text})
                    elif isinstance(part, ImagePart):
                        parts.append(
                            await _media_part(part.image, part.media_type, part.provider_options)
                        )
                    elif isinstance(part, FilePart):
                        parts.append(
                            await _media_part(part.data, part.media_type, part.provider_options)
                        )
            contents.append({"role": "user", "parts": parts})

        elif isinstance(message, AssistantModelMessage):
            parts = []
            if isinstance(message.content, str):
                parts.append({"text": message.content})
            else:
                for part in message.content:
                    google_opts = (part.provider_options or {}).get("google") or {}
                    signature = google_opts.get("thought_signature")
                    if isinstance(part, TextPart):
                        entry: dict[str, Any] = {"text": part.text}
                        if signature:
                            entry["thought_signature"] = base64.b64decode(signature)
                        parts.append(entry)
                    elif isinstance(part, ReasoningPart):
                        # Thought summaries are not replayed; signatures ride
                        # on the parts they were attached to.
                        continue
                    elif isinstance(part, ToolCallPart):
                        entry = {
                            "function_call": {
                                "name": part.tool_name,
                                "args": part.input or {},
                            }
                        }
                        call_id = google_opts.get("function_call_id")
                        if call_id:
                            entry["function_call"]["id"] = call_id
                        if signature:
                            entry["thought_signature"] = base64.b64decode(signature)
                        parts.append(entry)
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif isinstance(message, ToolModelMessage):
            parts = []
            for part in message.content:
                google_opts = (part.provider_options or {}).get("google") or {}
                response_entry: dict[str, Any] = {
                    "name": part.tool_name,
                    "response": _function_response_value(part),
                }
                call_id = google_opts.get("function_call_id")
                if call_id:
                    response_entry["id"] = call_id
                parts.append({"function_response": response_entry})
            contents.append({"role": "user", "parts": parts})
    return contents


def _map_usage(metadata: Any) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=getattr(metadata, "prompt_token_count", None),
        output_tokens=getattr(metadata, "candidates_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
        reasoning_tokens=getattr(metadata, "thoughts_token_count", None),
        cached_input_tokens=getattr(metadata, "cached_content_token_count", None),
    )


def _finish_name(finish_reason: Any) -> Optional[str]:
    if finish_reason is None:
        return None
    return getattr(finish_reason, "name", None) or str(finish_reason)


def _signature_options(part: Any) -> Optional[dict[str, dict[str, Any]]]:
    signature = getattr(part, "thought_signature", None)
    if not signature:
        return None
    if isinstance(signature, bytes):
        signature = base64.b64encode(signature).decode()
    return {"google": {"thought_signature": signature}}


@dataclass
class GoogleLanguageModel(LanguageModel):
    model_id: str
    api_key: Optional[str] = None
    provider: str = "google.generative-ai"
    _client_cache: Any = field(default=None, repr=False, compare=False)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            from google import genai
        except ImportError as exc:
            raise MissingDependencyError("google-genai", "google") from exc
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client_cache = genai.Client(**kwargs)
        return self._client_cache

    async def _build(self, options: CallOptions) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from ._util import system_and_rest

        system_texts, rest = system_and_rest(options.prompt)
        contents = await convert_to_gemini_contents(rest)

        config: dict[str, Any] = {}
        if system_texts:
            config["system_instruction"] = "\n\n".join(system_texts)
        if options.max_output_tokens is not None:
            config["max_output_tokens"] = options.max_output_tokens
        if options.temperature is not None:
            config["temperature"] = options.temperature
        if options.top_p is not None:
            config["top_p"] = options.top_p
        if options.top_k is not None:
            config["top_k"] = options.top_k
        if options.presence_penalty is not None:
            config["presence_penalty"] = options.presence_penalty
        if options.frequency_penalty is not None:
            config["frequency_penalty"] = options.frequency_penalty
        if options.stop_sequences:
            config["stop_sequences"] = options.stop_sequences
        if options.seed is not None:
            config["seed"] = options.seed
        if options.tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": spec.name,
                            "description": spec.description or "",
                            "parameters_json_schema": spec.input_schema,
                        }
                        for spec in options.tools
                    ]
                }
            ]
        if options.tool_choice is not None:
            choice = options.tool_choice
            mode_map = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
            fc_config: dict[str, Any]
            if isinstance(choice, str) and choice in mode_map:
                fc_config = {"mode": mode_map[choice]}
            elif isinstance(choice, dict):
                fc_config = {
                    "mode": "ANY",
                    "allowed_function_names": [
                        choice.get("tool_name") or choice.get("toolName")
                    ],
                }
            else:
                fc_config = {"mode": "AUTO"}
            config["tool_config"] = {"function_calling_config": fc_config}
        if options.response_format and options.response_format.get("type") == "json":
            config["response_mime_type"] = "application/json"
            if options.response_format.get("schema"):
                config["response_json_schema"] = options.response_format["schema"]
        for name, value in (options.provider_options.get("google") or {}).items():
            config.setdefault(name, value)
        return contents, config

    def _content_parts(self, candidate: Any) -> list[AssistantContentPart]:
        parts: list[AssistantContentPart] = []
        candidate_content = getattr(candidate, "content", None)
        for part in (getattr(candidate_content, "parts", None) or []):
            signature_opts = _signature_options(part)
            if getattr(part, "text", None) is not None:
                if getattr(part, "thought", False):
                    parts.append(
                        ReasoningPart(text=part.text, provider_options=signature_opts)
                    )
                else:
                    parts.append(
                        TextPart(text=part.text, provider_options=signature_opts)
                    )
            elif getattr(part, "function_call", None) is not None:
                call = part.function_call
                call_id = getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:24]}"
                provider_options = signature_opts or {}
                provider_options.setdefault("google", {})["function_call_id"] = getattr(
                    call, "id", None
                )
                parts.append(
                    ToolCallPart(
                        tool_call_id=call_id,
                        tool_name=call.name,
                        input=dict(call.args or {}),
                        provider_options=provider_options,
                    )
                )
            elif getattr(part, "inline_data", None) is not None:
                blob = part.inline_data
                data = blob.data
                if isinstance(data, str):
                    data = base64.b64decode(data)
                parts.append(FilePart(data=data, media_type=blob.mime_type or "application/octet-stream"))
        return parts

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        client = self._client()
        contents, config = await self._build(options)
        try:
            response = await client.aio.models.generate_content(
                model=self.model_id, contents=contents, config=config or None
            )
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Google") from exc

        candidates = getattr(response, "candidates", None) or []
        content: list[AssistantContentPart] = (
            self._content_parts(candidates[0]) if candidates else []
        )
        raw_finish = _finish_name(candidates[0].finish_reason) if candidates else None
        has_tool_calls = any(isinstance(p, ToolCallPart) for p in content)
        finish = _FINISH_REASONS.get(raw_finish or "", "unknown")
        if finish == "stop" and has_tool_calls:
            finish = "tool-calls"
        return ProviderResult(
            content=content,
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            usage=_map_usage(getattr(response, "usage_metadata", None)),
            response=ResponseMetadata(
                id=getattr(response, "response_id", None),
                model_id=getattr(response, "model_version", None) or self.model_id,
            ),
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        contents, config = await self._build(options)
        try:
            stream = await client.aio.models.generate_content_stream(
                model=self.model_id, contents=contents, config=config or None
            )
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Google") from exc

        usage = Usage()
        raw_finish: Optional[str] = None
        has_tool_calls = False
        sent_metadata = False
        text_open = False
        reasoning_open = False

        try:
            async for chunk in stream:
                if not sent_metadata:
                    sent_metadata = True
                    yield ResponseMetadataPart(
                        id=getattr(chunk, "response_id", None),
                        model_id=getattr(chunk, "model_version", None) or self.model_id,
                    )
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = _map_usage(chunk.usage_metadata)
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                candidate = candidates[0]
                for part in self._content_parts(candidate):
                    if isinstance(part, ReasoningPart):
                        if not reasoning_open:
                            reasoning_open = True
                            yield ReasoningStart(id="r0")
                        yield ReasoningDelta(id="r0", text=part.text)
                    elif isinstance(part, TextPart):
                        if reasoning_open:
                            reasoning_open = False
                            yield ReasoningEnd(id="r0")
                        if not text_open:
                            text_open = True
                            yield TextStart(id="0")
                        yield TextDelta(id="0", text=part.text)
                    elif isinstance(part, ToolCallPart):
                        has_tool_calls = True
                        yield ToolInputStart(
                            id=part.tool_call_id, tool_name=part.tool_name
                        )
                        yield ToolInputEnd(id=part.tool_call_id)
                        yield part
                    elif isinstance(part, FilePart):
                        yield FilePartEvent(
                            media_type=part.media_type, data=to_bytes(part.data)
                        )
                if candidate.finish_reason is not None:
                    raw_finish = _finish_name(candidate.finish_reason)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Google") from exc

        if reasoning_open:
            yield ReasoningEnd(id="r0")
        if text_open:
            yield TextEnd(id="0")
        finish = _FINISH_REASONS.get(raw_finish or "", "unknown")
        if finish == "stop" and has_tool_calls:
            finish = "tool-calls"
        yield Finish(finish_reason=finish, raw_finish_reason=raw_finish, total_usage=usage)
