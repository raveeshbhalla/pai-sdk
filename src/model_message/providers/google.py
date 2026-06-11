"""Google Gemini provider via the google-genai SDK.

Notes:
- System messages map to config.system_instruction.
- http(s) image/file URLs are downloaded (Gemini only takes inline bytes or
  Files API URIs); Files API URIs can be passed via FilePart with
  provider_options={"google": {"file_uri": ...}} or via FileIdData.
- Gemini 3 thought signatures are preserved in
  provider_options["google"]["thought_signature"] and echoed on replay.
- providerOptions under the "google" key are merged into the generation
  config (e.g. {"google": {"thinking_config": {"thinking_level": "high"}}}).
- When grounding is enabled, grounding chunks with web URIs are mapped to
  UrlSourcePart entries appended to content (and SourceStreamPart in
  streaming). Sources are deduped by URI in streaming.
- ProviderResult.request / ResponseMetadataPart.request echo the JSON-able
  request (model/contents/config); bytes in contents are summarized as
  "<bytes>" placeholders to keep it JSON-serializable.
- When options.include_raw_chunks is True, a RawPart(raw_value=chunk.model_dump())
  is yielded for each raw stream chunk in addition to mapped parts.
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
    FileIdData,
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
from ..results import FinishReason, InputTokenDetails, OutputTokenDetails, ResponseMetadata, Usage
from ..stream import (
    Finish,
    FilePartEvent,
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


def _file_uri_from_file_id(file_id_data: FileIdData) -> str:
    """Extract the file URI string from a FileIdData instance."""
    id_val = file_id_data.id
    if isinstance(id_val, str):
        return id_val
    # dict variant: look for "file_uri" key, else take the sole value
    if isinstance(id_val, dict):
        if "file_uri" in id_val:
            return id_val["file_uri"]
        if id_val:
            return next(iter(id_val.values()))
    return str(id_val)


async def _media_part(data: Any, media_type: Optional[str], provider_options) -> dict[str, Any]:
    # FileIdData: map to file_data (Files API / GCS URI reference)
    if isinstance(data, FileIdData):
        file_uri = _file_uri_from_file_id(data)
        return {"file_data": {"file_uri": file_uri, "mime_type": media_type}}
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
    prompt = getattr(metadata, "prompt_token_count", None)
    cached = getattr(metadata, "cached_content_token_count", None)
    thoughts = getattr(metadata, "thoughts_token_count", None)
    candidates = getattr(metadata, "candidates_token_count", None)

    # input_token_details: cache_read=cached, no_cache=prompt-cached (when both present)
    input_details: Optional[InputTokenDetails] = None
    if cached is not None or prompt is not None:
        no_cache: Optional[int] = None
        if prompt is not None and cached is not None:
            no_cache = prompt - cached
        input_details = InputTokenDetails(
            cache_read_tokens=cached,
            no_cache_tokens=no_cache,
        )

    # output_token_details: reasoning=thoughts, text=candidates
    output_details: Optional[OutputTokenDetails] = None
    if thoughts is not None or candidates is not None:
        output_details = OutputTokenDetails(
            reasoning_tokens=thoughts,
            text_tokens=candidates,
        )

    return Usage(
        input_tokens=prompt,
        output_tokens=candidates,
        total_tokens=getattr(metadata, "total_token_count", None),
        reasoning_tokens=thoughts,
        cached_input_tokens=cached,
        input_token_details=input_details,
        output_token_details=output_details,
    )


def _grounding_sources(candidate: Any) -> list[UrlSourcePart]:
    """Extract UrlSourcePart entries from grounding_metadata.grounding_chunks.

    Only chunks with a non-empty web.uri are included. Deduplication by URI
    is left to the caller (streaming) or omitted (generate, where the full
    list is already deduplicated by the response).

    SDK verified fields:
    - candidate.grounding_metadata: GroundingMetadata | None
    - GroundingMetadata.grounding_chunks: list[GroundingChunk] | None
    - GroundingChunk.web: GroundingChunkWeb | None
    - GroundingChunkWeb.uri: str | None
    - GroundingChunkWeb.title: str | None
    """
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if grounding_metadata is None:
        return []
    chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    sources: list[UrlSourcePart] = []
    for idx, chunk in enumerate(chunks):
        web = getattr(chunk, "web", None)
        if web is None:
            continue
        uri = getattr(web, "uri", None)
        if not uri:
            continue
        title = getattr(web, "title", None)
        sources.append(
            UrlSourcePart(id=f"source_{idx}", url=uri, title=title)
        )
    return sources


def _sanitize_contents_for_request(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-able copy of contents with raw bytes replaced by '<bytes>'."""
    result = []
    for content in contents:
        sanitized_parts = []
        for part in content.get("parts", []):
            if "inline_data" in part:
                inline = dict(part["inline_data"])
                if isinstance(inline.get("data"), bytes):
                    inline["data"] = "<bytes>"
                sanitized_parts.append({"inline_data": inline})
            else:
                sanitized_parts.append(part)
        result.append({**content, "parts": sanitized_parts})
    return result


def _build_request_echo(model_id: str, contents: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-able request dict to echo back on results."""
    return {
        "model": model_id,
        "contents": _sanitize_contents_for_request(contents),
        "config": config,
    }


def _build_provider_metadata(candidate: Any) -> Optional[dict[str, dict[str, Any]]]:
    """Build provider_metadata['google'] from a response candidate, defensively."""
    meta: dict[str, Any] = {}

    finish_message = getattr(candidate, "finish_message", None)
    if finish_message is not None:
        meta["finish_message"] = finish_message

    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if grounding_metadata is not None:
        try:
            # Keep it small: only web_search_queries and grounding_chunk count
            summary: dict[str, Any] = {}
            chunks = getattr(grounding_metadata, "grounding_chunks", None)
            if chunks is not None:
                summary["grounding_chunk_count"] = len(chunks)
            web_queries = getattr(grounding_metadata, "web_search_queries", None)
            if web_queries:
                summary["web_search_queries"] = web_queries
            meta["grounding_metadata"] = summary
        except Exception:  # noqa: BLE001
            pass

    safety_ratings = getattr(candidate, "safety_ratings", None)
    if safety_ratings is not None:
        try:
            meta["safety_ratings"] = [
                r.model_dump(exclude_none=True) for r in safety_ratings
            ]
        except Exception:  # noqa: BLE001
            pass

    if not meta:
        return None
    return {"google": meta}


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

    def _compute_warnings(self, options: CallOptions) -> list[Any]:
        """Return CallWarning list for anything we cannot honour.

        Currently: a tool_choice dict naming a tool that is not in tools
        (type="other", because the setting itself is valid — the name is wrong).
        All other CallOptions fields are mapped to Gemini config in _build.
        """
        from ..results import CallWarning

        warnings: list[Any] = []
        choice = options.tool_choice
        if isinstance(choice, dict):
            tool_name = choice.get("tool_name") or choice.get("toolName")
            tool_names = {spec.name for spec in (options.tools or [])}
            if tool_name and tool_names and tool_name not in tool_names:
                warnings.append(
                    CallWarning(
                        type="other",
                        message=(
                            f"tool_choice names '{tool_name}' but it is not in the tools list. "
                            f"Gemini will use ANY mode over all available tools."
                        ),
                    )
                )
        return warnings

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
        warnings = self._compute_warnings(options)
        request_echo = _build_request_echo(self.model_id, contents, config)
        try:
            response = await client.aio.models.generate_content(
                model=self.model_id, contents=contents, config=config or None
            )
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, "Google") from exc

        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        content: list[AssistantContentPart] = (
            self._content_parts(candidate) if candidate is not None else []
        )

        # Append grounding sources to content
        if candidate is not None:
            sources = _grounding_sources(candidate)
            content.extend(sources)

        raw_finish = _finish_name(candidate.finish_reason) if candidate is not None else None
        has_tool_calls = any(isinstance(p, ToolCallPart) for p in content)
        finish = _FINISH_REASONS.get(raw_finish or "", "unknown")
        if finish == "stop" and has_tool_calls:
            finish = "tool-calls"

        provider_metadata = _build_provider_metadata(candidate) if candidate is not None else None

        return ProviderResult(
            content=content,
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            usage=_map_usage(getattr(response, "usage_metadata", None)),
            response=ResponseMetadata(
                id=getattr(response, "response_id", None),
                model_id=getattr(response, "model_version", None) or self.model_id,
            ),
            warnings=warnings,
            provider_metadata=provider_metadata,
            request=request_echo,
        )

    async def do_stream(
        self, options: CallOptions
    ) -> AsyncIterator[ProviderStreamPart]:
        client = self._client()
        contents, config = await self._build(options)
        warnings = self._compute_warnings(options)
        request_echo = _build_request_echo(self.model_id, contents, config)
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
        # Track URIs of sources already emitted to deduplicate across chunks
        seen_source_uris: set[str] = set()
        last_candidate: Any = None

        try:
            async for chunk in stream:
                # Emit raw chunk if requested (before any processing)
                if options.include_raw_chunks:
                    try:
                        yield RawPart(raw_value=chunk.model_dump())
                    except Exception:  # noqa: BLE001
                        yield RawPart(raw_value=None)

                if not sent_metadata:
                    sent_metadata = True
                    yield ResponseMetadataPart(
                        id=getattr(chunk, "response_id", None),
                        model_id=getattr(chunk, "model_version", None) or self.model_id,
                        request=request_echo,
                    )
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = _map_usage(chunk.usage_metadata)
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                candidate = candidates[0]
                last_candidate = candidate
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

                # Emit new grounding sources (deduped by URI)
                for source in _grounding_sources(candidate):
                    if source.url not in seen_source_uris:
                        seen_source_uris.add(source.url)
                        yield SourceStreamPart(source=source)

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
