"""Pure-mapping tests for the Google Gemini adapter extras (no network).

All tests construct SDK pydantic types via model_validate or direct
instantiation — no API calls are made.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.genai import types as gtypes

from pai_sdk.messages import FileIdData, FilePart, ImagePart, TextPart, UrlSourcePart, UserModelMessage
from pai_sdk.provider import CallOptions, FunctionToolSpec
from pai_sdk.providers.google import (
    GoogleLanguageModel,
    _grounding_sources,
    _map_usage,
    _build_request_echo,
    _sanitize_contents_for_request,
    convert_to_gemini_contents,
)
from pai_sdk.results import InputTokenDetails, OutputTokenDetails
from pai_sdk.stream import RawPart, SourceStreamPart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate_with_grounding(chunks: list[dict]) -> Any:
    """Build a Candidate object with grounding_metadata via model_validate."""
    return gtypes.Candidate.model_validate({
        "content": {"role": "model", "parts": [{"text": "answer"}]},
        "finish_reason": "STOP",
        "grounding_metadata": {
            "grounding_chunks": chunks,
            "web_search_queries": ["test query"],
        },
    })


def _make_response(candidate_dict: dict, usage_dict: dict | None = None) -> Any:
    """Build a GenerateContentResponse via model_validate."""
    data: dict[str, Any] = {"candidates": [candidate_dict]}
    if usage_dict:
        data["usage_metadata"] = usage_dict
    return gtypes.GenerateContentResponse.model_validate(data)


def _make_model(model_id: str = "gemini-2.5-flash") -> GoogleLanguageModel:
    return GoogleLanguageModel(model_id=model_id)


# ---------------------------------------------------------------------------
# 1. Grounding → sources in do_generate path
# ---------------------------------------------------------------------------


def test_grounding_sources_web_chunks():
    """_grounding_sources() extracts UrlSourcePart from web chunks with URIs."""
    candidate = _make_candidate_with_grounding([
        {"web": {"uri": "https://example.com/a", "title": "Page A"}},
        {"web": {"uri": "https://example.com/b", "title": None}},
        {"web": {}},               # no URI — should be skipped
        {"retrieved_context": {}}, # no web field — skipped
    ])
    sources = _grounding_sources(candidate)
    assert len(sources) == 2
    assert sources[0].url == "https://example.com/a"
    assert sources[0].title == "Page A"
    assert sources[0].id == "source_0"
    assert sources[1].url == "https://example.com/b"
    assert sources[1].title is None
    assert sources[1].id == "source_1"


def test_grounding_sources_no_metadata():
    """_grounding_sources() returns [] when grounding_metadata is absent."""
    candidate = gtypes.Candidate.model_validate({
        "content": {"role": "model", "parts": [{"text": "hi"}]},
        "finish_reason": "STOP",
    })
    assert _grounding_sources(candidate) == []


async def test_do_generate_grounding_appended_to_content():
    """do_generate appends UrlSourcePart entries to content from grounding."""
    model = _make_model()
    response = _make_response({
        "content": {"role": "model", "parts": [{"text": "grounded answer"}]},
        "finish_reason": "STOP",
        "grounding_metadata": {
            "grounding_chunks": [
                {"web": {"uri": "https://wiki.example.com", "title": "Wiki"}},
            ],
        },
    })

    fake_aio = AsyncMock()
    fake_aio.models.generate_content = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(prompt=standardize_prompt(prompt="tell me"))
    result = await model.do_generate(opts)

    text_parts = [p for p in result.content if isinstance(p, TextPart)]
    source_parts = [p for p in result.content if isinstance(p, UrlSourcePart)]
    assert len(text_parts) == 1
    assert text_parts[0].text == "grounded answer"
    assert len(source_parts) == 1
    assert source_parts[0].url == "https://wiki.example.com"
    assert source_parts[0].title == "Wiki"
    assert source_parts[0].id == "source_0"


# ---------------------------------------------------------------------------
# 2. Usage details mapping
# ---------------------------------------------------------------------------


def test_map_usage_full_details():
    """_map_usage fills input_token_details and output_token_details."""
    metadata = gtypes.GenerateContentResponseUsageMetadata.model_validate({
        "prompt_token_count": 100,
        "cached_content_token_count": 30,
        "candidates_token_count": 50,
        "thoughts_token_count": 20,
        "total_token_count": 170,
    })
    usage = _map_usage(metadata)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 170
    assert usage.reasoning_tokens == 20
    assert usage.cached_input_tokens == 30

    assert usage.input_token_details is not None
    assert usage.input_token_details.cache_read_tokens == 30
    assert usage.input_token_details.no_cache_tokens == 70  # 100 - 30

    assert usage.output_token_details is not None
    assert usage.output_token_details.reasoning_tokens == 20
    assert usage.output_token_details.text_tokens == 50


def test_map_usage_no_cache():
    """_map_usage sets no_cache_tokens=None when cached is absent."""
    metadata = gtypes.GenerateContentResponseUsageMetadata.model_validate({
        "prompt_token_count": 80,
        "candidates_token_count": 40,
        "total_token_count": 120,
    })
    usage = _map_usage(metadata)
    assert usage.input_token_details is not None
    assert usage.input_token_details.cache_read_tokens is None
    # no_cache is None because cached is None (can't compute the split)
    assert usage.input_token_details.no_cache_tokens is None


def test_map_usage_no_thoughts():
    """_map_usage sets output_token_details.reasoning_tokens=None when absent."""
    metadata = gtypes.GenerateContentResponseUsageMetadata.model_validate({
        "candidates_token_count": 60,
    })
    usage = _map_usage(metadata)
    assert usage.output_token_details is not None
    assert usage.output_token_details.reasoning_tokens is None
    assert usage.output_token_details.text_tokens == 60


def test_map_usage_none():
    """_map_usage returns empty Usage when metadata is None."""
    usage = _map_usage(None)
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.input_token_details is None
    assert usage.output_token_details is None


# ---------------------------------------------------------------------------
# 3. FileIdData → file_data mapping in convert_to_gemini_contents
# ---------------------------------------------------------------------------


async def test_file_id_data_string_maps_to_file_data():
    """FilePart with FileIdData(id=str) maps to file_data with that URI."""
    messages = [
        UserModelMessage(content=[
            FilePart(
                data=FileIdData(id="files/abc123"),
                media_type="application/pdf",
            )
        ])
    ]
    contents = await convert_to_gemini_contents(messages)
    part = contents[0]["parts"][0]
    assert "file_data" in part
    assert part["file_data"]["file_uri"] == "files/abc123"
    assert part["file_data"]["mime_type"] == "application/pdf"


async def test_file_id_data_dict_with_file_uri_key():
    """FileIdData(id={"file_uri": "..."}) maps using the file_uri key."""
    messages = [
        UserModelMessage(content=[
            FilePart(
                data=FileIdData(id={"file_uri": "files/xyz999"}),
                media_type="image/png",
            )
        ])
    ]
    contents = await convert_to_gemini_contents(messages)
    part = contents[0]["parts"][0]
    assert part["file_data"]["file_uri"] == "files/xyz999"


async def test_file_id_data_dict_sole_value():
    """FileIdData(id={"somekey": "value"}) maps using the first dict value."""
    messages = [
        UserModelMessage(content=[
            FilePart(
                data=FileIdData(id={"gcs_uri": "gs://bucket/file.mp4"}),
                media_type="video/mp4",
            )
        ])
    ]
    contents = await convert_to_gemini_contents(messages)
    part = contents[0]["parts"][0]
    assert part["file_data"]["file_uri"] == "gs://bucket/file.mp4"


async def test_image_part_file_id_data():
    """ImagePart with FileIdData maps to file_data."""
    messages = [
        UserModelMessage(content=[
            ImagePart(image=FileIdData(id="files/img001"), media_type="image/jpeg")
        ])
    ]
    contents = await convert_to_gemini_contents(messages)
    part = contents[0]["parts"][0]
    assert part["file_data"]["file_uri"] == "files/img001"
    assert part["file_data"]["mime_type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# 4. Request echo presence
# ---------------------------------------------------------------------------


async def test_request_echo_in_do_generate():
    """do_generate sets result.request with model/contents/config."""
    model = _make_model()
    response = _make_response({
        "content": {"role": "model", "parts": [{"text": "hello"}]},
        "finish_reason": "STOP",
    })

    fake_aio = AsyncMock()
    fake_aio.models.generate_content = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(
        prompt=standardize_prompt(system="Be helpful", prompt="hi"),
        max_output_tokens=50,
    )
    result = await model.do_generate(opts)

    assert result.request is not None
    assert result.request["model"] == "gemini-2.5-flash"
    assert "contents" in result.request
    assert "config" in result.request
    assert result.request["config"]["system_instruction"] == "Be helpful"
    assert result.request["config"]["max_output_tokens"] == 50


async def test_request_echo_bytes_sanitized():
    """Inline bytes in request echo are replaced with '<bytes>'."""
    import base64

    model = _make_model()
    response = _make_response({
        "content": {"role": "model", "parts": [{"text": "ok"}]},
        "finish_reason": "STOP",
    })

    fake_aio = AsyncMock()
    fake_aio.models.generate_content = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    model._client_cache = fake_client

    png = b"\x89PNG\r\n\x1a\nfake"
    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(
        prompt=[
            UserModelMessage(content=[ImagePart(image=png, media_type="image/png")])
        ],
    )
    result = await model.do_generate(opts)

    # The contents in the echo must be JSON-serializable (no bytes)
    import json
    json_str = json.dumps(result.request)  # must not raise
    assert "<bytes>" in json_str


# ---------------------------------------------------------------------------
# 5. include_raw_chunks behavior
# ---------------------------------------------------------------------------


async def test_include_raw_chunks_yields_raw_parts():
    """do_stream emits RawPart for each chunk when include_raw_chunks=True."""
    model = _make_model()

    # Build two canned chunk objects
    chunk1 = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}],
    })
    chunk2 = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": " world"}]}, "finish_reason": "STOP"}],
        "usage_metadata": {"candidates_token_count": 2, "prompt_token_count": 5, "total_token_count": 7},
    })

    async def fake_stream():
        yield chunk1
        yield chunk2

    fake_aio_stream = AsyncMock()
    fake_aio_stream.generate_content_stream = AsyncMock(return_value=fake_stream())
    fake_client = MagicMock()
    fake_client.aio.models = fake_aio_stream
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(
        prompt=standardize_prompt(prompt="hi"),
        include_raw_chunks=True,
    )

    parts = []
    async for part in model.do_stream(opts):
        parts.append(part)

    raw_parts = [p for p in parts if isinstance(p, RawPart)]
    # Expect one RawPart per chunk (2 chunks)
    assert len(raw_parts) == 2
    # Each raw_value should be a dict (model_dump of the chunk)
    for rp in raw_parts:
        assert isinstance(rp.raw_value, dict)


async def test_no_raw_chunks_by_default():
    """do_stream does NOT emit RawPart when include_raw_chunks=False (default)."""
    model = _make_model()

    chunk = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finish_reason": "STOP"}],
    })

    async def fake_stream():
        yield chunk

    fake_aio_stream = AsyncMock()
    fake_aio_stream.generate_content_stream = AsyncMock(return_value=fake_stream())
    fake_client = MagicMock()
    fake_client.aio.models = fake_aio_stream
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(prompt=standardize_prompt(prompt="hi"))

    parts = []
    async for part in model.do_stream(opts):
        parts.append(part)

    raw_parts = [p for p in parts if isinstance(p, RawPart)]
    assert len(raw_parts) == 0


# ---------------------------------------------------------------------------
# 6. Grounding sources in do_stream (deduplication)
# ---------------------------------------------------------------------------


async def test_do_stream_grounding_sources_deduped():
    """do_stream emits SourceStreamPart per unique URI, deduped across chunks."""
    model = _make_model()

    chunk1 = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Answer "}]},
            "grounding_metadata": {
                "grounding_chunks": [
                    {"web": {"uri": "https://a.com", "title": "A"}},
                    {"web": {"uri": "https://b.com", "title": "B"}},
                ],
            },
        }],
    })
    chunk2 = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "here."}]},
            "finish_reason": "STOP",
            "grounding_metadata": {
                "grounding_chunks": [
                    # https://a.com repeats — should be deduped
                    {"web": {"uri": "https://a.com", "title": "A"}},
                    {"web": {"uri": "https://c.com", "title": "C"}},
                ],
            },
        }],
    })

    async def fake_stream():
        yield chunk1
        yield chunk2

    fake_aio_stream = AsyncMock()
    fake_aio_stream.generate_content_stream = AsyncMock(return_value=fake_stream())
    fake_client = MagicMock()
    fake_client.aio.models = fake_aio_stream
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(prompt=standardize_prompt(prompt="search this"))

    parts = []
    async for part in model.do_stream(opts):
        parts.append(part)

    source_parts = [p for p in parts if isinstance(p, SourceStreamPart)]
    # a.com (chunk1), b.com (chunk1), c.com (chunk2) — a.com from chunk2 deduped
    assert len(source_parts) == 3
    uris = [sp.source.url for sp in source_parts]
    assert "https://a.com" in uris
    assert "https://b.com" in uris
    assert "https://c.com" in uris
    # a.com should appear exactly once
    assert uris.count("https://a.com") == 1


# ---------------------------------------------------------------------------
# 7. provider_metadata in do_generate
# ---------------------------------------------------------------------------


async def test_provider_metadata_with_grounding():
    """do_generate sets provider_metadata['google'] with grounding summary."""
    model = _make_model()
    response = _make_response({
        "content": {"role": "model", "parts": [{"text": "result"}]},
        "finish_reason": "STOP",
        "grounding_metadata": {
            "grounding_chunks": [
                {"web": {"uri": "https://x.com", "title": "X"}},
                {"web": {"uri": "https://y.com", "title": "Y"}},
            ],
            "web_search_queries": ["my query"],
        },
    })

    fake_aio = AsyncMock()
    fake_aio.models.generate_content = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(prompt=standardize_prompt(prompt="search"))
    result = await model.do_generate(opts)

    assert result.provider_metadata is not None
    google_meta = result.provider_metadata["google"]
    assert "grounding_metadata" in google_meta
    assert google_meta["grounding_metadata"]["grounding_chunk_count"] == 2
    assert google_meta["grounding_metadata"]["web_search_queries"] == ["my query"]


async def test_provider_metadata_none_when_no_extra_info():
    """do_generate returns None provider_metadata when no extras are present."""
    model = _make_model()
    response = _make_response({
        "content": {"role": "model", "parts": [{"text": "plain"}]},
        "finish_reason": "STOP",
    })

    fake_aio = AsyncMock()
    fake_aio.models.generate_content = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    opts = CallOptions(prompt=standardize_prompt(prompt="hi"))
    result = await model.do_generate(opts)

    assert result.provider_metadata is None


# ---------------------------------------------------------------------------
# 8. Request echo in do_stream (ResponseMetadataPart)
# ---------------------------------------------------------------------------


async def test_stream_request_echo_in_response_metadata():
    """do_stream attaches request echo to ResponseMetadataPart."""
    model = _make_model()

    chunk = gtypes.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finish_reason": "STOP"}],
    })

    async def fake_stream():
        yield chunk

    fake_aio_stream = AsyncMock()
    fake_aio_stream.generate_content_stream = AsyncMock(return_value=fake_stream())
    fake_client = MagicMock()
    fake_client.aio.models = fake_aio_stream
    model._client_cache = fake_client

    from pai_sdk._prompt import standardize_prompt
    from pai_sdk.stream import ResponseMetadataPart as RMP
    opts = CallOptions(prompt=standardize_prompt(prompt="hi"), max_output_tokens=10)

    parts = []
    async for part in model.do_stream(opts):
        parts.append(part)

    metadata_parts = [p for p in parts if isinstance(p, RMP)]
    assert len(metadata_parts) == 1
    req = metadata_parts[0].request
    assert req is not None
    assert req["model"] == "gemini-2.5-flash"
    assert req["config"]["max_output_tokens"] == 10


# ---------------------------------------------------------------------------
# 9. _sanitize_contents_for_request helper
# ---------------------------------------------------------------------------


def test_sanitize_contents_replaces_bytes():
    """Bytes in inline_data are replaced with '<bytes>' placeholder."""
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "look at this"},
                {"inline_data": {"mime_type": "image/png", "data": b"\x89PNG..."}},
            ],
        }
    ]
    sanitized = _sanitize_contents_for_request(contents)
    assert sanitized[0]["parts"][0] == {"text": "look at this"}
    assert sanitized[0]["parts"][1]["inline_data"]["data"] == "<bytes>"
    assert sanitized[0]["parts"][1]["inline_data"]["mime_type"] == "image/png"
    # original is not mutated
    assert isinstance(contents[0]["parts"][1]["inline_data"]["data"], bytes)


def test_sanitize_contents_non_bytes_unchanged():
    """Non-bytes inline_data data fields pass through unchanged."""
    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": "base64string"}},
            ],
        }
    ]
    sanitized = _sanitize_contents_for_request(contents)
    assert sanitized[0]["parts"][0]["inline_data"]["data"] == "base64string"
