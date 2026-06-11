"""Anthropic adapter parity tests: server-side tools, citations -> sources,
warnings, usage details + provider_metadata, request echo, raw chunks, and
FileIdData -> Files-API mapping.

Canned response payloads are built so the real anthropic SDK parses them,
driven through httpx.MockTransport (same style as test_sdk_integration).
"""

from __future__ import annotations

import json

import httpx
import pytest

from model_message import (
    AssistantModelMessage,
    FileIdData,
    FilePart,
    ImagePart,
    UserModelMessage,
    generate_text,
    step_count_is,
    stream_text,
)
from model_message._prompt import standardize_prompt
from model_message.provider import CallOptions
from model_message.providers.anthropic import AnthropicLanguageModel

anthropic_sdk = pytest.importorskip("anthropic")


def anthropic_model(handler) -> AnthropicLanguageModel:
    model = AnthropicLanguageModel(model_id="claude-opus-4-8", api_key="test")
    model._client_cache = anthropic_sdk.AsyncAnthropic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


def options(messages, **kwargs):
    return CallOptions(prompt=messages, **kwargs)


def sse(events: list[tuple[str, dict]]) -> bytes:
    out = []
    for name, data in events:
        out.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode()


# --- server-side (provider-executed) tools ----------------------------------


WEB_SEARCH_RESPONSE = {
    "id": "msg_ws1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "python release"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://python.org/a",
                    "title": "Python A",
                    "encrypted_content": "enc-a",
                    "page_age": "1 day",
                },
                {
                    "type": "web_search_result",
                    "url": "https://python.org/b",
                    "title": "Python B",
                    "encrypted_content": "enc-b",
                    "page_age": None,
                },
            ],
        },
        {
            "type": "text",
            "text": "Python 3.14 is out.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://python.org/a",
                    "title": "Python A",
                    "cited_text": "Python 3.14",
                    "encrypted_index": "idx-1",
                }
            ],
        },
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_read_input_tokens": 8,
        "cache_creation_input_tokens": 4,
    },
}

FOLLOWUP_RESPONSE = {
    "id": "msg_ws2",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [{"type": "text", "text": "Anything else?"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 60, "output_tokens": 5},
}


async def test_server_tool_use_parts_sources_and_history_echo():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        payload = WEB_SEARCH_RESPONSE if len(requests) == 1 else FOLLOWUP_RESPONSE
        return httpx.Response(200, json=payload)

    # No local tools registered -> provider-executed calls must NOT trigger
    # local execution (no error, loop completes in one step).
    result = await generate_text(
        model=anthropic_model(handler),
        prompt="Search for the latest python release",
    )

    assert result.text == "Python 3.14 is out."
    assert result.finish_reason == "stop"

    # provider-executed tool call + result present in content
    calls = [p for p in result.content if getattr(p, "type", None) == "tool-call"]
    assert len(calls) == 1
    assert calls[0].provider_executed is True
    assert calls[0].tool_name == "web_search"
    assert calls[0].tool_call_id == "srvtoolu_1"

    tool_results = [
        p for p in result.content if getattr(p, "type", None) == "tool-result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].provider_executed is True
    # serialized block content
    assert tool_results[0].output.value[0]["url"] == "https://python.org/a"

    # sources: 2 web_search_result entries + 1 citation (deduped by url? no — 3)
    urls = [s.url for s in result.sources]
    assert "https://python.org/a" in urls
    assert "https://python.org/b" in urls
    assert all(s.id for s in result.sources)
    assert len(result.sources) == 3  # 2 results + 1 citation

    # follow up: append response.messages then send another turn
    history = [
        UserModelMessage(content="Search for the latest python release"),
        *result.response.messages,
        UserModelMessage(content="thanks"),
    ]
    await generate_text(model=anthropic_model(handler), messages=history)

    # second request must echo the server_tool_use + web_search_tool_result
    second = requests[1]
    assistant_blocks = second["messages"][1]["content"]
    types = [b["type"] for b in assistant_blocks]
    assert "server_tool_use" in types
    assert "web_search_tool_result" in types
    server_block = next(b for b in assistant_blocks if b["type"] == "server_tool_use")
    assert server_block["id"] == "srvtoolu_1"
    assert server_block["input"] == {"query": "python release"}
    ws_block = next(
        b for b in assistant_blocks if b["type"] == "web_search_tool_result"
    )
    assert ws_block["tool_use_id"] == "srvtoolu_1"
    assert ws_block["content"][0]["url"] == "https://python.org/a"


# --- citations -> sources (no server tool) ----------------------------------


CITATION_RESPONSE = {
    "id": "msg_c1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [
        {
            "type": "text",
            "text": "See the docs.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://docs.example.com",
                    "title": "Docs",
                    "cited_text": "the docs",
                    "encrypted_index": "ix",
                }
            ],
        }
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


async def test_citations_become_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CITATION_RESPONSE)

    result = await generate_text(model=anthropic_model(handler), prompt="docs?")
    assert result.text == "See the docs."
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://docs.example.com"
    assert result.sources[0].title == "Docs"
    assert result.sources[0].source_type == "url"


# --- warnings for unsupported settings --------------------------------------


PLAIN_RESPONSE = {
    "id": "msg_p1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": "STOP",
    "usage": {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 2,
    },
}


async def test_warnings_for_seed_and_penalties():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=PLAIN_RESPONSE)

    result = await generate_text(
        model=anthropic_model(handler),
        prompt="hi",
        seed=42,
        presence_penalty=0.5,
        frequency_penalty=0.3,
    )
    settings = {w.setting for w in result.warnings}
    assert settings == {"seed", "presence_penalty", "frequency_penalty"}
    assert all(w.type == "unsupported-setting" for w in result.warnings)


# --- usage details + provider_metadata + request echo -----------------------


async def test_usage_details_provider_metadata_and_request_echo():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=PLAIN_RESPONSE)

    result = await generate_text(
        model=anthropic_model(handler), prompt="hi", max_output_tokens=99
    )

    usage = result.usage
    assert usage.cached_input_tokens == 5
    assert usage.input_token_details is not None
    assert usage.input_token_details.cache_read_tokens == 5
    assert usage.input_token_details.cache_write_tokens == 2
    assert usage.input_token_details.no_cache_tokens == 12

    meta = result.provider_metadata["anthropic"]
    assert meta["cache_read_input_tokens"] == 5
    assert meta["cache_creation_input_tokens"] == 2
    assert meta["stop_sequence"] == "STOP"

    # request echo: present, JSON-able, no extra_headers leaked
    assert result.request is not None
    assert result.request["max_tokens"] == 99
    assert "extra_headers" not in result.request


def test_request_echo_strips_headers():
    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    from model_message.providers._util import request_echo

    request = model._request(
        options(standardize_prompt(prompt="hi"), headers={"x-test": "1"})
    )
    assert "extra_headers" in request
    echoed = request_echo(request)
    assert "extra_headers" not in echoed
    assert echoed["model"] == "claude-opus-4-8"


# --- include_raw_chunks -> RawPart ------------------------------------------


RAW_SSE = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_r1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


async def test_include_raw_chunks_yields_rawpart():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(RAW_SSE),
            headers={"content-type": "text/event-stream"},
        )

    model = anthropic_model(handler)
    parts = [
        p
        async for p in model.do_stream(
            options(standardize_prompt(prompt="hi"), include_raw_chunks=True)
        )
    ]
    raw_parts = [p for p in parts if p.type == "raw"]
    # one RawPart per SSE event
    assert len(raw_parts) == len(RAW_SSE)
    assert raw_parts[0].raw_value["type"] == "message_start"

    # without the flag, no RawParts
    model2 = anthropic_model(handler)
    parts2 = [
        p
        async for p in model2.do_stream(
            options(standardize_prompt(prompt="hi"))
        )
    ]
    assert not [p for p in parts2 if p.type == "raw"]


# --- streaming server tools -------------------------------------------------


STREAM_SERVER_TOOL_SSE = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_st1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_s1",
                "name": "web_search",
                "input": {},
            },
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"x"}'},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_s1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://s.example.com",
                        "title": "S",
                        "encrypted_content": "enc",
                        "page_age": None,
                    }
                ],
            },
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


async def test_streaming_server_tools_emit_parts_and_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(STREAM_SERVER_TOOL_SSE),
            headers={"content-type": "text/event-stream"},
        )

    model = anthropic_model(handler)
    parts = [
        p async for p in model.do_stream(options(standardize_prompt(prompt="go")))
    ]

    tool_calls = [p for p in parts if p.type == "tool-call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].provider_executed is True
    assert tool_calls[0].input == {"query": "x"}

    tool_results = [p for p in parts if p.type == "tool-result"]
    assert len(tool_results) == 1
    assert tool_results[0].provider_executed is True
    assert tool_results[0].tool_name == "web_search"

    sources = [p for p in parts if p.type == "source"]
    assert len(sources) == 1
    assert sources[0].source.url == "https://s.example.com"

    starts = [
        p
        for p in parts
        if p.type == "tool-input-start" and p.provider_executed
    ]
    assert len(starts) == 1


async def test_streaming_server_tool_end_to_end_via_stream_text():
    """The full stream_text pipeline must not locally execute provider tools."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(STREAM_SERVER_TOOL_SSE),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(model=anthropic_model(handler), prompt="go")
    full = [p async for p in result.full_stream]
    assert await result.finish_reason == "stop"
    sources = await result.sources
    assert any(s.url == "https://s.example.com" for s in sources)
    # provider-executed tool-result surfaced, no tool-error
    assert not [p for p in full if getattr(p, "type", None) == "tool-error"]


# --- FileIdData mapping (pure _request, no transport) -----------------------


def test_file_id_image_and_document_mapping():
    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    request = model._request(
        options(
            [
                UserModelMessage(
                    content=[
                        ImagePart(
                            image=FileIdData(id="file_img_1"),
                            media_type="image/png",
                        ),
                        FilePart(
                            data=FileIdData(id="file_doc_1"),
                            media_type="application/pdf",
                            filename="report.pdf",
                        ),
                    ]
                )
            ]
        )
    )
    content = request["messages"][0]["content"]
    image_block = content[0]
    assert image_block["type"] == "image"
    assert image_block["source"] == {"type": "file", "file_id": "file_img_1"}
    doc_block = content[1]
    assert doc_block["type"] == "document"
    assert doc_block["source"] == {"type": "file", "file_id": "file_doc_1"}
    assert doc_block["title"] == "report.pdf"


def test_file_id_dict_id_resolution():
    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    request = model._request(
        options(
            [
                UserModelMessage(
                    content=[
                        ImagePart(
                            image=FileIdData(id={"file_id": "file_xyz"}),
                            media_type="image/jpeg",
                        ),
                    ]
                )
            ]
        )
    )
    block = request["messages"][0]["content"][0]
    assert block["source"] == {"type": "file", "file_id": "file_xyz"}
