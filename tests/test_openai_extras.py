"""Extra-feature tests for the OpenAI Chat + Responses adapters.

Like test_sdk_integration, these run the full pipeline through the real
openai SDK with a mocked HTTP transport, so canned payloads must satisfy the
SDK's response parsing (required fields present).
"""

from __future__ import annotations

import json

import httpx
import pytest

from pai_sdk import generate_text, stream_text
from pai_sdk.messages import (
    AssistantModelMessage,
    FileIdData,
    FilePart,
    ImagePart,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UrlSourcePart,
    UserModelMessage,
)
from pai_sdk.providers.openai_chat import (
    OpenAIChatLanguageModel,
    convert_to_chat_messages,
)
from pai_sdk.providers.openai_responses import (
    OpenAIResponsesLanguageModel,
    convert_to_responses_input,
)
from pai_sdk.results import CallWarning

openai_sdk = pytest.importorskip("openai")


def chat_model(handler) -> OpenAIChatLanguageModel:
    model = OpenAIChatLanguageModel(model_id="gpt-5.4", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


def responses_model(handler) -> OpenAIResponsesLanguageModel:
    model = OpenAIResponsesLanguageModel(model_id="gpt-5.4", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


def sse(events: list[tuple[str, dict]]) -> bytes:
    out = []
    for name, data in events:
        out.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode()


# --- Responses: reasoning replay request shape -------------------------------


def test_reasoning_replay_request_shape():
    """A ReasoningPart carrying openai item_id/encrypted_content replays as a
    reasoning input item BEFORE the assistant message's other items."""
    messages = [
        UserModelMessage(content="hi"),
        AssistantModelMessage(
            content=[
                ReasoningPart(
                    text="thinking",
                    provider_options={
                        "openai": {
                            "item_id": "rs_1",
                            "encrypted_content": "enc==",
                        }
                    },
                ),
                TextPart(text="hello"),
                ToolCallPart(
                    tool_call_id="call_1",
                    tool_name="t",
                    input={"a": 1},
                ),
            ]
        ),
    ]
    items = convert_to_responses_input(messages)
    # user, reasoning, assistant text, function_call (in that order)
    assert items[1] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "thinking"}],
        "encrypted_content": "enc==",
    }
    # reasoning comes before the assistant's text and function_call items
    types = [it.get("type") or it.get("role") for it in items]
    assert types.index("reasoning") < types.index("assistant")
    assert types.index("reasoning") < types.index("function_call")


def test_reasoning_replay_skipped_without_ids():
    """No item_id and no encrypted_content → no reasoning input item."""
    messages = [
        AssistantModelMessage(
            content=[ReasoningPart(text="thinking"), TextPart(text="hi")]
        )
    ]
    items = convert_to_responses_input(messages)
    assert all(it.get("type") != "reasoning" for it in items)


def test_reasoning_replay_omits_encrypted_when_none():
    messages = [
        AssistantModelMessage(
            content=[
                ReasoningPart(
                    text="t", provider_options={"openai": {"item_id": "rs_9"}}
                )
            ]
        )
    ]
    items = convert_to_responses_input(messages)
    reasoning = [it for it in items if it.get("type") == "reasoning"][0]
    assert "encrypted_content" not in reasoning


# --- Responses: previous_response_id -----------------------------------------


async def test_previous_response_id_top_level():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_RESPONSES_TEXT)

    await generate_text(
        model=responses_model(handler),
        prompt="hi",
        provider_options={"openai": {"previous_response_id": "resp_prev"}},
    )
    body = requests[0]
    assert body["previous_response_id"] == "resp_prev"
    # not nested under extra_body
    assert "previous_response_id" not in body.get("extra_body", {})


# --- Responses: built-in web_search tool + annotations -----------------------


_RESPONSES_WEB_SEARCH = {
    "id": "resp_ws",
    "object": "response",
    "created_at": 1,
    "model": "gpt-5.4",
    "status": "completed",
    "output": [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "weather paris"},
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "It is sunny.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://example.com/paris",
                            "title": "Paris Weather",
                            "start_index": 0,
                            "end_index": 5,
                        }
                    ],
                }
            ],
        },
    ],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 0},
        "input_tokens_details": {"cached_tokens": 0},
    },
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


async def test_responses_web_search_and_annotations():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_RESPONSES_WEB_SEARCH)

    result = await generate_text(
        model=responses_model(handler),
        prompt="weather in paris?",
        provider_options={"openai": {"tools": [{"type": "web_search"}]}},
    )
    # built-in tool merged into request tools
    assert requests[0]["tools"] == [{"type": "web_search"}]

    # provider-executed tool call present in content
    builtin_calls = [
        p
        for p in result.content
        if isinstance(p, ToolCallPart) and p.provider_executed
    ]
    assert builtin_calls and builtin_calls[0].tool_name == "web_search"
    assert builtin_calls[0].tool_call_id == "ws_1"
    assert builtin_calls[0].input == {"type": "search", "query": "weather paris"}

    # url citation -> source
    assert result.sources
    src = result.sources[0]
    assert isinstance(src, UrlSourcePart)
    assert src.url == "https://example.com/paris"
    assert src.title == "Paris Weather"

    # text still present, finish reason is stop (provider-executed call does not
    # turn finish into tool-calls)
    assert result.text == "It is sunny."
    assert result.finish_reason == "stop"


def test_provider_executed_parts_skipped_on_replay():
    """Provider-executed tool call/result parts are server-side; not replayed
    as input items."""
    messages = [
        AssistantModelMessage(
            content=[
                ToolCallPart(
                    tool_call_id="ws_1",
                    tool_name="web_search",
                    input={"query": "x"},
                    provider_executed=True,
                ),
                ToolResultPart(
                    tool_call_id="ws_1",
                    tool_name="web_search",
                    output={"type": "json", "value": []},
                    provider_executed=True,
                ),
                TextPart(text="answer"),
            ]
        )
    ]
    items = convert_to_responses_input(messages)
    assert all(it.get("type") != "function_call" for it in items)
    # only the assistant text item is replayed
    assert any(it.get("role") == "assistant" for it in items)


# --- Responses: usage details + warnings + request echo + raw chunks ---------


_RESPONSES_TEXT = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-5.4",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Paris.", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 4},
        "input_tokens_details": {"cached_tokens": 6},
    },
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


async def test_responses_usage_details_warnings_request_echo():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_RESPONSES_TEXT)

    result = await generate_text(
        model=responses_model(handler),
        prompt="capital of france?",
        seed=42,  # unsupported by Responses → warning
        top_k=10,  # unsupported → warning
    )
    usage = result.usage
    assert usage.input_token_details.cache_read_tokens == 6
    assert usage.input_token_details.no_cache_tokens == 14
    assert usage.output_token_details.reasoning_tokens == 4
    assert usage.output_token_details.text_tokens == 6

    settings = {w.setting for w in result.warnings}
    assert "seed" in settings and "top_k" in settings
    assert all(isinstance(w, CallWarning) for w in result.warnings)
    assert all(w.type == "unsupported-setting" for w in result.warnings)

    # request echo, minus headers
    assert result.request["model"] == "gpt-5.4"
    assert "extra_headers" not in result.request


async def test_responses_raw_chunks_streaming():
    sse_events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_s",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.4",
                    "status": "in_progress",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi",
                "logprobs": [],
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "id": "resp_s",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "total_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(
        model=responses_model(handler),
        prompt="hi",
        include_raw_chunks=True,
    )
    parts = [p async for p in result.full_stream]
    raws = [p for p in parts if p.type == "raw"]
    assert raws, "expected RawPart parts when include_raw_chunks=True"
    assert await result.text == "Hi"


async def test_responses_streaming_web_search_and_source():
    sse_events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_s",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.4",
                    "status": "in_progress",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "in_progress",
                    "action": {"type": "search", "query": "q"},
                },
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "q"},
                },
            },
        ),
        (
            "response.output_text.annotation.added",
            {
                "type": "response.output_text.annotation.added",
                "sequence_number": 3,
                "item_id": "msg_1",
                "output_index": 1,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 1,
                },
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_s",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "total_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(model=responses_model(handler), prompt="search")
    parts = [p async for p in result.full_stream]

    builtin = [
        p
        for p in parts
        if p.type == "tool-call" and getattr(p, "provider_executed", None)
    ]
    assert builtin and builtin[0].tool_name == "web_search"

    sources = [p for p in parts if p.type == "source"]
    assert sources and sources[0].source.url == "https://example.com"


# --- Responses: FileIdData mappings ------------------------------------------


def test_responses_file_id_mappings():
    msg = UserModelMessage(
        content=[
            FilePart(
                data=FileIdData(id="file_123"),
                media_type="application/pdf",
                filename="doc.pdf",
            ),
            ImagePart(image=FileIdData(id="file_img")),
        ]
    )
    items = convert_to_responses_input([msg])
    content = items[0]["content"]
    assert content[0] == {
        "type": "input_file",
        "file_id": "file_123",
        "filename": "doc.pdf",
    }
    assert content[1] == {"type": "input_image", "file_id": "file_img"}


# --- Chat: usage details -----------------------------------------------------


_CHAT_TEXT = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 4},
        "prompt_tokens_details": {"cached_tokens": 6},
    },
}


async def test_chat_usage_details_and_request_echo():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_CHAT_TEXT)

    result = await generate_text(model=chat_model(handler), prompt="hi")
    usage = result.usage
    assert usage.input_token_details.cache_read_tokens == 6
    assert usage.input_token_details.no_cache_tokens == 14
    assert usage.output_token_details.reasoning_tokens == 4
    assert usage.output_token_details.text_tokens == 6
    assert result.request["model"] == "gpt-5.4"
    assert "extra_headers" not in result.request


async def test_chat_top_k_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "top_k" not in body  # top_k is never sent on Chat Completions
        return httpx.Response(200, json=_CHAT_TEXT)

    result = await generate_text(model=chat_model(handler), prompt="hi", top_k=5)
    settings = {w.setting for w in result.warnings}
    assert "top_k" in settings
    assert all(isinstance(w, CallWarning) for w in result.warnings)


# --- Chat: annotations -> sources --------------------------------------------


_CHAT_WITH_ANNOTATIONS = {
    "id": "chatcmpl-a",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "See here.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://example.org",
                            "title": "Example Org",
                            "start_index": 0,
                            "end_index": 4,
                        },
                    }
                ],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


async def test_chat_annotations_to_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_CHAT_WITH_ANNOTATIONS)

    result = await generate_text(model=chat_model(handler), prompt="hi")
    assert result.sources
    src = result.sources[0]
    assert isinstance(src, UrlSourcePart)
    assert src.url == "https://example.org"
    assert src.title == "Example Org"


# --- Chat: FileIdData mappings -----------------------------------------------


def test_chat_file_id_mapping():
    msg = UserModelMessage(
        content=[
            FilePart(data=FileIdData(id="file_99"), media_type="application/pdf")
        ]
    )
    converted = convert_to_chat_messages([msg])
    assert converted[0]["content"][0] == {
        "type": "file",
        "file": {"file_id": "file_99"},
    }


def test_chat_image_file_id_raises():
    msg = UserModelMessage(content=[ImagePart(image=FileIdData(id="file_img"))])
    with pytest.raises(ValueError, match="file id"):
        convert_to_chat_messages([msg])


async def test_chat_raw_chunks_streaming():
    chunks = [
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.4",
            "choices": [
                {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(
        model=chat_model(handler), prompt="hi", include_raw_chunks=True
    )
    parts = [p async for p in result.full_stream]
    assert [p for p in parts if p.type == "raw"]
    assert await result.text == "Hi"
