"""End-to-end tests through the real provider SDKs with mocked HTTP.

These run the full pipeline: generate_text/stream_text -> adapter ->
provider SDK -> (mock) HTTP -> response parsing -> results.
"""

from __future__ import annotations

import json

import httpx
import pytest

from model_message import generate_text, step_count_is, stream_text, tool
from model_message.providers.anthropic import AnthropicLanguageModel
from model_message.providers.openai_chat import OpenAIChatLanguageModel
from model_message.providers.openai_responses import OpenAIResponsesLanguageModel

anthropic_sdk = pytest.importorskip("anthropic")
openai_sdk = pytest.importorskip("openai")


def anthropic_model(handler) -> AnthropicLanguageModel:
    model = AnthropicLanguageModel(model_id="claude-opus-4-8", api_key="test")
    model._client_cache = anthropic_sdk.AsyncAnthropic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


def openai_chat_model(handler) -> OpenAIChatLanguageModel:
    model = OpenAIChatLanguageModel(model_id="gpt-5.4", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


def openai_responses_model(handler) -> OpenAIResponsesLanguageModel:
    model = OpenAIResponsesLanguageModel(model_id="gpt-5.4", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


# --- Anthropic ----------------------------------------------------------------


ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 20, "output_tokens": 10},
}

ANTHROPIC_TEXT_RESPONSE = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [{"type": "text", "text": "It is 72F in Paris."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 40, "output_tokens": 12},
}


async def test_anthropic_tool_loop_e2e():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        payload = (
            ANTHROPIC_TOOL_RESPONSE if len(requests) == 1 else ANTHROPIC_TEXT_RESPONSE
        )
        return httpx.Response(200, json=payload)

    result = await generate_text(
        model=anthropic_model(handler),
        system="be helpful",
        prompt="Weather in Paris?",
        tools={
            "get_weather": tool(
                description="get weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                execute=lambda input: f"72F in {input['city']}",
            )
        },
        stop_when=step_count_is(3),
        max_output_tokens=500,
    )

    assert result.text == "It is 72F in Paris."
    assert result.finish_reason == "stop"
    assert result.total_usage.output_tokens == 22
    assert len(result.steps) == 2

    # wire format of the first request
    first = requests[0]
    assert first["system"] == "be helpful"
    assert first["max_tokens"] == 500
    assert first["tools"][0]["name"] == "get_weather"
    # second request must carry the tool round-trip
    second = requests[1]
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][1]["type"] == "tool_use"
    tool_result = second["messages"][2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "72F in Paris",
    }


def sse(events: list[tuple[str, dict]]) -> bytes:
    out = []
    for name, data in events:
        out.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode()


ANTHROPIC_SSE = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_s1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 9, "output_tokens": 1},
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
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 12},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


async def test_anthropic_streaming_e2e():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(ANTHROPIC_SSE),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(model=anthropic_model(handler), prompt="hi")
    chunks = [c async for c in result.text_stream]
    assert "".join(chunks) == "Hello world"
    assert chunks == ["Hello", " world"]
    assert await result.finish_reason == "stop"
    usage = await result.usage
    assert usage.input_tokens == 9
    assert usage.output_tokens == 12


# --- OpenAI Chat Completions ----------------------------------------------------


OPENAI_CHAT_TOOL_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 7,
        "total_tokens": 22,
        "completion_tokens_details": {"reasoning_tokens": 3},
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}

OPENAI_CHAT_TEXT_RESPONSE = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 2,
    "model": "gpt-5.4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "72F in Paris."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
}


async def test_openai_chat_tool_loop_e2e():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        payload = (
            OPENAI_CHAT_TOOL_RESPONSE if len(requests) == 1 else OPENAI_CHAT_TEXT_RESPONSE
        )
        return httpx.Response(200, json=payload)

    result = await generate_text(
        model=openai_chat_model(handler),
        system="be helpful",
        prompt="Weather in Paris?",
        tools={"get_weather": tool(execute=lambda input: f"72F in {input['city']}")},
        stop_when=step_count_is(3),
    )

    assert result.text == "72F in Paris."
    assert result.steps[0].usage.reasoning_tokens == 3
    assert result.steps[0].usage.cached_input_tokens == 4

    second = requests[1]
    assert second["messages"][0]["role"] == "system"
    assistant = second["messages"][2]
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    tool_msg = second["messages"][3]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "72F in Paris",
    }


def chat_chunks(chunks: list[dict]) -> bytes:
    out = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


OPENAI_CHAT_SSE = [
    {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
]


async def test_openai_chat_streaming_e2e():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            content=chat_chunks(OPENAI_CHAT_SSE),
            headers={"content-type": "text/event-stream"},
        )

    result = stream_text(model=openai_chat_model(handler), prompt="hi")
    assert await result.text == "Hello"
    assert (await result.usage).total_tokens == 7
    assert await result.finish_reason == "stop"


# --- OpenAI Responses -----------------------------------------------------------


OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-5.4",
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking about it."}],
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Paris.", "annotations": []}
            ],
        },
    ],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 2},
        "input_tokens_details": {"cached_tokens": 0},
    },
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


async def test_openai_responses_e2e():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=OPENAI_RESPONSES_RESPONSE)

    result = await generate_text(
        model=openai_responses_model(handler),
        system="be terse",
        prompt="Capital of France?",
        provider_options={"openai": {"reasoning": {"effort": "low"}}},
    )
    assert result.text == "Paris."
    assert result.reasoning_text == "Thinking about it."
    assert result.usage.reasoning_tokens == 2
    assert result.finish_reason == "stop"

    body = requests[0]
    assert body["instructions"] == "be terse"
    assert body["reasoning"] == {"effort": "low"}
    assert body["input"][0]["role"] == "user"
