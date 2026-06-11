"""Extra-feature tests for the OpenRouter adapter (provider metadata, usage
cache-write, reasoning details). Runs through the real openai SDK with a
mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from model_message import generate_text
from model_message.messages import ReasoningPart
from model_message.providers.openrouter import OpenRouterLanguageModel

openai_sdk = pytest.importorskip("openai")


def openrouter_model(handler) -> OpenRouterLanguageModel:
    model = OpenRouterLanguageModel(model_id="anthropic/claude-opus-4.6", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


_OPENROUTER_RESPONSE = {
    "id": "gen-1",
    "object": "chat.completion",
    "created": 1,
    "model": "anthropic/claude-opus-4.6",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there.",
                "reasoning": "Let me think.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Let me think."}
                ],
            },
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
        }
    ],
    "usage": {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
        "cost": 0.00123,
        "is_byok": False,
        "cost_details": {
            "upstream_inference_cost": 0.0005,
            "cache_discount": 0.0001,
        },
        "prompt_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 12},
    },
}


async def test_openrouter_provider_metadata_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_OPENROUTER_RESPONSE)

    result = await generate_text(model=openrouter_model(handler), prompt="hi")

    meta = result.provider_metadata["openrouter"]
    assert meta["cost"] == 0.00123
    assert meta["is_byok"] is False
    assert meta["cost_details"]["upstream_inference_cost"] == 0.0005
    assert meta["cost_details"]["cache_discount"] == 0.0001
    assert meta["native_finish_reason"] == "STOP"

    # cache read + write tokens mapped into input_token_details
    details = result.usage.input_token_details
    assert details.cache_read_tokens == 8
    assert details.cache_write_tokens == 12
    assert details.no_cache_tokens == 22  # 30 - 8

    # reasoning_details preserved on the reasoning part
    reasoning = [p for p in result.content if isinstance(p, ReasoningPart)]
    assert reasoning
    ro = (reasoning[0].provider_options or {}).get("openrouter") or {}
    assert ro["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Let me think."}
    ]


_OPENROUTER_PLAIN = {
    "id": "gen-2",
    "object": "chat.completion",
    "created": 1,
    "model": "anthropic/claude-opus-4.6",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


async def test_openrouter_no_metadata_when_absent():
    """Without OpenRouter extensions, provider_metadata stays None and usage
    has no extra details."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_OPENROUTER_PLAIN)

    result = await generate_text(model=openrouter_model(handler), prompt="hi")
    assert result.provider_metadata is None
    assert result.usage.input_token_details is None


async def test_openrouter_provider_options_flow_through():
    """OpenRouter-specific request params land in the body via extra_body."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_OPENROUTER_PLAIN)

    await generate_text(
        model=openrouter_model(handler),
        prompt="hi",
        provider_options={
            "openrouter": {"provider": {"order": ["anthropic"]}},
        },
    )
    body = requests[0]
    assert body["provider"] == {"order": ["anthropic"]}
