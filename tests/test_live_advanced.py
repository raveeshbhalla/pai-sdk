"""Live tests for server-validated round-trips: tool loops, reasoning replay,
trace dump->load->continue, gnarly structured-output schemas, error mapping.

These exercise the request shapes providers validate server-side (tool-call
IDs in replayed history, Anthropic thinking signatures, OpenAI encrypted
reasoning items) — the paths mock tests cannot prove.
"""

from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from model_message import (
    APICallError,
    ReasoningPart,
    generate_object,
    generate_text,
    step_count_is,
    stream_text,
    tool,
)
from model_message.providers import anthropic, google, openai
from model_message.serialize import dump_messages_json, load_messages

from test_live import MAX_TOKENS, MODELS, _skip_unless

pytestmark = pytest.mark.live


def weather_toolset(log: list):
    def execute(input):
        city = (input or {}).get("city", "unknown")
        log.append(city)
        return f"It is 72F and sunny in {city}."

    return {
        "get_weather": tool(
            description="Get the current weather for a city.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            execute=execute,
        )
    }


# ---------------------------------------------------------------------------
# 1. Tool loop, including parallel calls (server validates replayed history)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_tool_loop_parallel_calls(make_model):
    calls: list = []
    result = await generate_text(
        model=make_model(),
        prompt="What's the weather in Paris and in London? "
        "Use the get_weather tool for each city, then summarize both.",
        tools=weather_toolset(calls),
        stop_when=step_count_is(4),
        max_output_tokens=MAX_TOKENS,
    )
    assert len(calls) >= 2
    assert {c.lower() for c in calls} >= {"paris", "london"}
    assert len(result.steps) >= 2
    assert result.steps[0].finish_reason == "tool-calls"
    assert result.finish_reason == "stop"
    assert "72" in result.text
    executed = [r for step in result.steps for r in step.tool_results]
    assert len(executed) >= 2
    assert all(not r.is_error for r in executed)


# ---------------------------------------------------------------------------
# 2. Streaming tool loop (tool input deltas -> execution -> next step, live)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_streaming_tool_loop(make_model):
    calls: list = []
    result = stream_text(
        model=make_model(),
        prompt="What's the weather in Tokyo? Use the get_weather tool.",
        tools=weather_toolset(calls),
        stop_when=step_count_is(3),
        max_output_tokens=MAX_TOKENS,
    )
    types = [part.type async for part in result.full_stream]
    assert "tool-call" in types
    assert "tool-result" in types
    assert types.count("finish-step") >= 2
    assert calls and "tokyo" in calls[0].lower()
    assert "72" in (await result.text)
    assert await result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3. Reasoning replay — server-enforced signatures / encrypted items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_model,provider_options,reasoning_metadata_key",
    [
        pytest.param(
            lambda: anthropic("claude-sonnet-4-6"),
            {"anthropic": {"thinking": {"type": "adaptive"}}},
            "signature",
            id="anthropic-signed-thinking",
            marks=_skip_unless("ANTHROPIC_API_KEY"),
        ),
        pytest.param(
            lambda: openai("gpt-5.4-mini"),
            {
                "openai": {
                    "reasoning": {"effort": "medium", "summary": "auto"},
                    "store": False,
                    "include": ["reasoning.encrypted_content"],
                }
            },
            "encrypted_content",
            id="openai-encrypted-reasoning",
            marks=_skip_unless("OPENAI_API_KEY"),
        ),
    ],
)
async def test_reasoning_replay_through_tool_loop(
    make_model, provider_options, reasoning_metadata_key
):
    """Thinking/reasoning + a tool call forces the loop's second request to
    replay reasoning blocks. Anthropic validates the thinking signature;
    OpenAI validates the encrypted reasoning item. A 200 on step 2 IS the
    assertion that our capture/replay is correct.

    Reasoning *presence* is model-discretionary (the model may decide a step
    needs no reasoning and emit none), so the part-level assertions are
    conditional on reasoning tokens having been spent — but when reasoning
    parts exist they must carry the replay metadata, or step 2 would have
    been rejected.
    """
    calls: list = []
    result = await generate_text(
        model=make_model(),
        prompt="What's the weather in Berlin? Think about which tool to use, "
        "then use the get_weather tool.",
        tools=weather_toolset(calls),
        stop_when=step_count_is(3),
        max_output_tokens=8000,
        provider_options=provider_options,
    )
    assert calls and "berlin" in calls[0].lower()
    assert len(result.steps) >= 2  # the replay request succeeded
    assert result.finish_reason == "stop"
    assert "72" in result.text

    reasoning_parts = [
        p for step in result.steps for p in step.content if isinstance(p, ReasoningPart)
    ]
    spent_reasoning_tokens = any(
        (step.usage.reasoning_tokens or 0) > 0 for step in result.steps
    )
    if spent_reasoning_tokens:
        assert reasoning_parts, "reasoning tokens spent but no ReasoningPart parsed"
    for part in reasoning_parts:
        provider_meta = next(iter((part.provider_options or {}).values()), {})
        assert provider_meta.get(reasoning_metadata_key), (
            f"reasoning part missing replay metadata '{reasoning_metadata_key}'"
        )


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(
            lambda: google("gemini-2.5-flash"),
            id="google-thoughts",
            marks=_skip_unless("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        )
    ],
)
async def test_gemini_thought_summaries(make_model):
    result = await generate_text(
        model=make_model(),
        prompt="In one sentence: why is the sky blue?",
        max_output_tokens=8000,
        provider_options={
            "google": {"thinking_config": {"include_thoughts": True, "thinking_budget": 512}}
        },
    )
    assert result.text
    assert result.reasoning, "expected thought-summary ReasoningParts"
    assert (result.usage.reasoning_tokens or 0) > 0


# ---------------------------------------------------------------------------
# 4. The log -> replay contract: dump a real tool conversation to JSON, load
#    it back, continue the conversation (server validates the replayed IDs).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_dump_load_continue(make_model):
    model = make_model()
    calls: list = []
    tools = weather_toolset(calls)
    first_user = {"role": "user", "content": "What's the weather in Oslo? Use the tool."}

    first = await generate_text(
        model=model,
        messages=[first_user],
        tools=tools,
        stop_when=step_count_is(3),
        max_output_tokens=MAX_TOKENS,
    )
    assert calls and len(first.steps) >= 2

    # full trace -> JSON -> back, then continue on the restored history
    trace = dump_messages_json([first_user, *first.response.messages])
    history = load_messages(trace)
    followup = await generate_text(
        model=model,
        messages=[
            *history,
            {
                "role": "user",
                "content": "What temperature did the tool report? Reply with digits only.",
            },
        ],
        tools=tools,
        max_output_tokens=MAX_TOKENS,
    )
    assert "72" in followup.text


# ---------------------------------------------------------------------------
# 5. Gnarly structured-output schema (nested, lists, optionals, enums)
# ---------------------------------------------------------------------------


class Address(BaseModel):
    city: str
    country: str


class Person(BaseModel):
    name: str
    role: Literal["engineer", "designer", "manager"]
    age: Optional[int] = None
    addresses: list[Address]
    nickname: Optional[str] = None


@pytest.mark.parametrize("make_model", MODELS)
async def test_structured_output_nested_schema(make_model):
    result = await generate_object(
        model=make_model(),
        schema=Person,
        prompt=(
            "Extract the person: Maria Santos, 34, is an engineer known as 'Mia'. "
            "She lives in Lisbon, Portugal and keeps an apartment in Porto, Portugal."
        ),
        max_output_tokens=MAX_TOKENS,
    )
    person = result.object
    assert isinstance(person, Person)
    assert person.role == "engineer"
    assert person.age == 34
    cities = {a.city.lower() for a in person.addresses}
    assert "lisbon" in cities
    assert all(a.country.lower() == "portugal" for a in person.addresses)


# ---------------------------------------------------------------------------
# 6. Error & edge mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_invalid_model_raises_api_call_error(make_model):
    real = make_model()
    bad = type(real)(**{**real.__dict__, "model_id": "definitely-not-a-model-xyz"})
    bad._client_cache = None
    with pytest.raises(APICallError) as err:
        await generate_text(model=bad, prompt="hi", max_retries=0)
    assert err.value.status_code in (400, 404)


@pytest.mark.parametrize("make_model", MODELS)
async def test_truncation_maps_to_length(make_model):
    result = await generate_text(
        model=make_model(),
        prompt="Write a 1000-word essay about the ocean.",
        max_output_tokens=32,
    )
    assert result.finish_reason == "length"
