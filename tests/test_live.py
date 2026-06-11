"""Live end-to-end tests against real provider APIs.

Coverage matrix: {plain text input, structured message input} x
{text output, structured output}, plus the streaming variants of each output
mode — across Anthropic, OpenAI (Responses + Chat Completions), and Gemini.
Multimodal input and tool calls are intentionally out of scope here.

Keys load from .env.local (see conftest). Each provider's tests skip cleanly
when its key is absent. Deselect all live tests with: pytest -m "not live".
Cheap/small models keep cost negligible.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from model_message import Output, generate_object, generate_text, stream_object, stream_text
from model_message.providers import anthropic, google, openai

pytestmark = pytest.mark.live


def _skip_unless(*env_vars: str):
    present = any(os.environ.get(v) for v in env_vars)
    return pytest.mark.skipif(
        not present, reason=f"requires one of {env_vars} (set in .env.local)"
    )


# (model factory thunk, id) — thunks so model construction happens at test
# time, after conftest loaded .env.local.
MODELS = [
    pytest.param(
        lambda: anthropic("claude-haiku-4-5"),
        id="anthropic-messages",
        marks=_skip_unless("ANTHROPIC_API_KEY"),
    ),
    pytest.param(
        lambda: openai("gpt-5.4-mini"),
        id="openai-responses",
        marks=_skip_unless("OPENAI_API_KEY"),
    ),
    pytest.param(
        lambda: openai.chat("gpt-5.4-mini"),
        id="openai-chat",
        marks=_skip_unless("OPENAI_API_KEY"),
    ),
    pytest.param(
        lambda: google("gemini-2.5-flash"),
        id="google-gemini",
        marks=_skip_unless("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
]

MAX_TOKENS = 4000  # headroom for reasoning models so we never finish on length


class Contact(BaseModel):
    name: str
    email: str
    plan: str


# A multi-turn, multi-part conversation: system prompt, prior turns, and a
# list-of-parts user message — the "structured text input" case.
STRUCTURED_MESSAGES = [
    {"role": "user", "content": "My name is Alice and my favorite number is 7."},
    {"role": "assistant", "content": "Nice to meet you, Alice."},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is my favorite number? Reply with just the digit.",
            }
        ],
    },
]

EXTRACT_PROMPT = (
    "Extract the contact from this note: "
    "Jane Doe (jane@example.com) signed up for the Enterprise plan."
)


# ---------------------------------------------------------------------------
# 1. plain text input -> text output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_text_in_text_out(make_model):
    result = await generate_text(
        model=make_model(),
        prompt="Reply with exactly the word: pong",
        max_output_tokens=MAX_TOKENS,
    )
    assert "pong" in result.text.lower()
    assert result.finish_reason == "stop"
    assert (result.usage.input_tokens or 0) > 0
    assert (result.usage.output_tokens or 0) > 0
    assert result.response.messages[0].role == "assistant"


# ---------------------------------------------------------------------------
# 2. structured message input (system + history + parts) -> text output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_structured_in_text_out(make_model):
    result = await generate_text(
        model=make_model(),
        system="You answer in as few words as possible.",
        messages=STRUCTURED_MESSAGES,
        max_output_tokens=MAX_TOKENS,
    )
    assert "7" in result.text
    assert result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3. plain text input -> structured output (generate_object)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_text_in_structured_out(make_model):
    result = await generate_object(
        model=make_model(),
        schema=Contact,
        prompt=EXTRACT_PROMPT,
        max_output_tokens=MAX_TOKENS,
    )
    assert isinstance(result.object, Contact)
    assert result.object.email == "jane@example.com"
    assert "jane" in result.object.name.lower()
    assert "enterprise" in result.object.plan.lower()
    assert (result.usage.output_tokens or 0) > 0


# ---------------------------------------------------------------------------
# 4. structured message input -> structured output (output= on generate_text)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_structured_in_structured_out(make_model):
    class FavoriteNumber(BaseModel):
        person: str
        number: int

    result = await generate_text(
        model=make_model(),
        system="Extract the requested data.",
        messages=[
            *STRUCTURED_MESSAGES[:2],
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return my name and favorite number."}
                ],
            },
        ],
        output=Output.object(schema=FavoriteNumber),
        max_output_tokens=MAX_TOKENS,
    )
    assert isinstance(result.output, FavoriteNumber)
    assert result.output.number == 7
    assert "alice" in result.output.person.lower()


# ---------------------------------------------------------------------------
# 5. streaming text output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_text_in_stream_text_out(make_model):
    result = stream_text(
        model=make_model(),
        prompt="Count from 1 to 5 as digits separated by spaces.",
        max_output_tokens=MAX_TOKENS,
    )
    deltas = [d async for d in result.text_stream]
    final_text = await result.text
    assert "".join(deltas) == final_text
    for digit in "12345":
        assert digit in final_text
    assert await result.finish_reason == "stop"
    usage = await result.usage
    assert (usage.output_tokens or 0) > 0


# ---------------------------------------------------------------------------
# 6. streaming structured output (stream_object partials + final)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
async def test_text_in_stream_structured_out(make_model):
    result = stream_object(
        model=make_model(),
        schema=Contact,
        prompt=EXTRACT_PROMPT,
        max_output_tokens=MAX_TOKENS,
    )
    partials = [p async for p in result.partial_object_stream]
    final = await result.object
    assert isinstance(final, Contact)
    assert final.email == "jane@example.com"
    assert len(partials) >= 1
    assert all(isinstance(p, dict) for p in partials)
    # the last partial should already contain the final email
    assert partials[-1].get("email") == "jane@example.com"
