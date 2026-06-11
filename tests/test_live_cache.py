"""Live prompt-cache usage-field tests + pricing integration.

Anthropic caching is explicit (cache_control breakpoint) and deterministic, so
those are hard assertions: the first call must report a cache WRITE and the
second an equal-prefix cache READ. OpenAI and Gemini cache implicitly — hits
are likely but not guaranteed, so those tests retry briefly and skip (not
fail) if the provider reports no hit.
"""

from __future__ import annotations

import asyncio

import pytest

from model_message import generate_text
from model_message.pricing import estimate_cost, get_pricing
from model_message.providers import anthropic, google, openai

from test_live import _skip_unless

pytestmark = pytest.mark.live

# A deterministic ~8K-token document (well above every model's minimum
# cacheable prefix; Haiku's minimum is 4096 tokens).
BIG_DOCUMENT = "\n".join(
    f"Clause {i}: the party of the first part shall deliver {i % 97} widgets "
    f"to the depot at location {i * 7 % 1009} no later than day {i % 365} of "
    "the contract year, subject to the inspection provisions herein."
    for i in range(900)
)

QUESTION = "Per the contract above, answer in one word: what items are delivered?"


@pytest.mark.parametrize(
    "noop", [pytest.param(None, marks=_skip_unless("ANTHROPIC_API_KEY"))]
)
async def test_anthropic_explicit_cache_write_then_read(noop):
    model = anthropic("claude-haiku-4-5")

    async def call():
        return await generate_text(
            model=model,
            system=BIG_DOCUMENT,
            prompt=QUESTION,
            max_output_tokens=200,
            # top-level cache_control auto-places a breakpoint on the last
            # cacheable block (the system prompt)
            provider_options={"anthropic": {"cache_control": {"type": "ephemeral"}}},
        )

    first = await call()
    details_first = first.usage.input_token_details
    assert details_first is not None
    assert (details_first.cache_write_tokens or 0) > 1000, "expected a cache write"

    second = await call()
    details_second = second.usage.input_token_details
    assert details_second is not None
    assert (second.usage.cached_input_tokens or 0) > 1000, "expected a cache read"
    assert (details_second.cache_read_tokens or 0) > 1000
    # the cached span is what the first call wrote
    assert details_second.cache_read_tokens >= (details_first.cache_write_tokens or 0)

    # pricing integration: cache accounting must flow into the estimate, and
    # the cached call must be cheaper than pricing the same tokens uncached
    pricing = get_pricing("claude-haiku-4-5")
    estimate = estimate_cost(second.usage, model=model)
    assert estimate.cache_read_cost > 0
    assert estimate.total > 0
    uncached_equivalent = (
        (estimate.uncached_input_tokens + estimate.cache_read_tokens)
        / 1_000_000
        * pricing.input
        + estimate.output_tokens / 1_000_000 * pricing.output
    )
    assert estimate.total < uncached_equivalent


async def _implicit_cache_second_call(make_call, detail_getter, attempts=3):
    """Call once to seed, then retry the second call until the provider
    reports a cache hit. Returns (usage, hit_tokens) or (usage, 0)."""
    await make_call()
    usage, hit = None, 0
    for _ in range(attempts):
        await asyncio.sleep(2)
        result = await make_call()
        usage = result.usage
        hit = detail_getter(usage) or 0
        if hit > 0:
            break
    return usage, hit


@pytest.mark.parametrize(
    "noop", [pytest.param(None, marks=_skip_unless("OPENAI_API_KEY"))]
)
async def test_openai_implicit_cache_read(noop):
    model = openai("gpt-5.4-mini")

    def call():
        return generate_text(
            model=model,
            system=BIG_DOCUMENT,
            prompt=QUESTION,
            max_output_tokens=2000,
        )

    usage, hit = await _implicit_cache_second_call(
        call, lambda u: (u.input_token_details and u.input_token_details.cache_read_tokens)
    )
    if hit == 0:
        pytest.skip("OpenAI reported no automatic cache hit (implicit cache miss)")
    assert hit > 1000
    assert usage.cached_input_tokens == hit
    estimate = estimate_cost(usage, model=model)
    assert estimate.cache_read_cost > 0
    assert estimate.cache_read_cost < estimate.cache_read_tokens / 1_000_000 * 0.75


@pytest.mark.parametrize(
    "noop", [pytest.param(None, marks=_skip_unless("GEMINI_API_KEY", "GOOGLE_API_KEY"))]
)
async def test_gemini_implicit_cache_read(noop):
    model = google("gemini-2.5-flash")

    def call():
        return generate_text(
            model=model,
            system=BIG_DOCUMENT,
            prompt=QUESTION,
            max_output_tokens=2000,
        )

    usage, hit = await _implicit_cache_second_call(
        call, lambda u: (u.input_token_details and u.input_token_details.cache_read_tokens)
    )
    if hit == 0:
        pytest.skip("Gemini reported no implicit cache hit")
    assert usage.cached_input_tokens == hit
    estimate = estimate_cost(usage, model=model)
    assert estimate.cache_read_cost > 0
    assert estimate.total > 0
