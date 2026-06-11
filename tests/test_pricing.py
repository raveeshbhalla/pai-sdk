import pytest

from model_message.pricing import (
    CostEstimate,
    ModelPricing,
    NoPricingError,
    estimate_cost,
    get_pricing,
    register_pricing,
)
from model_message.results import InputTokenDetails, OutputTokenDetails, Usage


def test_basic_estimate_with_details():
    usage = Usage(
        input_tokens=1000,
        output_tokens=500,
        input_token_details=InputTokenDetails(
            no_cache_tokens=400, cache_read_tokens=500, cache_write_tokens=100
        ),
        output_token_details=OutputTokenDetails(text_tokens=300, reasoning_tokens=200),
    )
    pricing = ModelPricing(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
    estimate = estimate_cost(usage, pricing=pricing)
    assert estimate.uncached_input_tokens == 400
    assert estimate.cache_read_tokens == 500
    assert estimate.cache_write_tokens == 100
    assert estimate.output_tokens == 500  # text + reasoning
    assert estimate.input_cost == pytest.approx(400 / 1e6 * 1.0)
    assert estimate.cache_read_cost == pytest.approx(500 / 1e6 * 0.1)
    assert estimate.cache_write_cost == pytest.approx(100 / 1e6 * 1.25)
    assert estimate.output_cost == pytest.approx(500 / 1e6 * 5.0)
    assert estimate.total == pytest.approx(
        estimate.input_cost
        + estimate.cache_read_cost
        + estimate.cache_write_cost
        + estimate.output_cost
    )


def test_fallback_without_details():
    usage = Usage(input_tokens=1000, output_tokens=200, cached_input_tokens=300)
    pricing = ModelPricing(input=2.0, output=10.0)  # no cache rates -> input rate
    estimate = estimate_cost(usage, pricing=pricing)
    assert estimate.uncached_input_tokens == 1000
    assert estimate.cache_read_tokens == 300
    assert estimate.cache_read_cost == pytest.approx(300 / 1e6 * 2.0)
    assert estimate.output_tokens == 200


def test_gemini_style_reasoning_excluded_from_output_is_billed():
    """Gemini's output_tokens excludes thoughts; details carry both."""
    usage = Usage(
        input_tokens=100,
        output_tokens=50,  # candidates only
        output_token_details=OutputTokenDetails(text_tokens=50, reasoning_tokens=400),
    )
    estimate = estimate_cost(usage, pricing=ModelPricing(input=1.0, output=10.0))
    assert estimate.output_tokens == 450


def test_lookup_exact_substring_and_unknown():
    assert get_pricing("claude-haiku-4-5").input == 1.0
    # substring match: Bedrock-prefixed and date-suffixed ids resolve
    assert get_pricing("anthropic.claude-haiku-4-5").input == 1.0
    assert get_pricing("claude-haiku-4-5-20251001").input == 1.0
    # longest match wins: gpt-5.4-mini must not match the gpt-5.4 entry
    assert get_pricing("gpt-5.4-mini").input == 0.75
    assert get_pricing("totally-unknown-model") is None
    with pytest.raises(NoPricingError):
        estimate_cost(Usage(input_tokens=1), model="totally-unknown-model")


def test_register_and_model_object_lookup():
    register_pricing("my-custom-model", ModelPricing(input=1.0, output=2.0))
    estimate = estimate_cost(
        Usage(input_tokens=1_000_000, output_tokens=1_000_000), model="my-custom-model"
    )
    assert estimate.total == pytest.approx(3.0)

    from conftest import FakeModel

    register_pricing("fake-1", ModelPricing(input=1.0, output=1.0))
    estimate = estimate_cost(Usage(input_tokens=500_000), model=FakeModel())
    assert estimate.input_cost == pytest.approx(0.5)


def test_estimates_sum():
    pricing = ModelPricing(input=1.0, output=2.0)
    a = estimate_cost(Usage(input_tokens=1_000_000, output_tokens=0), pricing=pricing)
    b = estimate_cost(Usage(input_tokens=0, output_tokens=1_000_000), pricing=pricing)
    combined = a + b
    assert combined.total == pytest.approx(3.0)
    assert combined.uncached_input_tokens == 1_000_000
    assert combined.output_tokens == 1_000_000


def test_currency_mismatch_rejected():
    a = estimate_cost(Usage(input_tokens=1), pricing=ModelPricing(1.0, 1.0))
    b = estimate_cost(
        Usage(input_tokens=1), pricing=ModelPricing(1.0, 1.0, currency="EUR")
    )
    with pytest.raises(ValueError):
        a + b


def test_requires_model_or_pricing():
    with pytest.raises(ValueError):
        estimate_cost(Usage(input_tokens=1))
