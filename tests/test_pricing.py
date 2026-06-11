import pytest

from pai_sdk.pricing import (
    CostEstimate,
    ModelPricing,
    NoPricingError,
    estimate_cost,
    get_pricing,
    register_pricing,
)
from pai_sdk.results import InputTokenDetails, OutputTokenDetails, Usage


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


# ---------------------------------------------------------------------------
# Server-hosted pricing sources
# ---------------------------------------------------------------------------

from pai_sdk.pricing import parse_pricing_data  # noqa: E402


def test_parse_litellm_format():
    parsed = parse_pricing_data(
        {
            "claude-haiku-4-5": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 5e-06,
                "cache_read_input_token_cost": 1e-07,
                "cache_creation_input_token_cost": 1.25e-06,
            },
            "gemini/gemini-2.5-flash": {
                "input_cost_per_token": 3e-07,
                "output_cost_per_token": 2.5e-06,
            },
            "sample_spec": {"notes": "not a model"},
        },
        "litellm",
    )
    haiku = parsed["claude-haiku-4-5"]
    assert haiku.input == pytest.approx(1.0)
    assert haiku.output == pytest.approx(5.0)
    assert haiku.cache_read == pytest.approx(0.1)
    assert haiku.cache_write == pytest.approx(1.25)
    # provider-prefixed key registered with a bare alias
    assert parsed["gemini-2.5-flash"].input == pytest.approx(0.30)
    assert "sample_spec" not in parsed


def test_parse_openrouter_format():
    parsed = parse_pricing_data(
        {
            "data": [
                {
                    "id": "anthropic/claude-haiku-4.5",
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000005",
                        "input_cache_read": "0.0000001",
                        "input_cache_write": "0.00000125",
                        "web_search": "0.01",
                    },
                },
                {"id": "broken/no-pricing", "pricing": {}},
            ]
        },
        "openrouter",
    )
    slug = parsed["anthropic/claude-haiku-4.5"]
    assert slug.input == pytest.approx(1.0)
    assert slug.cache_write == pytest.approx(1.25)
    assert parsed["claude-haiku-4.5"].output == pytest.approx(5.0)  # bare alias
    assert "broken/no-pricing" not in parsed


def test_parse_models_dev_and_simple_formats():
    parsed = parse_pricing_data(
        {
            "anthropic": {
                "models": {
                    "claude-haiku-4-5": {
                        "cost": {"input": 1.0, "output": 5.0, "cache_read": 0.1}
                    }
                }
            }
        },
        "models.dev",
    )
    assert parsed["claude-haiku-4-5"].cache_read == pytest.approx(0.1)

    parsed = parse_pricing_data(
        {"acme/my-model": {"input": 2.0, "output": 8.0, "cache_write": 2.5}},
        "simple",
    )
    assert parsed["acme/my-model"].cache_write == pytest.approx(2.5)
    assert parsed["my-model"].input == pytest.approx(2.0)


def test_parse_unknown_format_rejected():
    with pytest.raises(ValueError):
        parse_pricing_data({}, "nope")


@pytest.mark.live
@pytest.mark.parametrize("source", ["litellm", "openrouter"])
async def test_refresh_pricing_live(source):
    import httpx

    from pai_sdk.pricing import get_pricing, refresh_pricing

    try:
        count = await refresh_pricing(source)
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"pricing source unreachable: {exc}")
    assert count > 100
    # a model we use must now be priced from the live source
    assert get_pricing("gpt-5.4-mini") is not None
    assert get_pricing("claude-haiku-4-5") is not None
