"""Cost estimation from Usage — pricing tables + estimate_cost().

Providers report token counts with different semantics (OpenAI's input count
includes cached tokens, Anthropic's excludes them; Gemini's output count
excludes reasoning tokens, OpenAI's includes them). Our adapters normalize all
of that into `Usage.input_token_details` / `Usage.output_token_details`, and
this module prices the normalized view:

    uncached input x input rate
  + cache reads    x cache_read rate
  + cache writes   x cache_write rate
  + (text + reasoning) output x output rate

Built-in prices are ESTIMATES (USD per 1M tokens, last reviewed on
PRICING_AS_OF) for common models — verify against provider pricing pages for
billing-grade numbers, and override or extend with register_pricing() /
estimate_cost(pricing=...). For OpenRouter, prefer the authoritative
`result.provider_metadata["openrouter"]["cost"]` the API returns.

Fallback when a Usage has no detail breakdowns (e.g. a custom provider):
input_tokens are billed at the input rate, cached_input_tokens at the
cache-read rate, output_tokens at the output rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .errors import AISDKError
from .provider import LanguageModel
from .results import Usage

PRICING_AS_OF = "2026-06-11"

_MILLION = 1_000_000


class NoPricingError(AISDKError):
    """No pricing registered for a model id."""

    def __init__(self, model_id: str, known: list[str]) -> None:
        super().__init__(
            f"No pricing registered for model '{model_id}'. Register one with "
            f"register_pricing('{model_id}', ModelPricing(input=..., output=...)) "
            f"or pass pricing= explicitly. Known models: {', '.join(sorted(known))}"
        )
        self.model_id = model_id


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens. cache_read/cache_write default to the input rate
    when None (a conservative no-discount assumption)."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    currency: str = "USD"


@dataclass
class CostEstimate:
    """A priced breakdown of one Usage. Costs in `currency` (USD by default)."""

    uncached_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int  # text + reasoning (billable output)
    input_cost: float
    cache_read_cost: float
    cache_write_cost: float
    output_cost: float
    total: float
    currency: str = "USD"
    pricing_as_of: str = PRICING_AS_OF

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        if self.currency != other.currency:
            raise ValueError("Cannot sum cost estimates in different currencies.")
        return CostEstimate(
            uncached_input_tokens=self.uncached_input_tokens + other.uncached_input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            input_cost=self.input_cost + other.input_cost,
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
            cache_write_cost=self.cache_write_cost + other.cache_write_cost,
            output_cost=self.output_cost + other.output_cost,
            total=self.total + other.total,
            currency=self.currency,
        )


# ---------------------------------------------------------------------------
# Built-in pricing (estimates — see module docstring)
# ---------------------------------------------------------------------------

# Anthropic: cache reads ~0.1x input, cache writes ~1.25x input (5m TTL).
# OpenAI: cached input ~0.1x input. Gemini: implicit cache reads ~0.25x input.
_BUILTIN: dict[str, ModelPricing] = {
    # Anthropic
    "claude-fable-5": ModelPricing(10.0, 50.0, cache_read=1.0, cache_write=12.5),
    "claude-opus-4-8": ModelPricing(5.0, 25.0, cache_read=0.5, cache_write=6.25),
    "claude-opus-4-7": ModelPricing(5.0, 25.0, cache_read=0.5, cache_write=6.25),
    "claude-opus-4-6": ModelPricing(5.0, 25.0, cache_read=0.5, cache_write=6.25),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, cache_read=0.3, cache_write=3.75),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0, cache_read=0.3, cache_write=3.75),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, cache_read=0.1, cache_write=1.25),
    # OpenAI
    "gpt-5.5-pro": ModelPricing(30.0, 180.0),
    "gpt-5.5": ModelPricing(5.0, 30.0, cache_read=0.5),
    "gpt-5.4-mini": ModelPricing(0.75, 4.5, cache_read=0.075),
    "gpt-5.4-nano": ModelPricing(0.15, 0.9, cache_read=0.015),
    "gpt-5.4": ModelPricing(2.5, 15.0, cache_read=0.25),
    # Google
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, cache_read=0.3125),
    "gemini-2.5-flash-lite": ModelPricing(0.10, 0.40, cache_read=0.025),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50, cache_read=0.075),
}

_registry: dict[str, ModelPricing] = dict(_BUILTIN)


def register_pricing(model_id: str, pricing: ModelPricing) -> None:
    """Register or override pricing for a model id (exact or substring key)."""
    _registry[model_id] = pricing


def get_pricing(model_id: str) -> Optional[ModelPricing]:
    """Look up pricing: exact id first, then the longest registered key that
    is a substring of the id (handles 'anthropic.claude-haiku-4-5' on Bedrock,
    'openrouter/...'-style slugs, dated snapshots)."""
    if model_id in _registry:
        return _registry[model_id]
    matches = [key for key in _registry if key in model_id]
    if not matches:
        return None
    return _registry[max(matches, key=len)]


def estimate_cost(
    usage: Usage,
    *,
    model: Union[str, LanguageModel, None] = None,
    pricing: Optional[ModelPricing] = None,
) -> CostEstimate:
    """Price a Usage. Pass `pricing` explicitly, or `model` (a LanguageModel
    or a model-id string) to look it up. For multi-step results, pass
    result.total_usage; sum estimates across calls with `+`."""
    if pricing is None:
        if model is None:
            raise ValueError("Provide either pricing= or model=.")
        model_id = model if isinstance(model, str) else model.model_id
        pricing = get_pricing(model_id)
        if pricing is None:
            raise NoPricingError(model_id, list(_registry))

    input_details = usage.input_token_details
    output_details = usage.output_token_details

    if input_details is not None and input_details.no_cache_tokens is not None:
        uncached = input_details.no_cache_tokens
    else:
        uncached = usage.input_tokens or 0
    cache_read = (
        (input_details.cache_read_tokens if input_details else None)
        or usage.cached_input_tokens
        or 0
    )
    cache_write = (input_details.cache_write_tokens if input_details else None) or 0

    if output_details is not None and (
        output_details.text_tokens is not None
        or output_details.reasoning_tokens is not None
    ):
        output = (output_details.text_tokens or 0) + (output_details.reasoning_tokens or 0)
    else:
        output = usage.output_tokens or 0

    cache_read_rate = pricing.cache_read if pricing.cache_read is not None else pricing.input
    cache_write_rate = (
        pricing.cache_write if pricing.cache_write is not None else pricing.input
    )

    input_cost = uncached / _MILLION * pricing.input
    cache_read_cost = cache_read / _MILLION * cache_read_rate
    cache_write_cost = cache_write / _MILLION * cache_write_rate
    output_cost = output / _MILLION * pricing.output

    return CostEstimate(
        uncached_input_tokens=uncached,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=output,
        input_cost=input_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        output_cost=output_cost,
        total=input_cost + cache_read_cost + cache_write_cost + output_cost,
        currency=pricing.currency,
    )
