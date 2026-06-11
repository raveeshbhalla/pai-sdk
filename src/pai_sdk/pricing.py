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


# ---------------------------------------------------------------------------
# Server-hosted pricing sources
# ---------------------------------------------------------------------------

# Well-known public sources (no auth required):
PRICING_SOURCES = {
    # LiteLLM's community-maintained table — the de-facto standard; covers
    # nearly every model with input/output/cache rates (per token).
    "litellm": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
    # OpenRouter's public models API — per-token string prices per slug.
    "openrouter": "https://openrouter.ai/api/v1/models",
    # models.dev open catalog — per-1M prices nested per provider.
    "models.dev": "https://models.dev/api.json",
}


def _per_token(value) -> Optional[float]:
    """Per-token price (number or string) -> per-1M-tokens float."""
    if value is None:
        return None
    try:
        return float(value) * _MILLION
    except (TypeError, ValueError):
        return None


def _register_with_bare_alias(
    table: dict[str, ModelPricing], key: str, pricing: ModelPricing
) -> None:
    """Register under the full key; also alias the bare model id (the part
    after the last '/') without clobbering an existing exact entry."""
    table[key] = pricing
    if "/" in key:
        table.setdefault(key.rsplit("/", 1)[1], pricing)


def parse_pricing_data(data, format: str) -> dict[str, ModelPricing]:
    """Parse a pricing payload into {model_id: ModelPricing}.

    Formats:
    - "litellm": {model_id: {input_cost_per_token, output_cost_per_token,
      cache_read_input_token_cost?, cache_creation_input_token_cost?}}
    - "openrouter": {"data": [{"id": slug, "pricing": {"prompt", "completion",
      "input_cache_read"?, "input_cache_write"?}}]} (per-token strings)
    - "models.dev": {provider: {"models": {id: {"cost": {input, output,
      cache_read?, cache_write?}}}}} (per 1M tokens)
    - "simple": {model_id: {"input": per_1M, "output": per_1M,
      "cache_read"?: per_1M, "cache_write"?: per_1M}} — the schema to use
      for your own hosted table.
    """
    parsed: dict[str, ModelPricing] = {}

    if format == "litellm":
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            input_rate = _per_token(entry.get("input_cost_per_token"))
            output_rate = _per_token(entry.get("output_cost_per_token"))
            if input_rate is None or output_rate is None:
                continue
            _register_with_bare_alias(
                parsed,
                key,
                ModelPricing(
                    input=input_rate,
                    output=output_rate,
                    cache_read=_per_token(entry.get("cache_read_input_token_cost")),
                    cache_write=_per_token(entry.get("cache_creation_input_token_cost")),
                ),
            )

    elif format == "openrouter":
        for entry in data.get("data", []):
            slug = entry.get("id")
            rates = entry.get("pricing") or {}
            input_rate = _per_token(rates.get("prompt"))
            output_rate = _per_token(rates.get("completion"))
            if not slug or input_rate is None or output_rate is None:
                continue
            _register_with_bare_alias(
                parsed,
                slug,
                ModelPricing(
                    input=input_rate,
                    output=output_rate,
                    cache_read=_per_token(rates.get("input_cache_read")),
                    cache_write=_per_token(rates.get("input_cache_write")),
                ),
            )

    elif format == "models.dev":
        for provider_entry in data.values():
            models = (provider_entry or {}).get("models")
            if not isinstance(models, dict):
                continue
            for model_id, model_entry in models.items():
                cost = (model_entry or {}).get("cost") or {}
                if cost.get("input") is None or cost.get("output") is None:
                    continue
                parsed.setdefault(
                    model_id,
                    ModelPricing(
                        input=float(cost["input"]),
                        output=float(cost["output"]),
                        cache_read=(
                            float(cost["cache_read"])
                            if cost.get("cache_read") is not None
                            else None
                        ),
                        cache_write=(
                            float(cost["cache_write"])
                            if cost.get("cache_write") is not None
                            else None
                        ),
                    ),
                )

    elif format == "simple":
        for model_id, entry in data.items():
            if not isinstance(entry, dict) or "input" not in entry or "output" not in entry:
                continue
            _register_with_bare_alias(
                parsed,
                model_id,
                ModelPricing(
                    input=float(entry["input"]),
                    output=float(entry["output"]),
                    cache_read=(
                        float(entry["cache_read"])
                        if entry.get("cache_read") is not None
                        else None
                    ),
                    cache_write=(
                        float(entry["cache_write"])
                        if entry.get("cache_write") is not None
                        else None
                    ),
                ),
            )

    else:
        raise ValueError(
            f"Unknown pricing format '{format}'. "
            "Expected one of: litellm, openrouter, models.dev, simple."
        )

    return parsed


async def refresh_pricing(
    source: str = "litellm",
    *,
    format: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> int:
    """Fetch a server-hosted pricing table and merge it into the registry
    (fetched entries override built-ins; built-ins remain for models the
    source lacks). Returns the number of models loaded.

        await refresh_pricing()                       # LiteLLM community table
        await refresh_pricing("openrouter")           # OpenRouter models API
        await refresh_pricing("https://prices.my.co/models.json",
                              format="simple")        # your own hosted table

    `source` is a well-known name (litellm / openrouter / models.dev) or any
    URL; `format` defaults to the well-known name, and is required ("simple",
    or one of the named formats) for custom URLs.
    """
    import httpx

    url = PRICING_SOURCES.get(source, source)
    resolved_format = format or (source if source in PRICING_SOURCES else None)
    if resolved_format is None:
        raise ValueError(
            "Pass format= ('simple', 'litellm', 'openrouter', or 'models.dev') "
            "when refreshing from a custom URL."
        )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

    parsed = parse_pricing_data(data, resolved_format)
    _registry.update(parsed)
    return len(parsed)


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
