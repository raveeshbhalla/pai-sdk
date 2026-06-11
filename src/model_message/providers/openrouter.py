"""OpenRouter provider — OpenAI Chat Completions-compatible, with extras.

Model ids are OpenRouter slugs like "anthropic/claude-opus-4.6" or
"openai/gpt-5.4". OpenRouter-specific params (provider routing, fallback
models, normalized reasoning, transforms, plugins) go in
provider_options={"openrouter": {...}} and are merged into the request body.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..results import InputTokenDetails
from .openai_chat import OpenAIChatLanguageModel

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Defensive attribute/key access for SDK models that carry OpenRouter
    extension fields (kept as extra attributes / dict entries)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class OpenRouterLanguageModel(OpenAIChatLanguageModel):
    provider: str = "openrouter"
    provider_options_keys: tuple[str, ...] = ("openai", "openrouter")
    api_key_env: str = "OPENROUTER_API_KEY"

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = DEFAULT_BASE_URL
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

    def _map_usage(self, usage: Any):
        """Augment the base usage mapping with OpenRouter's
        ``prompt_tokens_details.cache_write_tokens`` (cache-write reads land in
        Usage.input_token_details.cache_write_tokens)."""
        mapped = super()._map_usage(usage)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cache_write = _get(prompt_details, "cache_write_tokens")
        if cache_write is not None:
            details = mapped.input_token_details or InputTokenDetails()
            mapped.input_token_details = InputTokenDetails(
                no_cache_tokens=details.no_cache_tokens,
                cache_read_tokens=details.cache_read_tokens,
                cache_write_tokens=cache_write,
            )
        return mapped

    def _reasoning_part(self, message: Any):
        """Attach OpenRouter ``reasoning_details`` to the reasoning part when
        present (preserved for replay via provider_options)."""
        part = super()._reasoning_part(message)
        details = _get(message, "reasoning_details")
        if details is not None:
            from ..messages import ReasoningPart

            if part is None:
                part = ReasoningPart(text="")
            part.provider_options = {
                **(part.provider_options or {}),
                "openrouter": {"reasoning_details": details},
            }
        return part

    def _extract_provider_metadata(
        self, response: Any
    ) -> Optional[dict[str, dict[str, Any]]]:
        """Surface OpenRouter response extensions: usage cost details and the
        per-choice native finish reason. All fields are accessed defensively
        since they are OpenRouter additions on top of the OpenAI shape."""
        meta: dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        cost = _get(usage, "cost")
        if cost is not None:
            meta["cost"] = cost
        cost_details = _get(usage, "cost_details")
        if cost_details is not None:
            meta["cost_details"] = {
                "upstream_inference_cost": _get(cost_details, "upstream_inference_cost"),
                "cache_discount": _get(cost_details, "cache_discount"),
            }
        is_byok = _get(usage, "is_byok")
        if is_byok is not None:
            meta["is_byok"] = is_byok

        choices = getattr(response, "choices", None) or []
        if choices:
            native = _get(choices[0], "native_finish_reason")
            if native is not None:
                meta["native_finish_reason"] = native

        return {"openrouter": meta} if meta else None


def attribution_headers(
    app_url: Optional[str] = None, app_title: Optional[str] = None
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        # New header name, plus the legacy one for older infra.
        headers["X-OpenRouter-Title"] = app_title
        headers["X-Title"] = app_title
    return headers
