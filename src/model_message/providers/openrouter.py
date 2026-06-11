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

from .openai_chat import OpenAIChatLanguageModel

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


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
