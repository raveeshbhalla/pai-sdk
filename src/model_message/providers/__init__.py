"""Provider factories.

    from model_message.providers import openai, anthropic, google, openrouter

    model = openai("gpt-5.4")              # OpenAI Responses API (default)
    model = openai.chat("gpt-5.4")         # OpenAI Chat Completions API
    model = anthropic("claude-opus-4-8")   # Anthropic Messages API
    model = google("gemini-2.5-flash")     # Gemini via google-genai
    model = openrouter("anthropic/claude-opus-4.6")  # OpenRouter

Or pass a "provider/model" string straight to generate_text/stream_text:
    generate_text(model="anthropic/claude-opus-4-8", ...)
"""

from __future__ import annotations

from typing import Optional

from ..errors import NoSuchProviderError
from ..provider import LanguageModel
from .anthropic import AnthropicLanguageModel
from .google import GoogleLanguageModel
from .openai_chat import OpenAIChatLanguageModel
from .openai_responses import OpenAIResponsesLanguageModel
from .openrouter import OpenRouterLanguageModel, attribution_headers


class OpenAIProvider:
    """openai("gpt-5.4") -> Responses API; openai.chat(...) for Chat Completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._headers = default_headers or {}

    def __call__(self, model_id: str) -> LanguageModel:
        return self.responses(model_id)

    def responses(self, model_id: str) -> LanguageModel:
        return OpenAIResponsesLanguageModel(
            model_id=model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=self._headers,
        )

    def chat(self, model_id: str) -> LanguageModel:
        return OpenAIChatLanguageModel(
            model_id=model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=self._headers,
        )


class AnthropicProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._headers = default_headers or {}

    def __call__(self, model_id: str) -> LanguageModel:
        return AnthropicLanguageModel(
            model_id=model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=self._headers,
        )


class GoogleProvider:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def __call__(self, model_id: str) -> LanguageModel:
        return GoogleLanguageModel(model_id=model_id, api_key=self._api_key)


class OpenRouterProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        app_title: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._headers = attribution_headers(app_url, app_title)

    def __call__(self, model_id: str) -> LanguageModel:
        return OpenRouterLanguageModel(
            model_id=model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=self._headers,
        )


# Default provider instances (credentials from the environment:
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY,
# OPENROUTER_API_KEY).
openai = OpenAIProvider()
anthropic = AnthropicProvider()
google = GoogleProvider()
openrouter = OpenRouterProvider()

create_openai = OpenAIProvider
create_anthropic = AnthropicProvider
create_google = GoogleProvider
create_openrouter = OpenRouterProvider

_REGISTRY = {
    "openai": openai,
    "anthropic": anthropic,
    "google": google,
    "gemini": google,
    "openrouter": openrouter,
}


def resolve_model_string(model: str) -> LanguageModel:
    """Resolve "provider/model-id" strings (e.g. "anthropic/claude-opus-4-8",
    "openrouter/google/gemini-2.5-flash")."""
    provider_name, _, model_id = model.partition("/")
    factory = _REGISTRY.get(provider_name)
    if factory is None or not model_id:
        raise NoSuchProviderError(
            f"Cannot resolve model string '{model}'. Expected 'provider/model-id' "
            f"with provider one of: {', '.join(sorted(_REGISTRY))}."
        )
    return factory(model_id)


__all__ = [
    "openai",
    "anthropic",
    "google",
    "openrouter",
    "create_openai",
    "create_anthropic",
    "create_google",
    "create_openrouter",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "OpenAIChatLanguageModel",
    "OpenAIResponsesLanguageModel",
    "AnthropicLanguageModel",
    "GoogleLanguageModel",
    "OpenRouterLanguageModel",
    "resolve_model_string",
]
