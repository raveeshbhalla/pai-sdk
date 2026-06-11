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
from .azure import AzureChatLanguageModel, AzureResponsesLanguageModel
from .bedrock import BedrockAnthropicLanguageModel
from .google import GoogleLanguageModel
from .openai_chat import OpenAIChatLanguageModel
from .openai_responses import OpenAIResponsesLanguageModel
from .openrouter import OpenRouterLanguageModel, attribution_headers
from .vertex import (
    VertexAnthropicLanguageModel,
    VertexGoogleLanguageModel,
)


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


class BedrockProvider:
    """bedrock("anthropic.claude-opus-4-8") -> Anthropic Claude on Amazon Bedrock.

    Bedrock Anthropic model ids carry an "anthropic." prefix (e.g.
    "anthropic.claude-opus-4-8"); the factory passes them through verbatim and
    does NOT add any prefix. aws_region falls back to the AWS_REGION env var;
    aws_access_key / aws_secret_key / aws_session_token are passed through to
    the underlying AsyncAnthropicBedrock client (else the default AWS
    credential chain is used).
    """

    def __init__(
        self,
        aws_region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._aws_region = aws_region
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key
        self._aws_session_token = aws_session_token
        self._base_url = base_url
        self._headers = default_headers or {}

    def __call__(self, model_id: str) -> LanguageModel:
        return BedrockAnthropicLanguageModel(
            model_id=model_id,
            base_url=self._base_url,
            default_headers=self._headers,
            aws_region=self._aws_region,
            aws_access_key=self._aws_access_key,
            aws_secret_key=self._aws_secret_key,
            aws_session_token=self._aws_session_token,
        )


class VertexProvider:
    """vertex("gemini-2.5-flash") -> Gemini on Vertex AI;
    vertex.anthropic("claude-opus-4-8") -> Claude on Vertex AI.

    project falls back to GOOGLE_CLOUD_PROJECT, location to
    GOOGLE_CLOUD_LOCATION (default "us-central1").
    """

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self._project = project
        self._location = location

    def __call__(self, model_id: str) -> LanguageModel:
        return VertexGoogleLanguageModel(
            model_id=model_id, project=self._project, location=self._location
        )

    def anthropic(self, model_id: str) -> LanguageModel:
        return VertexAnthropicLanguageModel(
            model_id=model_id, project=self._project, location=self._location
        )


class AzureProvider:
    """azure(deployment) -> Responses API; azure.chat(deployment) -> Chat
    Completions. The model id is the Azure *deployment* name.

    Env fallbacks: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    OPENAI_API_VERSION.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._headers = default_headers or {}

    def __call__(self, deployment: str) -> LanguageModel:
        return self.responses(deployment)

    def responses(self, deployment: str) -> LanguageModel:
        return AzureResponsesLanguageModel(
            model_id=deployment,
            api_key=self._api_key,
            default_headers=self._headers,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
        )

    def chat(self, deployment: str) -> LanguageModel:
        return AzureChatLanguageModel(
            model_id=deployment,
            api_key=self._api_key,
            default_headers=self._headers,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
        )


# Default provider instances (credentials from the environment:
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY,
# OPENROUTER_API_KEY).
openai = OpenAIProvider()
anthropic = AnthropicProvider()
google = GoogleProvider()
openrouter = OpenRouterProvider()
bedrock = BedrockProvider()
vertex = VertexProvider()
azure = AzureProvider()

create_openai = OpenAIProvider
create_anthropic = AnthropicProvider
create_google = GoogleProvider
create_openrouter = OpenRouterProvider
create_bedrock = BedrockProvider
create_vertex = VertexProvider
create_azure = AzureProvider

_REGISTRY = {
    "openai": openai,
    "anthropic": anthropic,
    "google": google,
    "gemini": google,
    "openrouter": openrouter,
    "bedrock": bedrock,
    "vertex": vertex,
    "azure": azure,
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
    "bedrock",
    "vertex",
    "azure",
    "create_openai",
    "create_anthropic",
    "create_google",
    "create_openrouter",
    "create_bedrock",
    "create_vertex",
    "create_azure",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "BedrockProvider",
    "VertexProvider",
    "AzureProvider",
    "OpenAIChatLanguageModel",
    "OpenAIResponsesLanguageModel",
    "AnthropicLanguageModel",
    "GoogleLanguageModel",
    "OpenRouterLanguageModel",
    "BedrockAnthropicLanguageModel",
    "VertexGoogleLanguageModel",
    "VertexAnthropicLanguageModel",
    "AzureResponsesLanguageModel",
    "AzureChatLanguageModel",
    "resolve_model_string",
]
