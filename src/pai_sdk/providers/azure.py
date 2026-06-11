"""Azure OpenAI provider, mirroring @ai-sdk/azure.

The request/response mapping is identical to the direct OpenAI provider — only
the SDK client differs (openai.AsyncAzureOpenAI, which targets an Azure
deployment endpoint with an api-version). The model id is the Azure
*deployment* name.

- azure(deployment)       -> Responses API. provider "azure.responses".
- azure.chat(deployment)  -> Chat Completions.  provider "azure.chat".

Env fallbacks: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..errors import MissingDependencyError
from .openai_chat import OpenAIChatLanguageModel
from .openai_responses import OpenAIResponsesLanguageModel

_DEFAULT_API_VERSION = "2024-10-21"


def _azure_client_kwargs(
    api_key: Optional[str],
    azure_endpoint: Optional[str],
    api_version: Optional[str],
    default_headers: dict[str, str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_retries": 0}
    endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        kwargs["azure_endpoint"] = endpoint
    version = (
        api_version
        or os.environ.get("OPENAI_API_VERSION")
        or _DEFAULT_API_VERSION
    )
    kwargs["api_version"] = version
    key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    if key:
        kwargs["api_key"] = key
    if default_headers:
        kwargs["default_headers"] = default_headers
    return kwargs


@dataclass
class AzureResponsesLanguageModel(OpenAIResponsesLanguageModel):
    provider: str = "azure.responses"
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import openai
        except ImportError as exc:
            raise MissingDependencyError("openai", "azure") from exc
        self._client_cache = openai.AsyncAzureOpenAI(
            **_azure_client_kwargs(
                self.api_key, self.azure_endpoint, self.api_version, self.default_headers
            )
        )
        return self._client_cache


@dataclass
class AzureChatLanguageModel(OpenAIChatLanguageModel):
    provider: str = "azure.chat"
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import openai
        except ImportError as exc:
            raise MissingDependencyError("openai", "azure") from exc
        self._client_cache = openai.AsyncAzureOpenAI(
            **_azure_client_kwargs(
                self.api_key, self.azure_endpoint, self.api_version, self.default_headers
            )
        )
        return self._client_cache
