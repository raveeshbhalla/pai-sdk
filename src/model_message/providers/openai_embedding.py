"""OpenAI embedding model, mirroring @ai-sdk/openai's text-embedding models.

Uses ``client.embeddings.create(model, input, dimensions?)``. The
``dimensions`` parameter is read from
``provider_options={"openai": {"dimensions": N}}``.

The client construction is overridable via the ``client_factory`` field so an
Azure variant can supply an ``AsyncAzureOpenAI`` client (see AzureProvider in
providers/__init__.py) without subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..embedding import EmbeddingModel, EmbeddingUsage, EmbedManyProviderResult
from ..errors import MissingDependencyError
from ._util import wrap_provider_error


@dataclass
class OpenAIEmbeddingModel(EmbeddingModel):
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: dict[str, str] = field(default_factory=dict)
    provider: str = "openai.embedding"
    max_embeddings_per_call: Optional[int] = 2048
    supports_parallel_calls: bool = True
    # When set, builds the AsyncOpenAI-compatible client (e.g. AsyncAzureOpenAI).
    client_factory: Optional[Callable[[], Any]] = None
    # providerOptions key whose "dimensions" is forwarded to the API.
    provider_options_key: str = "openai"
    _client_cache: Any = field(default=None, repr=False, compare=False)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        if self.client_factory is not None:
            self._client_cache = self.client_factory()
            return self._client_cache
        try:
            import openai
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai") from exc
        kwargs: dict[str, Any] = {"max_retries": 0}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        self._client_cache = openai.AsyncOpenAI(**kwargs)
        return self._client_cache

    async def do_embed(
        self,
        values: list[str],
        *,
        headers: Optional[dict[str, str]] = None,
        provider_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> EmbedManyProviderResult:
        client = self._client()
        opts = (provider_options or {}).get(self.provider_options_key) or {}
        request: dict[str, Any] = {"model": self.model_id, "input": values}
        if opts.get("dimensions") is not None:
            request["dimensions"] = opts["dimensions"]
        user = opts.get("user")
        if user is not None:
            request["user"] = user
        if headers:
            request["extra_headers"] = headers
        try:
            response = await client.embeddings.create(**request)
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        ordered = sorted(response.data, key=lambda d: d.index)
        embeddings = [list(d.embedding) for d in ordered]
        usage = response.usage
        tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        return EmbedManyProviderResult(
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=tokens),
            response=response,
        )


__all__ = ["OpenAIEmbeddingModel"]
