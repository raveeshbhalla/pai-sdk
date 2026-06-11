"""Google Gemini embedding model via google-genai's embed_content.

Uses ``client.aio.models.embed_content(model=, contents=, config=)``.
``output_dimensionality`` and other config keys are read from
``provider_options={"google": {...}}`` and merged into the EmbedContentConfig.

Verified SDK response shape (google-genai):
- response.embeddings: list[ContentEmbedding]
- ContentEmbedding.values: list[float]
- ContentEmbedding.statistics.token_count: float | None

The client construction is overridable via the ``client_factory`` field so a
Vertex variant can supply a ``genai.Client(vertexai=True, ...)`` client without
subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..embedding import EmbeddingModel, EmbeddingUsage, EmbedManyProviderResult
from ..errors import MissingDependencyError
from ._util import wrap_provider_error


@dataclass
class GoogleEmbeddingModel(EmbeddingModel):
    model_id: str
    api_key: Optional[str] = None
    provider: str = "google.embedding"
    # google-genai batches up to 100 contents per embed_content call.
    max_embeddings_per_call: Optional[int] = 100
    supports_parallel_calls: bool = True
    # When set, builds the genai.Client (e.g. with vertexai=True).
    client_factory: Optional[Callable[[], Any]] = None
    provider_options_key: str = "google"
    _client_cache: Any = field(default=None, repr=False, compare=False)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        if self.client_factory is not None:
            self._client_cache = self.client_factory()
            return self._client_cache
        try:
            from google import genai
        except ImportError as exc:
            raise MissingDependencyError("google-genai", "google") from exc
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client_cache = genai.Client(**kwargs)
        return self._client_cache

    async def do_embed(
        self,
        values: list[str],
        *,
        headers: Optional[dict[str, str]] = None,
        provider_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> EmbedManyProviderResult:
        client = self._client()
        config: dict[str, Any] = {}
        for name, value in ((provider_options or {}).get(self.provider_options_key) or {}).items():
            config.setdefault(name, value)
        try:
            response = await client.aio.models.embed_content(
                model=self.model_id,
                contents=values,
                config=config or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise wrap_provider_error(exc, self.provider) from exc

        content_embeddings = getattr(response, "embeddings", None) or []
        embeddings = [list(e.values or []) for e in content_embeddings]

        tokens: Optional[int] = 0
        for e in content_embeddings:
            stats = getattr(e, "statistics", None)
            count = getattr(stats, "token_count", None) if stats is not None else None
            if count is None:
                tokens = None
                break
            tokens += int(count)

        return EmbedManyProviderResult(
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=tokens),
            response=response,
        )


__all__ = ["GoogleEmbeddingModel"]
