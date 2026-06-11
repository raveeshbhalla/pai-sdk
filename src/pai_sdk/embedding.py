"""The embedding surface — the Python port of embed / embedMany /
EmbeddingModelV2 from the Vercel AI SDK.

A provider supplies an :class:`EmbeddingModel` (``do_embed``); the
:func:`embed` / :func:`embed_many` helpers split inputs into chunks of
``model.max_embeddings_per_call``, run those chunks concurrently (bounded by a
semaphore and gated on ``supports_parallel_calls``), retry retryable provider
errors, and stitch the results back together preserving input order.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .generate import _with_retry


@dataclass
class EmbeddingUsage:
    """Token usage for an embedding call."""

    tokens: Optional[int] = None


@dataclass
class EmbedManyProviderResult:
    """What an EmbeddingModel.do_embed returns for one batch of values."""

    embeddings: list[list[float]]
    usage: EmbeddingUsage
    response: Any = None


@dataclass
class EmbedResult:
    """Result of embed() for a single value."""

    value: str
    embedding: list[float]
    usage: EmbeddingUsage
    response: Any = None


@dataclass
class EmbedManyResult:
    """Result of embed_many() for a list of values (order preserved)."""

    values: list[str]
    embeddings: list[list[float]]
    usage: EmbeddingUsage


class EmbeddingModel(ABC):
    """An embedding model bound to one provider API + model id."""

    provider: str
    model_id: str
    # Max number of values one provider call accepts; None means no limit.
    max_embeddings_per_call: Optional[int] = None
    # Whether multiple chunked calls may run concurrently.
    supports_parallel_calls: bool = True

    @abstractmethod
    async def do_embed(
        self,
        values: list[str],
        *,
        headers: Optional[dict[str, str]] = None,
        provider_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> EmbedManyProviderResult:
        """Embed one batch of values (at most max_embeddings_per_call)."""


def _chunk(values: list[str], size: Optional[int]) -> list[list[str]]:
    if size is None or size <= 0 or len(values) <= size:
        return [list(values)] if values else []
    return [values[i : i + size] for i in range(0, len(values), size)]


async def embed(
    *,
    model: EmbeddingModel,
    value: str,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
) -> EmbedResult:
    """Embed a single value."""
    result = await _with_retry(
        lambda: model.do_embed(
            [value], headers=headers, provider_options=provider_options
        ),
        max_retries,
    )
    return EmbedResult(
        value=value,
        embedding=result.embeddings[0],
        usage=result.usage,
        response=result.response,
    )


async def embed_many(
    *,
    model: EmbeddingModel,
    values: list[str],
    max_parallel_calls: int = 2,
    max_retries: int = 2,
    headers: Optional[dict[str, str]] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
) -> EmbedManyResult:
    """Embed many values, chunking by model.max_embeddings_per_call and
    running chunks concurrently (bounded by max_parallel_calls) when the model
    supports parallel calls, serially otherwise. Output order matches input."""
    chunks = _chunk(values, model.max_embeddings_per_call)
    if not chunks:
        return EmbedManyResult(values=list(values), embeddings=[], usage=EmbeddingUsage(tokens=0))

    async def run(chunk: list[str]) -> EmbedManyProviderResult:
        return await _with_retry(
            lambda: model.do_embed(
                chunk, headers=headers, provider_options=provider_options
            ),
            max_retries,
        )

    if model.supports_parallel_calls and max_parallel_calls > 1 and len(chunks) > 1:
        semaphore = asyncio.Semaphore(max_parallel_calls)

        async def guarded(chunk: list[str]) -> EmbedManyProviderResult:
            async with semaphore:
                return await run(chunk)

        results = await asyncio.gather(*(guarded(chunk) for chunk in chunks))
    else:
        results = []
        for chunk in chunks:
            results.append(await run(chunk))

    embeddings: list[list[float]] = []
    total_tokens: Optional[int] = 0
    for result in results:
        embeddings.extend(result.embeddings)
        if result.usage.tokens is None:
            total_tokens = None
        elif total_tokens is not None:
            total_tokens += result.usage.tokens

    return EmbedManyResult(
        values=list(values),
        embeddings=embeddings,
        usage=EmbeddingUsage(tokens=total_tokens),
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors.

    Raises ValueError if the lengths differ or either vector is all zeros.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length: {len(a)} != {len(b)}."
        )
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity of a zero vector.")
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


__all__ = [
    "EmbeddingModel",
    "EmbeddingUsage",
    "EmbedManyProviderResult",
    "EmbedResult",
    "EmbedManyResult",
    "embed",
    "embed_many",
    "cosine_similarity",
]
