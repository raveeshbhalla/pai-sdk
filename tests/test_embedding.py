"""Tests for the embedding surface: cosine_similarity, embed/embed_many over a
fake model (chunking, ordering, parallel bound, usage summing), and end-to-end
through the real OpenAI and google-genai SDKs."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pai_sdk.embedding import (
    EmbeddingModel,
    EmbeddingUsage,
    EmbedManyProviderResult,
    cosine_similarity,
    embed,
    embed_many,
)

openai_sdk = pytest.importorskip("openai")


# --- cosine_similarity ---------------------------------------------------------


def test_cosine_identical():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_length_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])


# --- FakeEmbeddingModel --------------------------------------------------------


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic embedding: each value maps to [len(value), index]."""

    def __init__(
        self,
        max_embeddings_per_call=None,
        supports_parallel_calls=True,
        delay=0.0,
        tokens_per_call=3,
    ):
        self.provider = "fake.embedding"
        self.model_id = "fake-embed"
        self.max_embeddings_per_call = max_embeddings_per_call
        self.supports_parallel_calls = supports_parallel_calls
        self.delay = delay
        self.tokens_per_call = tokens_per_call
        self.calls: list[list[str]] = []
        self._concurrent = 0
        self.max_concurrent = 0

    async def do_embed(self, values, *, headers=None, provider_options=None):
        self.calls.append(list(values))
        self._concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self._concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            embeddings = [[float(len(v)), float(i)] for i, v in enumerate(values)]
        finally:
            self._concurrent -= 1
        return EmbedManyProviderResult(
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=self.tokens_per_call),
        )


async def test_embed_single():
    model = FakeEmbeddingModel()
    result = await embed(model=model, value="hello")
    assert result.value == "hello"
    assert result.embedding == [5.0, 0.0]
    assert result.usage.tokens == 3
    assert model.calls == [["hello"]]


async def test_embed_many_no_chunking():
    model = FakeEmbeddingModel(max_embeddings_per_call=None)
    values = ["a", "bb", "ccc"]
    result = await embed_many(model=model, values=values)
    assert len(model.calls) == 1
    assert model.calls[0] == values
    assert result.embeddings == [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
    assert result.values == values


async def test_embed_many_chunking_and_order():
    model = FakeEmbeddingModel(max_embeddings_per_call=2)
    values = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = await embed_many(model=model, values=values)
    # 5 values / chunk 2 -> chunks of [2, 2, 1]
    assert [len(c) for c in model.calls] == [2, 2, 1]
    # order preserved; first element of each embedding is the value length
    assert [e[0] for e in result.embeddings] == [1.0, 2.0, 3.0, 4.0, 5.0]


async def test_embed_many_usage_summing():
    model = FakeEmbeddingModel(max_embeddings_per_call=2, tokens_per_call=10)
    result = await embed_many(model=model, values=["a", "b", "c", "d", "e"])
    # 3 chunks * 10 tokens each
    assert result.usage.tokens == 30


async def test_embed_many_parallel_bound():
    model = FakeEmbeddingModel(
        max_embeddings_per_call=1, supports_parallel_calls=True, delay=0.02
    )
    values = [str(i) for i in range(6)]
    await embed_many(model=model, values=values, max_parallel_calls=2)
    assert len(model.calls) == 6
    assert model.max_concurrent <= 2
    assert model.max_concurrent == 2


async def test_embed_many_serial_when_not_parallel():
    model = FakeEmbeddingModel(
        max_embeddings_per_call=1, supports_parallel_calls=False, delay=0.01
    )
    values = [str(i) for i in range(5)]
    await embed_many(model=model, values=values, max_parallel_calls=4)
    assert model.max_concurrent == 1


async def test_embed_many_usage_none_propagates():
    model = FakeEmbeddingModel(max_embeddings_per_call=2, tokens_per_call=None)
    result = await embed_many(model=model, values=["a", "b", "c"])
    assert result.usage.tokens is None


# --- OpenAI end-to-end via mocked HTTP through the real SDK --------------------


def openai_embedding_model(handler):
    from pai_sdk.providers.openai_embedding import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel(model_id="text-embedding-3-small", api_key="test")
    model._client_cache = openai_sdk.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


async def test_openai_embedding_e2e():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"object": "embedding", "index": i, "embedding": [float(i), 0.5, -0.5]}
            for i in range(len(body["input"]))
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    model = openai_embedding_model(handler)
    result = await embed_many(
        model=model,
        values=["alpha", "beta"],
        provider_options={"openai": {"dimensions": 3}},
    )

    body = requests[0]
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["alpha", "beta"]
    assert body["dimensions"] == 3
    assert result.embeddings == [[0.0, 0.5, -0.5], [1.0, 0.5, -0.5]]
    assert result.usage.tokens == 8


async def test_openai_embedding_single_e2e():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    model = openai_embedding_model(handler)
    result = await embed(model=model, value="hello")
    assert result.embedding == [0.1, 0.2]
    assert result.usage.tokens == 3


async def test_openai_embedding_order_preserved_out_of_order_data():
    def handler(request: httpx.Request) -> httpx.Response:
        # Return data deliberately out of index order.
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [9.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    model = openai_embedding_model(handler)
    result = await embed_many(model=model, values=["a", "b"])
    assert result.embeddings == [[1.0], [9.0]]


# --- Google embedding response mapping (no network) ---------------------------


async def test_google_embedding_mapping():
    google_types = pytest.importorskip("google.genai.types")
    from pai_sdk.providers.google_embedding import GoogleEmbeddingModel

    response = google_types.EmbedContentResponse.model_validate(
        {
            "embeddings": [
                {
                    "values": [0.1, 0.2, 0.3],
                    "statistics": {"token_count": 5, "truncated": False},
                },
                {
                    "values": [0.4, 0.5, 0.6],
                    "statistics": {"token_count": 7, "truncated": False},
                },
            ]
        }
    )

    class _FakeAioModels:
        def __init__(self):
            self.kwargs = None

        async def embed_content(self, **kwargs):
            self.kwargs = kwargs
            return response

    class _FakeAio:
        def __init__(self, models):
            self.models = models

    class _FakeClient:
        def __init__(self, models):
            self.aio = _FakeAio(models)

    fake_models = _FakeAioModels()
    model = GoogleEmbeddingModel(model_id="text-embedding-004")
    model._client_cache = _FakeClient(fake_models)

    result = await embed_many(
        model=model,
        values=["a", "b"],
        provider_options={"google": {"output_dimensionality": 3}},
    )

    assert fake_models.kwargs["model"] == "text-embedding-004"
    assert fake_models.kwargs["contents"] == ["a", "b"]
    assert fake_models.kwargs["config"] == {"output_dimensionality": 3}
    assert result.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert result.usage.tokens == 12


def test_google_embedding_defaults():
    from pai_sdk.providers.google_embedding import GoogleEmbeddingModel

    model = GoogleEmbeddingModel(model_id="text-embedding-004")
    assert model.provider == "google.embedding"
    assert model.max_embeddings_per_call == 100


def test_openai_embedding_defaults():
    from pai_sdk.providers.openai_embedding import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel(model_id="text-embedding-3-small")
    assert model.provider == "openai.embedding"
    assert model.max_embeddings_per_call == 2048
    assert model.supports_parallel_calls is True


def test_factory_methods():
    from pai_sdk.providers import azure, google, openai, vertex
    from pai_sdk.providers.google_embedding import GoogleEmbeddingModel
    from pai_sdk.providers.openai_embedding import OpenAIEmbeddingModel

    assert isinstance(openai.embedding("text-embedding-3-small"), OpenAIEmbeddingModel)
    assert isinstance(google.embedding("text-embedding-004"), GoogleEmbeddingModel)

    azure_model = azure.embedding("my-deployment")
    assert isinstance(azure_model, OpenAIEmbeddingModel)
    assert azure_model.provider == "azure.embedding"
    assert azure_model.client_factory is not None

    vertex_model = vertex.embedding("text-embedding-004")
    assert isinstance(vertex_model, GoogleEmbeddingModel)
    assert vertex_model.provider == "google.vertex.embedding"
    assert vertex_model.client_factory is not None
