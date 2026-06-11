"""Tests for model_message.registry — create_provider_registry and custom_provider."""

from __future__ import annotations

import pytest

from model_message.errors import NoSuchProviderError
from model_message.provider import LanguageModel
from model_message.providers import openrouter
from model_message.registry import CustomProvider, ProviderRegistry, create_provider_registry, custom_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal provider factory — callable with a model_id."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def __call__(self, model_id: str) -> LanguageModel:
        # Return the FakeModel imported from conftest indirectly — we construct
        # one inline to keep this module self-contained.
        from conftest import FakeModel  # noqa: PLC0415 (local import fine in tests)

        m = FakeModel()
        m.provider = self.name
        m.model_id = model_id
        return m


class _FakeProviderWithEmbedding(_FakeProvider):
    """Provider that also has an `embedding` method."""

    def embedding(self, model_id: str) -> object:
        return {"provider": self.name, "embedding_model_id": model_id}


def _make_fake_model(provider: str = "fake", model_id: str = "fake-1") -> LanguageModel:
    from conftest import FakeModel  # noqa: PLC0415

    m = FakeModel()
    m.provider = provider
    m.model_id = model_id
    return m


# ---------------------------------------------------------------------------
# create_provider_registry — basic resolution
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_language_model_resolves_plain_factory(self) -> None:
        factory = _FakeProvider("openai")
        registry = create_provider_registry({"openai": factory})
        model = registry.language_model("openai:gpt-5.4")
        assert isinstance(model, LanguageModel)
        assert model.model_id == "gpt-5.4"
        assert model.provider == "openai"

    def test_language_model_passes_full_model_id_after_first_separator(self) -> None:
        """Model part may contain the separator itself."""
        factory = _FakeProvider("openai")
        registry = create_provider_registry({"openai": factory})
        # separator appears in the model-id part — only the first split matters
        model = registry.language_model("openai:org:gpt-5.4")
        assert model.model_id == "org:gpt-5.4"

    def test_language_model_with_slash_in_model_id(self) -> None:
        """openrouter:google/gemini-2.5-flash style — slashes in model-id pass through."""
        factory = _FakeProvider("openrouter")
        registry = create_provider_registry({"openrouter": factory})
        model = registry.language_model("openrouter:google/gemini-2.5-flash")
        assert model.model_id == "google/gemini-2.5-flash"

    def test_language_model_with_real_openrouter_factory(self) -> None:
        """Smoke-test that the real OpenRouterProvider factory works via the registry."""
        registry = create_provider_registry({"openrouter": openrouter})
        model = registry.language_model("openrouter:google/gemini-2.5-flash")
        assert isinstance(model, LanguageModel)
        assert model.model_id == "google/gemini-2.5-flash"

    def test_missing_separator_raises_no_such_provider_error(self) -> None:
        registry = create_provider_registry({"openai": _FakeProvider()})
        with pytest.raises(NoSuchProviderError, match="separator"):
            registry.language_model("openai-gpt-5.4")

    def test_unknown_provider_raises_no_such_provider_error(self) -> None:
        registry = create_provider_registry({"openai": _FakeProvider()})
        with pytest.raises(NoSuchProviderError, match="anthropic"):
            registry.language_model("anthropic:claude-opus-4-8")

    def test_unknown_provider_error_lists_available_providers(self) -> None:
        registry = create_provider_registry(
            {"openai": _FakeProvider(), "google": _FakeProvider()}
        )
        with pytest.raises(NoSuchProviderError) as exc_info:
            registry.language_model("unknown:some-model")
        msg = str(exc_info.value)
        assert "google" in msg
        assert "openai" in msg

    def test_missing_separator_error_lists_available_providers(self) -> None:
        registry = create_provider_registry({"openai": _FakeProvider()})
        with pytest.raises(NoSuchProviderError) as exc_info:
            registry.language_model("openai-gpt-5.4")
        assert "openai" in str(exc_info.value)

    # ------------------------------------------------------------------
    # Custom separator
    # ------------------------------------------------------------------

    def test_custom_separator(self) -> None:
        factory = _FakeProvider("openai")
        registry = create_provider_registry({"openai": factory}, separator=" > ")
        model = registry.language_model("openai > gpt-5.4")
        assert model.model_id == "gpt-5.4"

    def test_custom_separator_colon_in_id_treated_as_model_part(self) -> None:
        factory = _FakeProvider("openai")
        # When separator is " > ", a colon is just part of the model id.
        registry = create_provider_registry({"openai": factory}, separator=" > ")
        model = registry.language_model("openai > org:gpt-5.4")
        assert model.model_id == "org:gpt-5.4"

    # ------------------------------------------------------------------
    # embedding_model
    # ------------------------------------------------------------------

    def test_embedding_model_delegates_to_provider_embedding_attr(self) -> None:
        factory = _FakeProviderWithEmbedding("myembed")
        registry = create_provider_registry({"myembed": factory})
        result = registry.embedding_model("myembed:text-embedding-3-small")
        assert result == {
            "provider": "myembed",
            "embedding_model_id": "text-embedding-3-small",
        }

    def test_embedding_model_raises_when_provider_has_no_embedding(self) -> None:
        factory = _FakeProvider("openai")  # no embedding attr
        registry = create_provider_registry({"openai": factory})
        with pytest.raises(NoSuchProviderError, match="embedding"):
            registry.embedding_model("openai:text-embedding-3-small")

    def test_embedding_model_missing_separator_raises(self) -> None:
        registry = create_provider_registry({"openai": _FakeProviderWithEmbedding()})
        with pytest.raises(NoSuchProviderError, match="separator"):
            registry.embedding_model("openai-text-embedding-3-small")


# ---------------------------------------------------------------------------
# custom_provider
# ---------------------------------------------------------------------------


class TestCustomProvider:
    def test_exact_id_hit_returns_configured_model(self) -> None:
        fast_model = _make_fake_model("openai", "gpt-5.4-mini")
        cp = custom_provider(language_models={"fast": fast_model})
        result = cp("fast")
        assert result is fast_model

    def test_miss_with_fallback_delegates_to_fallback(self) -> None:
        fallback = _FakeProvider("openai")
        cp = custom_provider(language_models={}, fallback_provider=fallback)
        model = cp("gpt-5.4")
        assert model.model_id == "gpt-5.4"
        assert model.provider == "openai"

    def test_miss_with_no_fallback_raises_no_such_provider_error(self) -> None:
        cp = custom_provider(
            language_models={"fast": _make_fake_model(), "smart": _make_fake_model()}
        )
        with pytest.raises(NoSuchProviderError, match="gpt-5.4"):
            cp("gpt-5.4")

    def test_miss_error_lists_known_ids(self) -> None:
        cp = custom_provider(
            language_models={"fast": _make_fake_model(), "smart": _make_fake_model()}
        )
        with pytest.raises(NoSuchProviderError) as exc_info:
            cp("unknown")
        msg = str(exc_info.value)
        assert "fast" in msg
        assert "smart" in msg

    def test_exact_id_takes_precedence_over_fallback(self) -> None:
        pinned = _make_fake_model("pinned", "pinned-model")
        fallback = _FakeProvider("fallback")
        cp = custom_provider(
            language_models={"fast": pinned}, fallback_provider=fallback
        )
        assert cp("fast") is pinned

    # ------------------------------------------------------------------
    # custom_provider inside registry
    # ------------------------------------------------------------------

    def test_custom_provider_inside_registry(self) -> None:
        fast = _make_fake_model("openai", "gpt-5.4-mini")
        aliases = custom_provider(language_models={"fast": fast})
        registry = create_provider_registry(
            {"openai": _FakeProvider("openai"), "aliases": aliases}
        )
        model = registry.language_model("aliases:fast")
        assert model is fast

    def test_registry_with_fallback_custom_provider(self) -> None:
        fallback = _FakeProvider("openai")
        cp = custom_provider(fallback_provider=fallback)
        registry = create_provider_registry({"openai": cp})
        model = registry.language_model("openai:gpt-5.4")
        assert model.model_id == "gpt-5.4"

    # ------------------------------------------------------------------
    # embedding on custom_provider
    # ------------------------------------------------------------------

    def test_custom_provider_embedding_exact_hit(self) -> None:
        embed_stub = object()
        cp = custom_provider(embedding_models={"text-emb-3": embed_stub})
        assert cp.embedding("text-emb-3") is embed_stub

    def test_custom_provider_embedding_fallback_delegation(self) -> None:
        fallback = _FakeProviderWithEmbedding("myembed")
        cp = custom_provider(embedding_models={}, fallback_provider=fallback)
        result = cp.embedding("text-embedding-3-small")
        assert result["embedding_model_id"] == "text-embedding-3-small"

    def test_custom_provider_embedding_miss_no_fallback_raises(self) -> None:
        cp = custom_provider(embedding_models={"text-emb-3": object()})
        with pytest.raises(NoSuchProviderError, match="text-emb-4"):
            cp.embedding("text-emb-4")

    def test_custom_provider_embedding_miss_error_lists_known_ids(self) -> None:
        cp = custom_provider(embedding_models={"text-emb-3": object()})
        with pytest.raises(NoSuchProviderError) as exc_info:
            cp.embedding("unknown")
        assert "text-emb-3" in str(exc_info.value)

    def test_custom_provider_embedding_fallback_without_embedding_raises(self) -> None:
        # fallback has no embedding attribute — miss should raise
        fallback = _FakeProvider("openai")
        cp = custom_provider(embedding_models={}, fallback_provider=fallback)
        with pytest.raises(NoSuchProviderError):
            cp.embedding("some-embed-model")

    # ------------------------------------------------------------------
    # embedding_model in registry with custom_provider
    # ------------------------------------------------------------------

    def test_registry_embedding_model_via_custom_provider(self) -> None:
        embed_stub = object()
        aliases = custom_provider(embedding_models={"small": embed_stub})
        registry = create_provider_registry({"aliases": aliases})
        # CustomProvider has an `embedding` attr, so registry delegates to it
        result = registry.embedding_model("aliases:small")
        assert result is embed_stub

    # ------------------------------------------------------------------
    # empty / None args
    # ------------------------------------------------------------------

    def test_custom_provider_with_no_args_raises_on_any_model_id(self) -> None:
        cp = custom_provider()
        with pytest.raises(NoSuchProviderError):
            cp("anything")

    def test_custom_provider_returns_correct_type(self) -> None:
        cp = custom_provider()
        assert isinstance(cp, CustomProvider)

    def test_create_provider_registry_returns_correct_type(self) -> None:
        registry = create_provider_registry({})
        assert isinstance(registry, ProviderRegistry)
