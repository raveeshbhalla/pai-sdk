"""Provider registry — mirrors AI SDK createProviderRegistry / customProvider.

Usage::

    from model_message.providers import openai, anthropic
    from model_message.registry import create_provider_registry, custom_provider

    registry = create_provider_registry({
        "openai": openai,
        "anthropic": anthropic,
        "aliases": custom_provider(
            language_models={"fast": openai("gpt-5.4-mini")},
        ),
    })
    model = registry.language_model("openai:gpt-5.4")
    model = registry.language_model("aliases:fast")
    model = registry.language_model("openrouter:google/gemini-2.5-flash")
"""

from __future__ import annotations

from typing import Any, Optional

from .errors import NoSuchProviderError
from .provider import LanguageModel


class ProviderRegistry:
    """A registry of named providers that resolves ``<prefix><sep><model-id>``
    strings into :class:`~model_message.provider.LanguageModel` instances.

    Instantiate via :func:`create_provider_registry`.
    """

    def __init__(self, providers: dict[str, Any], *, separator: str = ":") -> None:
        self._providers = dict(providers)
        self._separator = separator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, id: str) -> tuple[str, str]:
        """Split *id* on the first separator occurrence.

        Returns ``(provider_prefix, model_id)``.

        Raises :class:`~model_message.errors.NoSuchProviderError` when the
        separator is absent.
        """
        sep = self._separator
        if sep not in id:
            raise NoSuchProviderError(
                f"Model id {id!r} does not contain the separator {sep!r}. "
                f"Expected format: '<provider>{sep}<model-id>'. "
                f"Available providers: {', '.join(sorted(self._providers))}."
            )
        prefix, _, model_id = id.partition(sep)
        return prefix, model_id

    def _resolve_provider(self, prefix: str) -> Any:
        provider = self._providers.get(prefix)
        if provider is None:
            raise NoSuchProviderError(
                f"No provider registered under {prefix!r}. "
                f"Available providers: {', '.join(sorted(self._providers))}."
            )
        return provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def language_model(self, id: str) -> LanguageModel:
        """Resolve *id* (e.g. ``"openai:gpt-5.4"``) to a :class:`LanguageModel`.

        The first occurrence of the separator divides the provider prefix from
        the model id, so model ids containing the separator (or slashes) are
        handled correctly — e.g. ``"openrouter:google/gemini-2.5-flash"``.
        """
        prefix, model_id = self._split(id)
        provider = self._resolve_provider(prefix)
        return provider(model_id)

    def embedding_model(self, id: str) -> Any:
        """Resolve *id* to an embedding model.

        Delegates to ``provider.embedding(model_id)`` when the resolved
        provider exposes an ``embedding`` attribute; raises
        :class:`~model_message.errors.NoSuchProviderError` otherwise.
        """
        prefix, model_id = self._split(id)
        provider = self._resolve_provider(prefix)
        embedding_fn = getattr(provider, "embedding", None)
        if embedding_fn is None:
            raise NoSuchProviderError(
                f"Provider {prefix!r} does not support embedding models. "
                "Only providers with an 'embedding' attribute can be used with "
                "embedding_model()."
            )
        return embedding_fn(model_id)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        providers_repr = ", ".join(repr(k) for k in sorted(self._providers))
        return (
            f"ProviderRegistry(providers=[{providers_repr}], "
            f"separator={self._separator!r})"
        )


class CustomProvider:
    """A provider whose ``__call__`` resolves model ids from a pre-built dict.

    ``language_models`` maps short alias strings to already-instantiated
    :class:`LanguageModel` objects (e.g. ``{"fast": openai("gpt-5.4-mini")}``).
    An optional *fallback_provider* is called when the id is not found in the
    dict.  If neither lookup succeeds a :class:`~model_message.errors.NoSuchProviderError`
    listing the known ids is raised.

    Instantiate via :func:`custom_provider`.
    """

    def __init__(
        self,
        *,
        language_models: Optional[dict[str, LanguageModel]] = None,
        embedding_models: Optional[dict[str, Any]] = None,
        fallback_provider: Any = None,
    ) -> None:
        self._language_models: dict[str, LanguageModel] = language_models or {}
        self._embedding_models: dict[str, Any] = embedding_models or {}
        self._fallback = fallback_provider

    # ------------------------------------------------------------------
    # Provider protocol — callable with model_id
    # ------------------------------------------------------------------

    def __call__(self, model_id: str) -> LanguageModel:
        """Return the :class:`LanguageModel` for *model_id*.

        Lookup order:

        1. ``language_models[model_id]``
        2. ``fallback_provider(model_id)`` (if provided)
        3. :class:`~model_message.errors.NoSuchProviderError`
        """
        if model_id in self._language_models:
            return self._language_models[model_id]
        if self._fallback is not None:
            return self._fallback(model_id)
        known = ", ".join(repr(k) for k in sorted(self._language_models))
        raise NoSuchProviderError(
            f"No language model registered under {model_id!r}. "
            f"Known ids: {known or '(none)'}."
        )

    # ------------------------------------------------------------------
    # Embedding support — duck-typed, no import from embeddings module
    # ------------------------------------------------------------------

    def embedding(self, model_id: str) -> Any:
        """Return the embedding model for *model_id*.

        Lookup order:

        1. ``embedding_models[model_id]``
        2. ``fallback_provider.embedding(model_id)`` (if the fallback exposes
           an ``embedding`` attribute)
        3. :class:`~model_message.errors.NoSuchProviderError`
        """
        if model_id in self._embedding_models:
            return self._embedding_models[model_id]
        fallback_embedding = getattr(self._fallback, "embedding", None)
        if fallback_embedding is not None:
            return fallback_embedding(model_id)
        known = ", ".join(repr(k) for k in sorted(self._embedding_models))
        raise NoSuchProviderError(
            f"No embedding model registered under {model_id!r}. "
            f"Known ids: {known or '(none)'}."
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        lm_keys = list(self._language_models)
        em_keys = list(self._embedding_models)
        parts = []
        if lm_keys:
            parts.append(f"language_models=[{', '.join(repr(k) for k in sorted(lm_keys))}]")
        if em_keys:
            parts.append(f"embedding_models=[{', '.join(repr(k) for k in sorted(em_keys))}]")
        if self._fallback is not None:
            parts.append(f"fallback_provider={self._fallback!r}")
        return f"CustomProvider({', '.join(parts)})"


def create_provider_registry(
    providers: dict[str, Any],
    *,
    separator: str = ":",
) -> ProviderRegistry:
    """Create a :class:`ProviderRegistry` from a dict of named providers.

    Each value in *providers* must be callable with a model-id string and
    return a :class:`~model_message.provider.LanguageModel` — either one of
    the built-in provider factory instances (``openai``, ``anthropic``, …) or
    a :class:`CustomProvider` created by :func:`custom_provider`.

    The *separator* (default ``":"``) is used to split the registry id into
    ``<prefix><sep><model-id>`` when calling :meth:`ProviderRegistry.language_model`.

    Example::

        from model_message.providers import openai, anthropic
        from model_message.registry import create_provider_registry, custom_provider

        registry = create_provider_registry(
            {
                "openai": openai,
                "anthropic": anthropic,
                "aliases": custom_provider(
                    language_models={"fast": openai("gpt-5.4-mini")},
                ),
            }
        )
        model = registry.language_model("openai:gpt-5.4")
        model = registry.language_model("aliases:fast")
    """
    return ProviderRegistry(providers, separator=separator)


def custom_provider(
    *,
    language_models: Optional[dict[str, LanguageModel]] = None,
    embedding_models: Optional[dict[str, Any]] = None,
    fallback_provider: Any = None,
) -> CustomProvider:
    """Create a :class:`CustomProvider` for use inside a registry or standalone.

    *language_models* maps short alias strings to pre-configured
    :class:`~model_message.provider.LanguageModel` instances — useful for
    giving friendly names to middleware-wrapped or pre-configured models.

    *embedding_models* maps ids to pre-configured embedding model objects
    (duck-typed; the embeddings module is not imported here).

    *fallback_provider* is called when a requested model id is not found in the
    dict; it must itself be callable with a model-id string.

    Example::

        from model_message.providers import openai
        from model_message.registry import custom_provider, create_provider_registry

        aliases = custom_provider(
            language_models={
                "fast": openai("gpt-5.4-mini"),
                "smart": openai("gpt-5.4"),
            },
        )
        registry = create_provider_registry({"aliases": aliases, "openai": openai})
        model = registry.language_model("aliases:fast")
    """
    return CustomProvider(
        language_models=language_models,
        embedding_models=embedding_models,
        fallback_provider=fallback_provider,
    )


__all__ = [
    "ProviderRegistry",
    "CustomProvider",
    "create_provider_registry",
    "custom_provider",
]
