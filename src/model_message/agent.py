"""Agent — the AI SDK ToolLoopAgent abstraction for Python.

An Agent stores default call parameters and provides ``generate()`` /
``stream()`` methods that merge per-call overrides before delegating to
``generate_text`` / ``stream_text``.

Key difference from bare ``generate_text``: when *no* ``stop_when`` is
supplied the Agent defaults to ``step_count_is(20)`` (matching the AI SDK
Agent default), whereas ``generate_text`` defaults to a single step.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from .generate import (
    StopCondition,
    StreamTextResult,
    generate_text,
    step_count_is,
    stream_text,
)
from .provider import LanguageModel
from .results import GenerateTextResult, StepResult
from .stream import TextStreamPart
from .tools import ToolSet

# The complete set of parameter names that can be overridden per-call.
_VALID_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "system",
        "tools",
        "tool_choice",
        "active_tools",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "stop_sequences",
        "seed",
        "max_retries",
        "headers",
        "provider_options",
        "stop_when",
        "on_step_finish",
    }
)


class Agent:
    """An AI SDK-style agent that wraps ``generate_text`` / ``stream_text``
    with stored defaults and a multi-step tool loop.

    Parameters
    ----------
    model:
        The language model (required). May be a ``LanguageModel`` instance or
        an AI SDK gateway model string (e.g. ``"anthropic/claude-opus-4-8"``).
    system:
        System prompt prepended to every call.
    tools:
        Default tool set. Per-call ``tools`` replaces this entirely.
    tool_choice:
        Default tool-choice policy.
    active_tools:
        Subset of tool names to expose by default.
    max_output_tokens:
        Maximum tokens to generate per step.
    temperature / top_p / top_k / presence_penalty / frequency_penalty:
        Sampling parameters.
    stop_sequences:
        Sequences that stop generation.
    seed:
        Random seed for deterministic sampling.
    max_retries:
        How many times to retry retryable API errors (default 2).
    headers:
        Extra HTTP headers forwarded to the provider.
    provider_options:
        Provider-specific passthrough options, keyed by provider name.
        Per-call values *shallow-merge* by provider key (per-call wins).
    stop_when:
        Stop condition(s). Defaults to ``step_count_is(20)`` — unlike the
        bare ``generate_text`` whose default is a single step.
    on_step_finish:
        Callback invoked after each step completes (sync or async).
    name:
        Optional display name for the agent (purely informational).
    """

    _AGENT_DEFAULT_MAX_STEPS = 20

    def __init__(
        self,
        *,
        model: Union[str, LanguageModel],
        system: Optional[str] = None,
        tools: Optional[ToolSet] = None,
        tool_choice: Optional[Any] = None,
        active_tools: Optional[Sequence[str]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        seed: Optional[int] = None,
        max_retries: int = 2,
        headers: Optional[dict[str, str]] = None,
        provider_options: Optional[dict[str, dict[str, Any]]] = None,
        stop_when: Union[StopCondition, Sequence[StopCondition], None] = None,
        on_step_finish: Optional[Callable[[StepResult], Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.system = system
        self.tools = tools
        self.tool_choice = tool_choice
        self.active_tools = active_tools
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop_sequences = stop_sequences
        self.seed = seed
        self.max_retries = max_retries
        self.headers = headers
        self.provider_options = provider_options
        self.stop_when = stop_when
        self.on_step_finish = on_step_finish
        self.name = name

    # ------------------------------------------------------------------
    # Private merge helper
    # ------------------------------------------------------------------

    def _merge(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge constructor defaults with per-call *overrides*.

        Rules:
        - Unknown override keys raise ``TypeError``.
        - A non-``None`` override value wins over the default.
        - ``tools`` override *replaces* the default entirely (no merging).
        - ``provider_options`` shallow-merges by provider key; per-call wins
          for each provider key it provides.
        - ``stop_when``: if neither the default nor the override specifies a
          value, fall back to ``step_count_is(20)`` (Agent default).
        """
        unknown = set(overrides) - _VALID_OVERRIDE_KEYS
        if unknown:
            pretty = ", ".join(sorted(unknown))
            valid = ", ".join(sorted(_VALID_OVERRIDE_KEYS))
            raise TypeError(
                f"Agent.generate/stream received unknown override key(s): "
                f"{pretty}. Valid keys are: {valid}"
            )

        def pick(name: str) -> Any:
            """Return override value if not None, otherwise the default."""
            val = overrides.get(name)
            return val if val is not None else getattr(self, name)

        # provider_options: shallow-merge by provider key
        base_po: dict[str, dict[str, Any]] = self.provider_options or {}
        override_po: dict[str, dict[str, Any]] = overrides.get("provider_options") or {}
        merged_po: dict[str, dict[str, Any]] = {**base_po, **override_po}

        # stop_when: use override, else default, else agent max-steps default
        stop_when = overrides.get("stop_when")
        if stop_when is None:
            stop_when = self.stop_when
        if stop_when is None:
            stop_when = step_count_is(self._AGENT_DEFAULT_MAX_STEPS)

        params: dict[str, Any] = {
            "model": pick("model"),
            "system": pick("system"),
            "tools": pick("tools"),
            "tool_choice": pick("tool_choice"),
            "active_tools": pick("active_tools"),
            "max_output_tokens": pick("max_output_tokens"),
            "temperature": pick("temperature"),
            "top_p": pick("top_p"),
            "top_k": pick("top_k"),
            "presence_penalty": pick("presence_penalty"),
            "frequency_penalty": pick("frequency_penalty"),
            "stop_sequences": pick("stop_sequences"),
            "seed": pick("seed"),
            "max_retries": pick("max_retries"),
            "headers": pick("headers"),
            "provider_options": merged_po or None,
            "stop_when": stop_when,
            "on_step_finish": pick("on_step_finish"),
        }
        return params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        prompt: Any = None,
        messages: Any = None,
        **overrides: Any,
    ) -> GenerateTextResult:
        """Generate text with the agent's defaults (multi-step by default).

        Parameters
        ----------
        prompt:
            A plain-text prompt string (mutually exclusive with *messages*).
        messages:
            Conversation history (mutually exclusive with *prompt*).
        **overrides:
            Any constructor parameter can be overridden per-call; per-call
            value wins when not ``None``.
        """
        params = self._merge(overrides)
        return await generate_text(
            prompt=prompt,
            messages=messages,
            **params,
        )

    def stream(
        self,
        *,
        prompt: Any = None,
        messages: Any = None,
        **overrides: Any,
    ) -> StreamTextResult:
        """Stream text with the agent's defaults (multi-step by default).

        Returns a ``StreamTextResult`` immediately; work begins on first
        iteration/await, identical to ``stream_text``.

        Parameters
        ----------
        prompt:
            A plain-text prompt string (mutually exclusive with *messages*).
        messages:
            Conversation history (mutually exclusive with *prompt*).
        **overrides:
            Any constructor parameter can be overridden per-call; per-call
            value wins when not ``None``.
        """
        params = self._merge(overrides)
        return stream_text(
            prompt=prompt,
            messages=messages,
            **params,
        )
