"""The provider interface — the Python analog of LanguageModelV3.

generate_text/stream_text normalize user input into CallOptions, then call
the model's do_generate/do_stream. Provider adapters translate CallOptions
to each provider's wire format and translate responses back into the shared
content-part types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .messages import AssistantContentPart, ModelMessage
from .results import FinishReason, ResponseMetadata, ToolChoice, Usage
from .stream import ProviderStreamPart


@dataclass
class FunctionToolSpec:
    """A normalized function tool definition sent to providers."""

    name: str
    description: Optional[str]
    input_schema: dict[str, Any]
    strict: Optional[bool] = None
    provider_options: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class CallOptions:
    """Normalized call options handed to a provider (LanguageModelV3CallOptions)."""

    prompt: list[ModelMessage]
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    seed: Optional[int] = None
    tools: list[FunctionToolSpec] = field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None
    response_format: Optional[dict[str, Any]] = None  # {"type": "json", "schema": ...}
    headers: Optional[dict[str, str]] = None
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """What a provider's do_generate returns."""

    content: list[AssistantContentPart]
    finish_reason: FinishReason
    usage: Usage
    raw_finish_reason: Optional[str] = None
    response: ResponseMetadata = field(default_factory=ResponseMetadata)
    warnings: list[str] = field(default_factory=list)
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None


class LanguageModel(ABC):
    """A language model bound to one provider API + model id."""

    provider: str
    model_id: str

    @abstractmethod
    async def do_generate(self, options: CallOptions) -> ProviderResult:
        """Run a single non-streaming generation."""

    @abstractmethod
    def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
        """Run a single streaming generation, yielding provider stream parts.

        Must end with a Finish part carrying finish_reason and usage.
        """
