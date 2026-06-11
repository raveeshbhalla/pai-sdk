"""Result types for generate_text/stream_text (AI SDK GenerateTextResult et al.)."""

from __future__ import annotations

import base64 as _base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, Union

from .messages import (
    AssistantContentPart,
    ModelMessage,
    ReasoningPart,
    SourcePart,
    ToolCallPart,
    ToolResultOutput,
)

FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other", "unknown"
]


def _add_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


@dataclass
class InputTokenDetails:
    """Breakdown of input tokens (AI SDK v6)."""

    no_cache_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None

    def __add__(self, other: "InputTokenDetails") -> "InputTokenDetails":
        return InputTokenDetails(
            no_cache_tokens=_add_opt(self.no_cache_tokens, other.no_cache_tokens),
            cache_read_tokens=_add_opt(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_add_opt(self.cache_write_tokens, other.cache_write_tokens),
        )


@dataclass
class OutputTokenDetails:
    """Breakdown of output tokens (AI SDK v6)."""

    text_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def __add__(self, other: "OutputTokenDetails") -> "OutputTokenDetails":
        return OutputTokenDetails(
            text_tokens=_add_opt(self.text_tokens, other.text_tokens),
            reasoning_tokens=_add_opt(self.reasoning_tokens, other.reasoning_tokens),
        )


def _add_details(a: Any, b: Any) -> Any:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b


@dataclass
class Usage:
    """Token usage (AI SDK LanguageModelUsage)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    input_token_details: Optional[InputTokenDetails] = None
    output_token_details: Optional[OutputTokenDetails] = None

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=_add_opt(self.input_tokens, other.input_tokens),
            output_tokens=_add_opt(self.output_tokens, other.output_tokens),
            total_tokens=_add_opt(self.total_tokens, other.total_tokens),
            reasoning_tokens=_add_opt(self.reasoning_tokens, other.reasoning_tokens),
            cached_input_tokens=_add_opt(
                self.cached_input_tokens, other.cached_input_tokens
            ),
            input_token_details=_add_details(
                self.input_token_details, other.input_token_details
            ),
            output_token_details=_add_details(
                self.output_token_details, other.output_token_details
            ),
        )


@dataclass
class CallWarning:
    """A structured warning from a provider (AI SDK CallWarning)."""

    type: Literal["unsupported-setting", "unsupported-tool", "other"]
    setting: Optional[str] = None
    message: Optional[str] = None

    @staticmethod
    def coerce(value: Any) -> "CallWarning":
        """Accept a CallWarning, a plain string, or a dict and normalize it."""
        if isinstance(value, CallWarning):
            return value
        if isinstance(value, str):
            return CallWarning(type="other", message=value)
        if isinstance(value, dict):
            return CallWarning(
                type=value.get("type", "other"),
                setting=value.get("setting"),
                message=value.get("message"),
            )
        return CallWarning(type="other", message=str(value))


def coerce_warnings(values: Optional[list[Any]]) -> list["CallWarning"]:
    """Normalize a list of warnings (strings/dicts/CallWarning) to CallWarning."""
    if not values:
        return []
    return [CallWarning.coerce(v) for v in values]


@dataclass
class GeneratedFile:
    """A file produced by the model (AI SDK GeneratedFile)."""

    data: bytes
    media_type: str

    @property
    def base64(self) -> str:
        return _base64.b64encode(self.data).decode()

    @property
    def bytes(self) -> bytes:
        """The raw bytes (AI SDK `.uint8Array` analog)."""
        return self.data


@dataclass
class ResponseMetadata:
    """Provider response metadata."""

    id: Optional[str] = None
    model_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    headers: Optional[dict[str, str]] = None
    body: Any = None
    # Assistant/tool messages generated during the call, ready to append to
    # your conversation history (AI SDK response.messages).
    messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class ToolResult:
    """A completed tool execution (AI SDK TypedToolResult)."""

    tool_call_id: str
    tool_name: str
    input: Any
    output: Any
    model_output: Optional[ToolResultOutput] = None
    is_error: bool = False
    provider_executed: bool = False


@dataclass
class StepResult:
    """The result of a single generation step within the tool loop."""

    content: list[AssistantContentPart]
    text: str
    reasoning: list[ReasoningPart]
    reasoning_text: Optional[str]
    tool_calls: list[ToolCallPart]
    tool_results: list[ToolResult]
    finish_reason: FinishReason
    raw_finish_reason: Optional[str]
    usage: Usage
    warnings: list[CallWarning]
    response: ResponseMetadata
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None
    sources: list[SourcePart] = field(default_factory=list)
    files: list[GeneratedFile] = field(default_factory=list)
    request: Any = None


@dataclass
class GenerateTextResult:
    """The result of generate_text() (final step + aggregates)."""

    text: str
    content: list[AssistantContentPart]
    reasoning: list[ReasoningPart]
    reasoning_text: Optional[str]
    tool_calls: list[ToolCallPart]
    tool_results: list[ToolResult]
    finish_reason: FinishReason
    raw_finish_reason: Optional[str]
    usage: Usage
    total_usage: Usage
    steps: list[StepResult]
    response: ResponseMetadata
    warnings: list[CallWarning]
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None
    # Parsed structured output when an object output spec was given (AI SDK
    # experimental_output). None otherwise.
    output: Any = None
    sources: list[SourcePart] = field(default_factory=list)
    files: list[GeneratedFile] = field(default_factory=list)
    request: Any = None


ToolChoice = Union[
    Literal["auto", "none", "required"],
    dict[str, Any],  # {"type": "tool", "tool_name": "..."}
]
