"""Result types for generate_text/stream_text (AI SDK GenerateTextResult et al.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, Union

from .messages import (
    AssistantContentPart,
    ModelMessage,
    ReasoningPart,
    ToolCallPart,
    ToolResultOutput,
)

FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other", "unknown"
]


@dataclass
class Usage:
    """Token usage (AI SDK LanguageModelUsage)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None

    def __add__(self, other: "Usage") -> "Usage":
        def add(a: Optional[int], b: Optional[int]) -> Optional[int]:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=add(self.input_tokens, other.input_tokens),
            output_tokens=add(self.output_tokens, other.output_tokens),
            total_tokens=add(self.total_tokens, other.total_tokens),
            reasoning_tokens=add(self.reasoning_tokens, other.reasoning_tokens),
            cached_input_tokens=add(self.cached_input_tokens, other.cached_input_tokens),
        )


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
    warnings: list[str]
    response: ResponseMetadata
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None


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
    warnings: list[str]
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None
    # Parsed structured output when an object output spec was given (AI SDK
    # experimental_output). None otherwise.
    output: Any = None


ToolChoice = Union[
    Literal["auto", "none", "required"],
    dict[str, Any],  # {"type": "tool", "tool_name": "..."}
]
