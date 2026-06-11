"""Stream part types for stream_text — the AI SDK TextStreamPart union.

Text/reasoning stream as start -> delta(s) -> end blocks tied by `id`;
tool inputs stream as tool-input-start -> tool-input-delta -> tool-input-end,
followed by a complete `tool-call` part, then `tool-result`/`tool-error`.
Steps are bracketed by start-step/finish-step; the whole run by start/finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from .messages import SourcePart, ToolCallPart, ToolResultOutput
from .results import FinishReason, ResponseMetadata, Usage


@dataclass
class StreamStart:
    type: Literal["start"] = "start"


@dataclass
class StartStep:
    request: Any = None
    warnings: list[str] = field(default_factory=list)
    type: Literal["start-step"] = "start-step"


@dataclass
class TextStart:
    id: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["text-start"] = "text-start"


@dataclass
class TextDelta:
    id: str
    text: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["text-delta"] = "text-delta"


@dataclass
class TextEnd:
    id: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["text-end"] = "text-end"


@dataclass
class ReasoningStart:
    id: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["reasoning-start"] = "reasoning-start"


@dataclass
class ReasoningDelta:
    id: str
    text: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["reasoning-delta"] = "reasoning-delta"


@dataclass
class ReasoningEnd:
    id: str
    provider_metadata: Optional[dict[str, Any]] = None
    type: Literal["reasoning-end"] = "reasoning-end"


@dataclass
class ToolInputStart:
    id: str  # tool call id
    tool_name: str
    provider_executed: Optional[bool] = None
    type: Literal["tool-input-start"] = "tool-input-start"


@dataclass
class ToolInputDelta:
    id: str
    delta: str
    type: Literal["tool-input-delta"] = "tool-input-delta"


@dataclass
class ToolInputEnd:
    id: str
    type: Literal["tool-input-end"] = "tool-input-end"


# Complete tool calls are emitted as ToolCallPart (type == "tool-call").


@dataclass
class ToolResultEvent:
    tool_call_id: str
    tool_name: str
    input: Any
    output: Any
    model_output: Optional[ToolResultOutput] = None
    provider_executed: Optional[bool] = None
    type: Literal["tool-result"] = "tool-result"


@dataclass
class SourceStreamPart:
    """A source citation surfaced mid-stream (AI SDK source part).

    Carries an already-validated UrlSourcePart/DocumentSourcePart in `source`.
    """

    source: "SourcePart"
    type: Literal["source"] = "source"


@dataclass
class ToolErrorEvent:
    tool_call_id: str
    tool_name: str
    input: Any
    error: Any
    type: Literal["tool-error"] = "tool-error"


@dataclass
class FilePartEvent:
    media_type: str
    data: bytes
    type: Literal["file"] = "file"


@dataclass
class ResponseMetadataPart:
    """Provider-internal: response id/model/timestamp, surfaced mid-stream.

    Adapters may attach the JSON-able `request` body they built; the stream
    loop copies it onto the step and FinishStep.
    """

    id: Optional[str] = None
    model_id: Optional[str] = None
    request: Any = None
    type: Literal["response-metadata"] = "response-metadata"


@dataclass
class FinishStep:
    response: ResponseMetadata
    usage: Usage
    finish_reason: FinishReason
    raw_finish_reason: Optional[str] = None
    provider_metadata: Optional[dict[str, Any]] = None
    request: Any = None
    type: Literal["finish-step"] = "finish-step"


@dataclass
class Finish:
    finish_reason: FinishReason
    total_usage: Usage
    raw_finish_reason: Optional[str] = None
    type: Literal["finish"] = "finish"


@dataclass
class ErrorPart:
    error: Any
    type: Literal["error"] = "error"


@dataclass
class RawPart:
    raw_value: Any
    type: Literal["raw"] = "raw"


TextStreamPart = Union[
    StreamStart,
    StartStep,
    TextStart,
    TextDelta,
    TextEnd,
    ReasoningStart,
    ReasoningDelta,
    ReasoningEnd,
    ToolInputStart,
    ToolInputDelta,
    ToolInputEnd,
    ToolCallPart,
    ToolResultEvent,
    ToolErrorEvent,
    FilePartEvent,
    SourceStreamPart,
    FinishStep,
    Finish,
    ErrorPart,
    RawPart,
]

# Parts a provider's do_stream may yield (subset + response metadata; the
# engine adds start/start-step/tool-result/finish-step/finish framing).
ProviderStreamPart = Union[
    TextStart,
    TextDelta,
    TextEnd,
    ReasoningStart,
    ReasoningDelta,
    ReasoningEnd,
    ToolInputStart,
    ToolInputDelta,
    ToolInputEnd,
    ToolCallPart,
    FilePartEvent,
    SourceStreamPart,
    ToolResultEvent,
    ResponseMetadataPart,
    Finish,
    ErrorPart,
    RawPart,
]
