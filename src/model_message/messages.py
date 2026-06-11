"""ModelMessage types — a Python port of the AI SDK's ModelMessage union.

Field names are snake_case in Python and camelCase on the wire (JSON aliases),
so serialized messages are interchangeable with AI SDK (TypeScript) messages.

Discriminators match AI SDK exactly: `role` on messages, `type` on parts.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer
from pydantic.alias_generators import to_camel

# providerOptions / providerMetadata: outer key is the provider name
# (e.g. "anthropic", "openai"), value is an arbitrary JSON object.
ProviderOptions = dict[str, dict[str, Any]]

# Raw bytes, a base64-encoded string, a data: URL, or an http(s) URL.
DataContent = Union[bytes, str]


class _Part(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="allow",
    )

    provider_options: Optional[ProviderOptions] = None


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


class TextPart(_Part):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Part):
    """An image in a user message.

    `image` accepts raw bytes, a base64 string, a data: URL, or an http(s) URL.
    `media_type` is optional and detected when omitted.
    """

    type: Literal["image"] = "image"
    image: DataContent
    media_type: Optional[str] = None

    @field_serializer("image")
    def _serialize_image(self, value: DataContent) -> str:
        return base64.b64encode(value).decode() if isinstance(value, bytes) else value


class FilePart(_Part):
    """A file (PDF, audio, arbitrary document, or image) in a message."""

    type: Literal["file"] = "file"
    data: DataContent
    media_type: str
    filename: Optional[str] = None

    @field_serializer("data")
    def _serialize_data(self, value: DataContent) -> str:
        return base64.b64encode(value).decode() if isinstance(value, bytes) else value


class ReasoningPart(_Part):
    """Reasoning/thinking text produced by the model (assistant messages)."""

    type: Literal["reasoning"] = "reasoning"
    text: str


class ToolCallPart(_Part):
    """A tool call requested by the model (assistant messages)."""

    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str
    tool_name: str
    input: Any = None
    provider_executed: Optional[bool] = None


# --- ToolResultOutput union -------------------------------------------------


class TextOutput(_Part):
    type: Literal["text"] = "text"
    value: str


class JsonOutput(_Part):
    type: Literal["json"] = "json"
    value: Any = None


class ErrorTextOutput(_Part):
    type: Literal["error-text"] = "error-text"
    value: str


class ErrorJsonOutput(_Part):
    type: Literal["error-json"] = "error-json"
    value: Any = None


class TextContentItem(_Part):
    type: Literal["text"] = "text"
    text: str


class MediaContentItem(_Part):
    """Binary content (e.g. an image) returned by a tool."""

    type: Literal["media"] = "media"
    data: DataContent  # raw bytes or base64 string
    media_type: str

    @field_serializer("data")
    def _serialize_data(self, value: DataContent) -> str:
        return base64.b64encode(value).decode() if isinstance(value, bytes) else value


ContentOutputItem = Annotated[
    Union[TextContentItem, MediaContentItem], Field(discriminator="type")
]


class ContentOutput(_Part):
    type: Literal["content"] = "content"
    value: list[ContentOutputItem]


ToolResultOutput = Annotated[
    Union[TextOutput, JsonOutput, ErrorTextOutput, ErrorJsonOutput, ContentOutput],
    Field(discriminator="type"),
]


class ToolResultPart(_Part):
    """The result of a tool call (tool messages, or assistant messages for
    provider-executed tools)."""

    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str
    tool_name: str
    output: ToolResultOutput
    provider_executed: Optional[bool] = None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

UserContentPart = Annotated[
    Union[TextPart, ImagePart, FilePart], Field(discriminator="type")
]
UserContent = Union[str, list[UserContentPart]]

AssistantContentPart = Annotated[
    Union[TextPart, FilePart, ReasoningPart, ToolCallPart, ToolResultPart],
    Field(discriminator="type"),
]
AssistantContent = Union[str, list[AssistantContentPart]]

ToolContent = list[ToolResultPart]


class _Message(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="allow",
    )

    provider_options: Optional[ProviderOptions] = None


class SystemModelMessage(_Message):
    role: Literal["system"] = "system"
    content: str


class UserModelMessage(_Message):
    role: Literal["user"] = "user"
    content: UserContent


class AssistantModelMessage(_Message):
    role: Literal["assistant"] = "assistant"
    content: AssistantContent


class ToolModelMessage(_Message):
    role: Literal["tool"] = "tool"
    content: ToolContent


ModelMessage = Annotated[
    Union[
        SystemModelMessage,
        UserModelMessage,
        AssistantModelMessage,
        ToolModelMessage,
    ],
    Field(discriminator="role"),
]

model_message_adapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)
model_messages_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(
    list[ModelMessage]
)
