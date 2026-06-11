"""ModelMessage types — a Python port of the AI SDK's ModelMessage union.

Field names are snake_case in Python and camelCase on the wire (JSON aliases),
so serialized messages are interchangeable with AI SDK (TypeScript) messages.

Discriminators match AI SDK exactly: `role` on messages, `type` on parts.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_serializer,
)
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
# File references (provider file ids — AI SDK FileData reference variant)
# ---------------------------------------------------------------------------


class FileIdData(BaseModel):
    """A reference to a provider-hosted file by id (AI SDK file-id variant).

    Used in place of inline bytes for FilePart.data / ImagePart.image when a
    provider exposes uploaded files by id. The `id` may be a plain string or a
    small mapping of provider-specific fields.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="allow",
    )

    type: Literal["file-id"] = "file-id"
    id: Union[str, dict[str, str]]


# Inline data (bytes / base64 / data: URL / http URL) or a provider file ref.
DataOrFileId = Union[bytes, str, FileIdData]


def _serialize_data_content(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, FileIdData):
        return value.model_dump(by_alias=True, exclude_none=True)
    return value


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
    image: DataOrFileId
    media_type: Optional[str] = None

    @field_serializer("image")
    def _serialize_image(self, value: DataOrFileId) -> Any:
        return _serialize_data_content(value)


class FilePart(_Part):
    """A file (PDF, audio, arbitrary document, or image) in a message."""

    type: Literal["file"] = "file"
    data: DataOrFileId
    media_type: str
    filename: Optional[str] = None

    @field_serializer("data")
    def _serialize_data(self, value: DataOrFileId) -> Any:
        return _serialize_data_content(value)


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


# --- v6 richer tool-result content items ------------------------------------


class FileDataContentItem(_Part):
    """Inline file data returned by a tool (AI SDK v6 file-data item)."""

    type: Literal["file-data"] = "file-data"
    data: DataContent  # raw bytes or base64 string
    media_type: str
    filename: Optional[str] = None

    @field_serializer("data")
    def _serialize_data(self, value: DataContent) -> str:
        return base64.b64encode(value).decode() if isinstance(value, bytes) else value


class FileUrlContentItem(_Part):
    """A file referenced by URL returned by a tool (AI SDK v6 file-url item)."""

    type: Literal["file-url"] = "file-url"
    url: str


class ImageDataContentItem(_Part):
    """Inline image data returned by a tool (AI SDK v6 image-data item)."""

    type: Literal["image-data"] = "image-data"
    data: DataContent  # raw bytes or base64 string
    media_type: str

    @field_serializer("data")
    def _serialize_data(self, value: DataContent) -> str:
        return base64.b64encode(value).decode() if isinstance(value, bytes) else value


class ImageUrlContentItem(_Part):
    """An image referenced by URL returned by a tool (AI SDK v6 image-url item)."""

    type: Literal["image-url"] = "image-url"
    url: str


ContentOutputItem = Annotated[
    Union[
        TextContentItem,
        MediaContentItem,
        FileDataContentItem,
        FileUrlContentItem,
        ImageDataContentItem,
        ImageUrlContentItem,
    ],
    Field(discriminator="type"),
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
# Sources (AI SDK Source union)
# ---------------------------------------------------------------------------


class UrlSourcePart(_Part):
    """A URL source cited by the model (AI SDK url source)."""

    type: Literal["source"] = "source"
    source_type: Literal["url"] = "url"
    id: str
    url: str
    title: Optional[str] = None
    provider_metadata: Optional[ProviderOptions] = None


class DocumentSourcePart(_Part):
    """A document source cited by the model (AI SDK document source)."""

    type: Literal["source"] = "source"
    source_type: Literal["document"] = "document"
    id: str
    media_type: str
    title: str
    filename: Optional[str] = None
    provider_metadata: Optional[ProviderOptions] = None


SourcePart = Annotated[
    Union[UrlSourcePart, DocumentSourcePart], Field(discriminator="source_type")
]


# ---------------------------------------------------------------------------
# Tool approvals (AI SDK v6 — types only)
# ---------------------------------------------------------------------------


class ToolApprovalRequest(_Part):
    """A request for approval before executing a tool (assistant content)."""

    type: Literal["tool-approval-request"] = "tool-approval-request"
    approval_id: str
    tool_call_id: str
    is_automatic: Optional[bool] = None


class ToolApprovalResponse(_Part):
    """A response to a tool approval request (tool message content)."""

    type: Literal["tool-approval-response"] = "tool-approval-response"
    approval_id: str
    approved: bool
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

UserContentPart = Annotated[
    Union[TextPart, ImagePart, FilePart], Field(discriminator="type")
]
UserContent = Union[str, list[UserContentPart]]

def _assistant_part_discriminator(value: Any) -> str:
    """Discriminate assistant content. `source` parts are further split by
    `sourceType` so url/document sources can coexist under one `type`."""
    if isinstance(value, BaseModel):
        type_ = getattr(value, "type", None)
        source_type = getattr(value, "source_type", None)
    else:
        type_ = value.get("type")
        source_type = value.get("sourceType", value.get("source_type"))
    if type_ == "source":
        return f"source:{source_type}"
    return type_


AssistantContentPart = Annotated[
    Union[
        Annotated[TextPart, Tag("text")],
        Annotated[FilePart, Tag("file")],
        Annotated[ReasoningPart, Tag("reasoning")],
        Annotated[ToolCallPart, Tag("tool-call")],
        Annotated[ToolResultPart, Tag("tool-result")],
        Annotated[UrlSourcePart, Tag("source:url")],
        Annotated[DocumentSourcePart, Tag("source:document")],
        Annotated[ToolApprovalRequest, Tag("tool-approval-request")],
    ],
    Discriminator(_assistant_part_discriminator),
]
AssistantContent = Union[str, list[AssistantContentPart]]

ToolContentPart = Annotated[
    Union[ToolResultPart, ToolApprovalResponse], Field(discriminator="type")
]
ToolContent = list[ToolContentPart]


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

pai_sdk_adapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)
pai_sdks_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(
    list[ModelMessage]
)
