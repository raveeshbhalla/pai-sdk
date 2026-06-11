"""Lossless message serialization for logs and traces.

ModelMessage is designed to be the trace schema: dump a conversation (including
result.response.messages) to JSON, store it, and load it back into typed
messages that can be re-sent verbatim. The wire format is camelCase and
compatible with the TypeScript AI SDK.

Two properties make this safe for structured logging layers built on top
(e.g. DSPy-signature-style typed messages):

1. Binary content serializes as base64, so dumps are always JSON-able.
2. Message models allow extra fields (`extra="allow"`), so subclasses that
   carry structured metadata — a prompt template, variable bindings, a
   signature id — round-trip through dump/load with that metadata intact,
   while providers only ever read the standard fields (role/content/...).
"""

from __future__ import annotations

import json
from typing import Any, Sequence, Union

from .messages import ModelMessage, model_messages_adapter


def dump_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize messages (typed or dicts) to JSON-able dicts (camelCase wire
    format, AI SDK compatible). Bytes become base64; None fields are omitted."""
    validated = model_messages_adapter.validate_python(list(messages))
    return [
        message.model_dump(by_alias=True, exclude_none=True, mode="json")
        for message in validated
    ]


def dump_messages_json(messages: Sequence[Any], *, indent: int | None = None) -> str:
    """Serialize messages to a JSON string."""
    return json.dumps(dump_messages(messages), indent=indent, ensure_ascii=False)


def load_messages(data: Union[str, Sequence[Any]]) -> list[ModelMessage]:
    """Load messages from a JSON string or a list of dicts back into typed
    ModelMessage instances, ready to pass to generate_text(messages=...)."""
    if isinstance(data, str):
        data = json.loads(data)
    return model_messages_adapter.validate_python(list(data))
