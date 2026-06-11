"""Prompt normalization: system/prompt/messages -> list[ModelMessage]."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .errors import InvalidPromptError
from .messages import (
    AssistantModelMessage,
    ModelMessage,
    SystemModelMessage,
    ToolModelMessage,
    UserModelMessage,
    model_message_adapter,
)

_MESSAGE_TYPES = (
    SystemModelMessage,
    UserModelMessage,
    AssistantModelMessage,
    ToolModelMessage,
)

Prompt = Union[str, Sequence[Any], None]


def standardize_prompt(
    *,
    system: Optional[str] = None,
    prompt: Prompt = None,
    messages: Optional[Sequence[Any]] = None,
) -> list[ModelMessage]:
    """Validate and normalize into a list of ModelMessage.

    - `prompt` and `messages` are mutually exclusive.
    - `prompt` may be a string (becomes a single user message) or a message list.
    - `system` is prepended as a SystemModelMessage.
    - Message lists accept ModelMessage instances or plain dicts
      (e.g. {"role": "user", "content": "hi"}).
    """
    if prompt is not None and messages is not None:
        raise InvalidPromptError("Provide either `prompt` or `messages`, not both.")
    if prompt is None and messages is None:
        raise InvalidPromptError("Provide `prompt` or `messages`.")

    result: list[ModelMessage] = []
    if system is not None:
        result.append(SystemModelMessage(content=system))

    source = prompt if prompt is not None else messages
    if isinstance(source, str):
        result.append(UserModelMessage(content=source))
        return result

    assert source is not None
    for item in source:
        if isinstance(item, _MESSAGE_TYPES):
            result.append(item)
        else:
            result.append(model_message_adapter.validate_python(item))
    return result
