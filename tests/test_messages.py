import base64

import pytest

from pai_sdk import (
    AssistantModelMessage,
    ImagePart,
    SystemModelMessage,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    UserModelMessage,
    pai_sdk_adapter,
    pai_sdks_adapter,
)
from pai_sdk._prompt import standardize_prompt
from pai_sdk.errors import InvalidPromptError


def test_dict_messages_validate():
    msg = pai_sdk_adapter.validate_python(
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    )
    assert isinstance(msg, UserModelMessage)
    assert isinstance(msg.content[0], TextPart)


def test_camel_case_wire_format_round_trip():
    """Serialized JSON must match the AI SDK wire format (camelCase)."""
    msg = AssistantModelMessage(
        content=[
            ToolCallPart(tool_call_id="call_1", tool_name="get_weather", input={"city": "Paris"})
        ]
    )
    dumped = msg.model_dump(by_alias=True, exclude_none=True)
    part = dumped["content"][0]
    assert part == {
        "type": "tool-call",
        "toolCallId": "call_1",
        "toolName": "get_weather",
        "input": {"city": "Paris"},
    }
    # round-trip back from camelCase
    restored = pai_sdk_adapter.validate_python(dumped)
    assert restored.content[0].tool_call_id == "call_1"


def test_tool_result_output_union():
    msg = pai_sdk_adapter.validate_python(
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "call_1",
                    "toolName": "t",
                    "output": {"type": "json", "value": {"a": 1}},
                }
            ],
        }
    )
    assert isinstance(msg, ToolModelMessage)
    assert msg.content[0].output.type == "json"
    assert msg.content[0].output.value == {"a": 1}


def test_image_bytes_serialize_as_base64():
    raw = b"\x89PNG\r\n\x1a\nrest"
    part = ImagePart(image=raw)
    dumped = part.model_dump(by_alias=True, exclude_none=True)
    assert dumped["image"] == base64.b64encode(raw).decode()


def test_standardize_prompt_string():
    messages = standardize_prompt(system="be terse", prompt="hello")
    assert isinstance(messages[0], SystemModelMessage)
    assert isinstance(messages[1], UserModelMessage)
    assert messages[1].content == "hello"


def test_standardize_prompt_exclusive():
    with pytest.raises(InvalidPromptError):
        standardize_prompt(prompt="a", messages=[{"role": "user", "content": "b"}])
    with pytest.raises(InvalidPromptError):
        standardize_prompt()


def test_standardize_messages_mixed():
    messages = standardize_prompt(
        messages=[
            UserModelMessage(content="hi"),
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert len(messages) == 2
    assert messages[1].role == "assistant"


def test_messages_list_adapter():
    messages = pai_sdks_adapter.validate_python(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
    )
    assert [m.role for m in messages] == ["system", "user", "assistant"]
