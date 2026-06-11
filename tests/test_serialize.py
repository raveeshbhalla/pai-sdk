"""Lossless trace round-trips + the typed-message (DSPy-signature-style) pattern."""

import json
from typing import Any

import pytest
from pydantic import model_validator

from model_message import (
    AssistantModelMessage,
    SystemModelMessage,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    JsonOutput,
    UserModelMessage,
    generate_text,
)
from model_message.serialize import dump_messages, dump_messages_json, load_messages

from conftest import FakeModel, text_step


def test_round_trip_full_conversation():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        AssistantModelMessage(
            content=[
                TextPart(text="checking"),
                ToolCallPart(tool_call_id="c1", tool_name="lookup", input={"q": "x"}),
            ]
        ),
        ToolModelMessage(
            content=[
                ToolResultPart(
                    tool_call_id="c1", tool_name="lookup", output=JsonOutput(value={"a": 1})
                )
            ]
        ),
    ]
    dumped = dump_messages(messages)
    assert dumped[2]["content"][1]["toolCallId"] == "c1"  # camelCase wire format
    text = dump_messages_json(messages)
    restored = load_messages(text)
    assert dump_messages(restored) == dumped  # stable fixed point
    assert restored[3].content[0].output.value == {"a": 1}


def test_bytes_become_base64_json_safe():
    messages = [
        UserModelMessage(
            content=[
                {"type": "text", "text": "look"},
                {"type": "image", "image": b"\x89PNG\r\n\x1a\nx"},
            ]
        )
    ]
    text = dump_messages_json(messages)
    json.loads(text)  # must be pure JSON
    restored = load_messages(text)
    assert restored[0].content[1].type == "image"


async def test_replay_loaded_messages_through_generate_text():
    model = FakeModel(steps=[text_step("ok")])
    history = load_messages(
        dump_messages_json(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ]
        )
    )
    result = await generate_text(model=model, messages=history)
    assert result.text == "ok"
    assert model.calls[0].prompt[0].role == "system"


# ---------------------------------------------------------------------------
# The TypedSystemMessage pattern (DSPy-signature-style structured logging):
# subclass a message type, carry the template + variable bindings as real
# fields, render `content` automatically. Flows through generate_text
# unchanged (providers only read role/content) and round-trips through
# dump/load with the structure intact — which is what a GEPA-style optimizer
# needs to separate instructions from variables in traces.
# ---------------------------------------------------------------------------


class TypedSystemMessage(SystemModelMessage):
    template: str
    variables: dict[str, Any]
    content: str = ""

    @model_validator(mode="after")
    def _render(self) -> "TypedSystemMessage":
        if not self.content:
            self.content = self.template.format(**self.variables)
        return self


async def test_typed_system_message_flows_through_engine():
    model = FakeModel(steps=[text_step("ok")])
    msg = TypedSystemMessage(
        template="You answer questions about {topic} for {audience}.",
        variables={"topic": "tax law", "audience": "engineers"},
    )
    await generate_text(model=model, messages=[msg, {"role": "user", "content": "hi"}])
    sent = model.calls[0].prompt[0]
    # the provider sees a plain system message with rendered content
    assert sent.role == "system"
    assert sent.content == "You answer questions about tax law for engineers."


def test_typed_system_message_survives_trace_round_trip():
    msg = TypedSystemMessage(
        template="You answer questions about {topic} for {audience}.",
        variables={"topic": "tax law", "audience": "engineers"},
    )
    dumped = dump_messages([msg])
    # structured fields are preserved in the trace, alongside rendered content
    assert dumped[0]["template"] == "You answer questions about {topic} for {audience}."
    assert dumped[0]["variables"] == {"topic": "tax law", "audience": "engineers"}
    assert dumped[0]["content"].startswith("You answer questions about tax law")

    # generic load keeps the extra fields (extra="allow") on the base type...
    restored = load_messages(dumped)
    assert restored[0].content.startswith("You answer questions about tax law")
    assert restored[0].model_dump()["template"] == msg.template
    # ...and the typed layer can re-validate into the typed class to mutate
    # instructions and re-render (the GEPA loop):
    typed = TypedSystemMessage.model_validate(dumped[0])
    evolved = TypedSystemMessage(
        template=typed.template.replace("answer questions", "give expert answers"),
        variables=typed.variables,
    )
    assert evolved.content == "You give expert answers about tax law for engineers."
