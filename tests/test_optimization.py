from __future__ import annotations

import pytest

from pai_sdk import (
    OptimizerTarget,
    PromptError,
    apply_optimizer_target,
    list_optimizer_targets,
    load_prompt,
    read_optimizer_target,
    system_instruction_target,
)


def test_optimizer_targets_are_selected_at_runtime():
    prompt = load_prompt(
        {
            "name": "runtime-targets",
            "messages": [
                {
                    "id": "system",
                    "role": "system",
                    "template": "Help {{company}} users.",
                },
                {
                    "id": "ticket",
                    "role": "user",
                    "template": "Ticket: {{ticket}}",
                },
            ],
            "tools": {
                "lookup": {
                    "description": "Look up customer data.",
                    "input": {"q": "string"},
                }
            },
        }
    )

    assert list_optimizer_targets(prompt) == [
        OptimizerTarget.message_template("system"),
        OptimizerTarget.message_template("ticket"),
        OptimizerTarget.tool_description("lookup"),
    ]
    assert read_optimizer_target(prompt, OptimizerTarget.message_template("system")) == (
        "Help {{company}} users."
    )
    assert read_optimizer_target(prompt, OptimizerTarget.tool_description("lookup")) == (
        "Look up customer data."
    )

    evolved = apply_optimizer_target(
        prompt,
        OptimizerTarget.message_template("system"),
        "Help {{company}} users with concise answers.",
    )
    assert evolved.messages[0].template.endswith("concise answers.")

    evolved = apply_optimizer_target(
        prompt,
        OptimizerTarget.tool_description("lookup"),
        "Look up customer context only when needed.",
    )
    assert evolved.tools["lookup"].description == "Look up customer context only when needed."

    with pytest.raises(PromptError):
        apply_optimizer_target(
            prompt,
            OptimizerTarget.message_template("system"),
            "Help everyone.",
        )
    with pytest.raises(PromptError, match="No message"):
        read_optimizer_target(prompt, OptimizerTarget.message_template("missing"))


def test_system_instruction_target_requires_disambiguation():
    prompt = load_prompt(
        {
            "name": "system-target",
            "messages": [
                {
                    "id": "instructions",
                    "role": "system",
                    "template": "Help {{company}} users.",
                },
                {
                    "id": "policy",
                    "role": "system",
                    "template": "Follow {{company}} policy.",
                },
                {"id": "user", "role": "user", "template": "{{ticket}}"},
            ],
        }
    )

    with pytest.raises(PromptError):
        system_instruction_target(prompt)

    assert system_instruction_target(prompt, message_id="instructions") == (
        OptimizerTarget.message_template("instructions")
    )
