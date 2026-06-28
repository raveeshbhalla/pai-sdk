"""Small helpers for optimizer scripts that mutate prompt configs.

Target selection belongs to the optimizer run, not the prompt YAML. These
helpers give scripts a typed way to name the text they chose to optimize and
apply a candidate while reusing Prompt's structural safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .prompts import Prompt, PromptError

OptimizerTargetKind = Literal["message_template", "tool_description"]


@dataclass(frozen=True)
class OptimizerTarget:
    """A run-time prompt text target selected by an optimizer script."""

    kind: OptimizerTargetKind
    id: str

    @classmethod
    def message_template(cls, message_id: str) -> "OptimizerTarget":
        return cls(kind="message_template", id=message_id)

    @classmethod
    def tool_description(cls, tool_name: str) -> "OptimizerTarget":
        return cls(kind="tool_description", id=tool_name)


def list_optimizer_targets(prompt: Prompt) -> list[OptimizerTarget]:
    """Return selectable text targets without marking any as optimizable."""

    targets = [
        OptimizerTarget.message_template(message.id)
        for message in prompt.messages
        if message.id is not None and message.template is not None
    ]
    targets.extend(
        OptimizerTarget.tool_description(name)
        for name, tool in prompt.tools.items()
        if tool.description is not None
    )
    return targets


def system_instruction_target(
    prompt: Prompt,
    *,
    message_id: Optional[str] = None,
) -> OptimizerTarget:
    """Select a system-template message as a run-time optimizer target.

    If a prompt has exactly one system template, it is selected automatically.
    Prompts with multiple system templates require an explicit message id so a
    runner cannot accidentally optimize policy/context text.
    """

    candidates = [
        message
        for message in prompt.messages
        if message.role == "system" and message.template is not None
    ]
    if message_id is not None:
        message = next((item for item in candidates if item.id == message_id), None)
        if message is None:
            raise PromptError(
                f"No system template message with id '{message_id}' to optimize."
            )
        return OptimizerTarget.message_template(message_id)
    if len(candidates) != 1:
        ids = ", ".join(message.id or "(missing id)" for message in candidates)
        raise PromptError(
            "Expected exactly one system template message; pass message_id= to "
            f"choose one explicitly. Candidates: {ids or '(none)'}."
        )
    message = candidates[0]
    if message.id is None:
        raise PromptError("System template messages need an id for optimization.")
    return OptimizerTarget.message_template(message.id)


def apply_optimizer_target(
    prompt: Prompt,
    target: OptimizerTarget,
    candidate_text: str,
) -> Prompt:
    """Apply an optimizer candidate to a selected target."""

    if target.kind == "message_template":
        return prompt.with_template(target.id, candidate_text)
    if target.kind == "tool_description":
        return prompt.with_tool_description(target.id, candidate_text)
    raise ValueError(f"Unknown optimizer target kind: {target.kind!r}")
