"""Helpers for external optimizer scripts that mutate prompt documents.

pai-sdk does not ship or depend on an optimizer (no GEPA, no LiteLLM). An
external runner — e.g. a script built on GEPA's `optimize_anything` — owns the
dataset, the search loop, and the choice of targets. These helpers give such a
script the two ends of the loop:

- `read_candidate(prompt, targets)` extracts the selected text regions as a
  `dict[address -> text]` — exactly the `str | dict[str, str]` candidate shape
  `optimize_anything` evolves.
- `apply_candidate(prompt, candidate)` rebuilds a Prompt from an evolved
  candidate while the structural contract holds by construction: template
  mutations preserve the `{{variable}}` set, tool/skill names and schemas
  never change. `apply_candidate(...).to_dict()` is the optimized JSON
  document to persist and load wherever it is needed.

Target selection belongs to the optimizer run, not the prompt document — a
document exposes stable addressable ids but never marks anything as
optimizable.

Addresses:

- ``message:<id>``                 a message template, by message id
- ``tool:<name>``                  a tool description, by tool name
- ``skill:<name>.description``     a skill's when-to-apply prose
- ``skill:<name>.instructions``    a skill's how-to template
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Union

from .prompts import Prompt, PromptError

OptimizerTargetKind = Literal[
    "message_template",
    "tool_description",
    "skill_description",
    "skill_instructions",
]


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

    @classmethod
    def skill_description(cls, skill_name: str) -> "OptimizerTarget":
        return cls(kind="skill_description", id=skill_name)

    @classmethod
    def skill_instructions(cls, skill_name: str) -> "OptimizerTarget":
        return cls(kind="skill_instructions", id=skill_name)

    @property
    def address(self) -> str:
        """The stable string key for this target in a candidate dict."""

        if self.kind == "message_template":
            return f"message:{self.id}"
        if self.kind == "tool_description":
            return f"tool:{self.id}"
        if self.kind == "skill_description":
            return f"skill:{self.id}.description"
        return f"skill:{self.id}.instructions"

    @classmethod
    def from_address(cls, address: str) -> "OptimizerTarget":
        """Parse a ``message:``/``tool:``/``skill:`` address string."""

        prefix, _, rest = address.partition(":")
        if not rest:
            raise PromptError(f"Invalid optimizer target address: '{address}'.")
        if prefix == "message":
            return cls.message_template(rest)
        if prefix == "tool":
            return cls.tool_description(rest)
        if prefix == "skill":
            name, _, part = rest.rpartition(".")
            if part == "description" and name:
                return cls.skill_description(name)
            if part == "instructions" and name:
                return cls.skill_instructions(name)
            raise PromptError(
                f"Invalid skill target address '{address}' (expected "
                "'skill:<name>.description' or 'skill:<name>.instructions')."
            )
        raise PromptError(
            f"Invalid optimizer target address '{address}' (expected a "
            "'message:', 'tool:', or 'skill:' prefix)."
        )


TargetLike = Union[OptimizerTarget, str]


def _coerce_target(target: TargetLike) -> OptimizerTarget:
    if isinstance(target, OptimizerTarget):
        return target
    return OptimizerTarget.from_address(target)


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
    for name in prompt.skills:
        targets.append(OptimizerTarget.skill_description(name))
        targets.append(OptimizerTarget.skill_instructions(name))
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


def read_optimizer_target(prompt: Prompt, target: TargetLike) -> str:
    """Return the current text for a selected optimizer target."""

    target = _coerce_target(target)
    if target.kind == "message_template":
        message = next((item for item in prompt.messages if item.id == target.id), None)
        if message is None:
            raise PromptError(f"No message with id '{target.id}'.")
        if message.template is None:
            raise PromptError(f"Message '{target.id}' has literal content, not a template.")
        return message.template
    if target.kind == "tool_description":
        tool_config = prompt.tools.get(target.id)
        if tool_config is None:
            raise PromptError(f"No tool named '{target.id}'.")
        if tool_config.description is None:
            raise PromptError(f"Tool '{target.id}' has no description.")
        return tool_config.description
    if target.kind in ("skill_description", "skill_instructions"):
        skill = prompt.skills.get(target.id)
        if skill is None:
            raise PromptError(f"No skill named '{target.id}'.")
        if target.kind == "skill_description":
            return skill.description
        return skill.instructions
    raise ValueError(f"Unknown optimizer target kind: {target.kind!r}")


def apply_optimizer_target(
    prompt: Prompt,
    target: TargetLike,
    candidate_text: str,
) -> Prompt:
    """Apply an optimizer candidate to a selected target."""

    target = _coerce_target(target)
    if target.kind == "message_template":
        return prompt.with_template(target.id, candidate_text)
    if target.kind == "tool_description":
        return prompt.with_tool_description(target.id, candidate_text)
    if target.kind == "skill_description":
        return prompt.with_skill_description(target.id, candidate_text)
    if target.kind == "skill_instructions":
        return prompt.with_skill_instructions(target.id, candidate_text)
    raise ValueError(f"Unknown optimizer target kind: {target.kind!r}")


def read_candidate(
    prompt: Prompt,
    targets: Iterable[TargetLike],
) -> dict[str, str]:
    """Extract the selected targets as an optimize_anything seed candidate.

    Returns `{address: current_text}` — pass it as `seed_candidate=`; the
    optimizer proposes evolved dicts with the same keys.
    """

    candidate: dict[str, str] = {}
    for target in targets:
        target = _coerce_target(target)
        if target.address in candidate:
            raise PromptError(f"Duplicate optimizer target: '{target.address}'.")
        candidate[target.address] = read_optimizer_target(prompt, target)
    return candidate


def apply_candidate(
    prompt: Prompt,
    candidate: dict[str, str],
) -> Prompt:
    """Rebuild a Prompt from an evolved `{address: text}` candidate.

    Every mutation goes through the structural contract (variable sets,
    names, and schemas are preserved by construction), so
    `apply_candidate(prompt, best).to_dict()` is a safe optimized document to
    persist and adopt without call-site changes.
    """

    evolved = prompt
    for address, text in candidate.items():
        if not isinstance(text, str):
            raise PromptError(
                f"Candidate value for '{address}' must be a string; got "
                f"{type(text).__name__}."
            )
        evolved = apply_optimizer_target(evolved, address, text)
    return evolved
