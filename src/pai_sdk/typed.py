"""Typed messages — templates + variable bindings as first-class message fields.

A TypedSystemMessage/TypedUserMessage/TypedAssistantMessage carries:
- `template`: text with Mustache-style `{{variable}}` placeholders
- `variables`: the bindings used to render it
- `optimize`: whether an optimizer (e.g. GEPA-style reflection) may rewrite
  the template text. Variables are structurally untouchable: placeholders are
  not part of the mutable text, and template mutations that change the
  placeholder set are rejected (see prompts.Prompt.with_template).

`content` renders automatically on construction. Providers only ever read
role/content, so these flow through generate_text unchanged; serialization
(dump_messages) keeps template/variables/optimize alongside the rendered
content, so traces stay structured and re-renderable.

Template syntax is deliberately minimal — plain `{{name}}` only (no format
specs, no attribute/index access) — so the same templates render identically
in a TypeScript implementation. Single braces are ordinary text, which keeps
JSON examples and other brace-heavy prompt text natural to write.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import Field, model_validator

from .errors import AISDKError
from .messages import AssistantModelMessage, SystemModelMessage, UserModelMessage


class TemplateError(AISDKError):
    """Invalid template syntax or missing/invalid variables."""


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TAG_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


def extract_variables(template: str) -> list[str]:
    """Placeholder names in a template, in order of first appearance.

    Only plain `{{name}}` placeholders are allowed; format specs
    (`{{x:>10}}`), conversions (`{{x!r}}`), positional (`{{0}}`, `{{}}`)
    and dotted/indexed access (`{{a.b}}`, `{{a[0]}}`) raise TemplateError —
    keeping templates portable across runtimes.
    """
    names: list[str] = []
    index = 0
    while True:
        open_index = template.find("{{", index)
        if open_index == -1:
            break

        close_index = template.find("}}", open_index + 2)
        if close_index == -1:
            raise TemplateError("Invalid template: unclosed '{{'.")

        raw_name = template[open_index + 2 : close_index]
        name = raw_name.strip()
        if not _NAME_RE.match(name):
            raise TemplateError(
                "Only plain {{name}} placeholders are supported; "
                f"got '{{{{{raw_name}}}}}'."
            )
        if name not in names:
            names.append(name)
        index = close_index + 2
    return names


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a template, requiring every placeholder to be bound."""
    names = extract_variables(template)
    missing = [name for name in names if name not in variables]
    if missing:
        raise TemplateError(
            f"Missing template variables: {', '.join(missing)}."
        )

    def replace(match: re.Match[str]) -> str:
        return str(variables[match.group(1).strip()])

    return _TAG_RE.sub(replace, template)


class _TypedMixin:
    """Shared render-on-construction behavior."""

    @model_validator(mode="after")
    def _render(self):
        if not self.content:
            self.content = render_template(self.template, self.variables)
        return self


class TypedSystemMessage(_TypedMixin, SystemModelMessage):
    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    optimize: bool = False
    id: Optional[str] = None
    content: str = ""


class TypedUserMessage(_TypedMixin, UserModelMessage):
    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    optimize: bool = False
    id: Optional[str] = None
    content: str = ""  # type: ignore[assignment] — text-only in v1


class TypedAssistantMessage(_TypedMixin, AssistantModelMessage):
    """Typed assistant turns (e.g. few-shot demonstrations)."""

    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    optimize: bool = False
    id: Optional[str] = None
    content: str = ""  # type: ignore[assignment]


TYPED_MESSAGE_TYPES = {
    "system": TypedSystemMessage,
    "user": TypedUserMessage,
    "assistant": TypedAssistantMessage,
}
