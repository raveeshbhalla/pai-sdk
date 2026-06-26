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

from typing import Any, Optional

from pydantic import Field, model_validator

from .errors import AISDKError
from .messages import AssistantModelMessage, SystemModelMessage, UserModelMessage


class TemplateError(AISDKError):
    """Invalid template syntax or missing/invalid variables."""


def escape_template_literals(text: str) -> str:
    """Escape literal Mustache opens so render_template returns the same text."""
    return text.replace("{{", r"\{{")


def _unescape_template_literals(text: str) -> str:
    return text.replace(r"\{{", "{{")


def _iter_tags(template: str):
    index = 0
    while True:
        open_index = template.find("{{", index)
        if open_index == -1:
            break
        if open_index > 0 and template[open_index - 1] == "\\":
            index = open_index + 2
            continue

        close_index = template.find("}}", open_index + 2)
        if close_index == -1:
            raise TemplateError("Invalid template: unclosed '{{'.")

        yield open_index, close_index + 2, template[open_index + 2 : close_index]
        index = close_index + 2


def extract_variables(template: str) -> list[str]:
    """Placeholder names in a template, in order of first appearance.

    Only plain `{{name}}` placeholders are allowed; format specs
    (`{{x:>10}}`), conversions (`{{x!r}}`), positional (`{{0}}`, `{{}}`)
    and dotted/indexed access (`{{a.b}}`, `{{a[0]}}`) raise TemplateError —
    keeping templates portable across runtimes.
    """
    names: list[str] = []
    for _start, _end, raw_name in _iter_tags(template):
        name = raw_name.strip()
        if not name.isidentifier():
            raise TemplateError(
                "Only plain {{name}} placeholders are supported; "
                f"got '{{{{{raw_name}}}}}'."
            )
        if name not in names:
            names.append(name)
    return names


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a template, requiring every placeholder to be bound."""
    names = extract_variables(template)
    missing = [name for name in names if name not in variables]
    if missing:
        raise TemplateError(
            f"Missing template variables: {', '.join(missing)}."
        )

    rendered: list[str] = []
    last_index = 0
    for start, end, raw_name in _iter_tags(template):
        rendered.append(_unescape_template_literals(template[last_index:start]))
        rendered.append(str(variables[raw_name.strip()]))
        last_index = end
    rendered.append(_unescape_template_literals(template[last_index:]))
    return "".join(rendered)


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
