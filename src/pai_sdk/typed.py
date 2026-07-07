"""Typed messages — templates + variable bindings as first-class message fields.

A TypedSystemMessage/TypedUserMessage/TypedAssistantMessage carries:
- `template`: text with Mustache-style `{{variable}}` placeholders
- `variables`: the bindings used to render it
- `id`: the stable prompt-config message id that produced it

Optimization intent is not message metadata: optimizer runs choose target ids
at run time. Variables are structurally untouchable: placeholders are not part
of the mutable text, and template mutations that change the placeholder set
are rejected (see prompts.Prompt.with_template).

`content` renders automatically on construction. Providers only ever read
role/content, so these flow through generate_text unchanged; serialization
(dump_messages) keeps template/variables/id alongside the rendered content,
so traces stay structured and re-renderable.

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
    escaped: list[str] = []
    index = 0
    while True:
        open_index = text.find("{{", index)
        if open_index == -1:
            escaped.append(text[index:])
            break

        backslash_count = _count_preceding_backslashes(text, open_index)
        backslashes_start = open_index - backslash_count
        escaped.append(text[index:backslashes_start])
        escaped.append("\\" * (backslash_count * 2 + 1))
        escaped.append("{{")
        index = open_index + 2
    return "".join(escaped)


def _count_preceding_backslashes(template: str, open_index: int) -> int:
    backslash_count = 0
    index = open_index - 1
    while index >= 0 and template[index] == "\\":
        backslash_count += 1
        index -= 1
    return backslash_count


def _unescape_template_literals(text: str) -> str:
    rendered: list[str] = []
    index = 0
    while True:
        open_index = text.find("{{", index)
        if open_index == -1:
            rendered.append(text[index:])
            break

        backslash_count = _count_preceding_backslashes(text, open_index)
        if backslash_count % 2 == 1:
            backslashes_start = open_index - backslash_count
            rendered.append(text[index:backslashes_start])
            rendered.append("\\" * (backslash_count // 2))
            rendered.append("{{")
            index = open_index + 2
        else:
            rendered.append(text[index : open_index + 2])
            index = open_index + 2
    return "".join(rendered)


def _is_escaped_tag_open(template: str, open_index: int) -> bool:
    return _count_preceding_backslashes(template, open_index) % 2 == 1


def _unescape_literal_before_tag(text: str) -> str:
    backslash_count = 0
    index = len(text) - 1
    while index >= 0 and text[index] == "\\":
        backslash_count += 1
        index -= 1
    if backslash_count == 0:
        return _unescape_template_literals(text)

    prefix = text[: len(text) - backslash_count]
    return _unescape_template_literals(prefix) + ("\\" * (backslash_count // 2))


def _iter_tags(template: str):
    index = 0
    while True:
        open_index = template.find("{{", index)
        if open_index == -1:
            break
        if _is_escaped_tag_open(template, open_index):
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
        rendered.append(_unescape_literal_before_tag(template[last_index:start]))
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
    id: Optional[str] = None
    content: str = ""


class TypedUserMessage(_TypedMixin, UserModelMessage):
    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None
    content: str = ""  # type: ignore[assignment] — text-only in v1


class TypedAssistantMessage(_TypedMixin, AssistantModelMessage):
    """Typed assistant turns (e.g. few-shot demonstrations)."""

    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None
    content: str = ""  # type: ignore[assignment]


TYPED_MESSAGE_TYPES = {
    "system": TypedSystemMessage,
    "user": TypedUserMessage,
    "assistant": TypedAssistantMessage,
}
