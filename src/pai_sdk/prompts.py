"""Prompt documents — prompts as data (JSON/YAML, in-repo or hosted).

The prompt document is the portable source of truth. It is a JSON-compatible,
JSON-Schema-validated bundle of everything model-facing: a model reference,
call parameters, structured input/output schemas, message templates with
`{{variable}}` slots, tool interfaces (description + input/output schemas),
and skills. The same document runs identically here and in the TypeScript
sibling (structured-ai-sdk); code-first conveniences (Pydantic models as
schemas, `tool(fn)`) are projections that compile INTO the document, never a
second source of truth.

    prompt = load_prompt("prompts/triage.yaml")
    result = await prompt.generate({"ticket_text": "...", "company_name": "Acme"})

Document format (`specVersion: pai.prompt.v1`; YAML needs the `yaml` extra).
The simple form covers the common case — one system prompt, one user template:

    name: support-triage
    model: anthropic/claude-haiku-4-5       # optional provider/model string
    params:                                 # AI SDK option names, sent verbatim
      maxOutputTokens: 1000
    input:                                  # optional structured input signature
      company_name: string
      ticket_text: string
    output:                                 # optional structured output —
      urgency: [low, medium, high]          # field: type shorthand (enum)
      summary: string                       # string/number/integer/boolean,
      tags: string[]                        # arrays, nested objects
    system: |
      You are a support triage assistant for {{company_name}}. ...
    user: "Ticket: {{ticket_text}}"

`system`/`user` accept a plain template string or
{template, content, id} for control. The general form replaces them with an
explicit `messages:` list (multiple system blocks, few-shot assistant turns,
stable ids for optimizer-selected targets):

    messages:
      - id: instructions
        role: system
        template: |
          You are a support triage assistant for {{company_name}}. ...
      - id: policy
        role: system
        content: "Never reveal internal data."
      - id: ticket
        role: user
        template: "Ticket: {{ticket_text}}"

`input` and `output` are field-type shorthand (above), full JSON Schemas via
`{schema: {...}}`, or — in code — Pydantic model classes, which compile to
plain JSON Schema on serialization.

Skills are named, addressable blocks of model-facing prose: `description`
says when the skill applies, `instructions` (a template) says how. They render
as system messages after the last declared system message:

    skills:
      escalation:
        description: When a ticket mentions legal threats or refunds over $500.
        instructions: |
          Escalate to a human. Summarize the thread for {{company_name}} first.

The optimization contract (for external optimizer runners such as GEPA's
optimize_anything):
- `{{variables}}` are structurally untouchable — they are bindings, not text.
- Optimizer scripts choose which message/tool/skill ids to target; the
  document carries no optimization intent. `with_template()` /
  `with_skill_instructions()` reject any mutation that changes a template's
  placeholder set; `with_tool_description()` / `with_skill_description()`
  rewrite prose while names and schemas stay fixed by construction.
- `content_hash()` identifies a candidate; `to_dict()` persists evolved
  prompts back to JSON/YAML.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import AISDKError, MissingDependencyError
from .generate import generate_text, step_count_is, stream_text
from .tools import Tool, tool as make_tool
from .messages import ModelMessage
from .output import Output, _strictify_schema
from .typed import TYPED_MESSAGE_TYPES, escape_template_literals, extract_variables

JSON_EXTENSIONS = (".json",)
YAML_EXTENSIONS = (".yaml", ".yml")

# The prompt-document spec version. Bumped only when documents or their
# rendering rules change incompatibly; runtimes reject versions they do not
# implement. Mirrors TRACE_SCHEMA_VERSION ("pai.trace.v1").
PROMPT_SPEC_VERSION = "pai.prompt.v1"

# JSON Schema for the prompt-document format itself — point editors at it
# (yaml-language-server: $schema=<path or URL>) to validate/autocomplete
# prompt files. PROMPT_CONFIG_SCHEMA is the parsed dict. structured-ai-sdk
# vendors this file byte-for-byte; see spec/README.md.
PROMPT_CONFIG_SCHEMA_PATH = Path(__file__).parent / "prompt-config.schema.json"
PROMPT_CONFIG_SCHEMA: dict[str, Any] = json.loads(
    PROMPT_CONFIG_SCHEMA_PATH.read_text()
)

_SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Document `params` keys are the AI SDK option vocabulary (camelCase), so a
# TypeScript runtime passes them to generateText/streamText verbatim. This
# table maps them onto pai-sdk's snake_case keyword arguments; unknown keys
# pass through unchanged.
_PARAM_KEY_TO_KWARG = {
    "maxOutputTokens": "max_output_tokens",
    "topP": "top_p",
    "topK": "top_k",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "stopSequences": "stop_sequences",
    "maxRetries": "max_retries",
    "providerOptions": "provider_options",
    "activeTools": "active_tools",
}
_PARAM_SNAKE_HINTS = {snake: camel for camel, snake in _PARAM_KEY_TO_KWARG.items()}


class PromptError(AISDKError):
    """Invalid prompt document or disallowed prompt mutation."""


def _js_float_repr(value: float) -> str:
    """Format a non-integral float exactly like ECMAScript Number::toString.

    Python's repr and JavaScript's String() both emit shortest-round-trip
    digits, but they disagree on when to switch to exponent notation and how
    to format the exponent ("1e-05" vs "1e-7"). Re-format Python's shortest
    digits under the ECMAScript rules so both runtimes emit identical bytes.
    """

    from decimal import Decimal

    sign = "-" if value < 0 else ""
    _s, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    digit_list = list(digit_tuple)
    while len(digit_list) > 1 and digit_list[-1] == 0:
        digit_list.pop()
        exponent += 1
    digits = "".join(map(str, digit_list))
    k = len(digits)
    n = exponent + k  # value == 0.digits * 10**n
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return sign + mantissa + "e" + ("+" if e >= 0 else "-") + str(abs(e))


def _canonical_fragment(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # JavaScript has one number type: integral doubles print as integers
        # (1.0 -> "1", 1e21 -> "1000000000000000000000").
        if value.is_integer():
            return str(int(value))
        return _js_float_repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[0])
        return "{" + ",".join(
            f"{json.dumps(key, ensure_ascii=False)}:{_canonical_fragment(item)}"
            for key, item in items
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_fragment(item) for item in value) + "]"
    raise PromptError(f"Value is not canonical-JSON serializable: {type(value).__name__}.")


def canonical_prompt_json(config: dict[str, Any]) -> str:
    """The canonical JSON serialization used for `content_hash()`.

    Sorted keys (by code point), compact separators, raw (non-ASCII-escaped)
    unicode, and ECMAScript-compatible number formatting — specified so a
    TypeScript implementation produces byte-identical output for every
    document (see spec/README.md).
    """

    return _canonical_fragment(config)


class PromptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    template: Optional[str] = None
    content: Optional[str] = None  # literal text — no interpolation
    id: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_body(self) -> "PromptMessage":
        if (self.template is None) == (self.content is None):
            raise ValueError("A prompt message needs exactly one of template/content.")
        if self.template is not None:
            extract_variables(self.template)  # validate syntax eagerly
        return self

    @property
    def text(self) -> str:
        return self.template if self.template is not None else self.content  # type: ignore[return-value]

    @property
    def variables(self) -> list[str]:
        return extract_variables(self.template) if self.template is not None else []


class PromptSkill(BaseModel):
    """A named, addressable block of model-facing prose.

    `description` is when-to-apply prose; `instructions` is a how-to template
    (with `{{variable}}` slots that join the prompt's input contract). Both
    are optimizer-targetable text; the skill NAME is the contract. Skills
    render as system messages with id `skill:<name>` after the last declared
    system message.
    """

    model_config = ConfigDict(extra="forbid")

    description: str
    instructions: str

    @model_validator(mode="after")
    def _validate_template(self) -> "PromptSkill":
        extract_variables(self.instructions)  # validate syntax eagerly
        return self

    @property
    def variables(self) -> list[str]:
        return extract_variables(self.instructions)


_SHORTHAND_TYPES = {
    "string": {"type": "string"},
    "number": {"type": "number"},
    "integer": {"type": "integer"},
    "int": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "bool": {"type": "boolean"},
}


def _compile_schema_shorthand(fields: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Compile field-type shorthand into a strict JSON Schema object.

    Field values: "string" | "number" | "integer" | "boolean" (or None for
    string), "<type>[]" for arrays, a list of literals for an enum, or a
    nested mapping for a nested object. All fields are required.
    """

    def field_schema(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "string"}
        if isinstance(value, str):
            if value.endswith("[]"):
                return {"type": "array", "items": field_schema(value[:-2])}
            if value in _SHORTHAND_TYPES:
                return dict(_SHORTHAND_TYPES[value])
            raise PromptError(
                f"Unknown {label} field type '{value}' (expected one of "
                f"{', '.join(_SHORTHAND_TYPES)}, '<type>[]', a list of enum "
                "values, or a nested mapping)."
            )
        if isinstance(value, list):
            return {"enum": value}
        if isinstance(value, dict):
            return _compile_schema_shorthand(value, label=label)
        raise PromptError(f"Unknown {label} field type: {value!r}")

    return {
        "type": "object",
        "properties": {name: field_schema(value) for name, value in fields.items()},
        "required": list(fields.keys()),
        "additionalProperties": False,
    }


def compile_schema_shorthand(fields: dict[str, Any]) -> dict[str, Any]:
    """Compile field-type shorthand into a strict JSON Schema object."""

    return _compile_schema_shorthand(fields, label="schema")


def compile_input_shorthand(fields: dict[str, Any]) -> dict[str, Any]:
    """Compile prompt input shorthand into a strict JSON Schema object."""

    return _compile_schema_shorthand(fields, label="input")


def compile_output_shorthand(fields: dict[str, Any]) -> dict[str, Any]:
    """Compile prompt output shorthand into a strict JSON Schema object."""

    return _compile_schema_shorthand(fields, label="output")


def _is_model_class(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_: dict[str, Any] = Field(alias="schema")
    name: Optional[str] = None
    description: Optional[str] = None
    # The Pydantic class this schema was compiled from, when constructed in
    # code. Never serialized — the document carries only plain JSON Schema.
    source_model: Optional[Any] = Field(default=None, exclude=True, repr=False)

    @field_validator("source_model")
    @classmethod
    def _source_model_is_code_only(cls, value: Any) -> Any:
        # A loaded document must not be able to smuggle in a second schema
        # that to_dict()/content_hash() would never reveal.
        if value is not None and not _is_model_class(value):
            raise ValueError(
                "source_model is code-only (a Pydantic model class); "
                "documents must not set it."
            )
        return value


class PromptOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_: dict[str, Any] = Field(alias="schema")
    name: Optional[str] = None
    description: Optional[str] = None
    # As on PromptInput; when present, result.output parses into this class.
    source_model: Optional[Any] = Field(default=None, exclude=True, repr=False)

    @field_validator("source_model")
    @classmethod
    def _source_model_is_code_only(cls, value: Any) -> Any:
        if value is not None and not _is_model_class(value):
            raise ValueError(
                "source_model is code-only (a Pydantic model class); "
                "documents must not set it."
            )
        return value


class PromptTool(BaseModel):
    """A tool *interface* declared in a prompt document.

    The document carries what is serializable — description and input/output
    schemas; behavior binds at call time via `prompt.generate(...,
    handlers={name: fn})`. Declared tools without a handler are client-side:
    calls come back on result.tool_calls. The name (the dict key) and the
    schemas are the contract; optimizer scripts may target DESCRIPTION text by
    tool name (descriptions are instructions — when-to-call errors are
    description failures).
    """

    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = None
    # Field-type shorthand (same grammar as output:), {"schema": {...}}, or —
    # in code — a Pydantic model class (compiled to JSON Schema on load).
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None
    # The execute function carried over when a runtime Tool (e.g. tool(fn))
    # is placed in a Prompt. Never serialized — behavior is code, the
    # document carries only the interface. Call-time handlers= win.
    bound_execute: Optional[Any] = Field(default=None, exclude=True, repr=False)

    @field_validator("bound_execute")
    @classmethod
    def _bound_execute_is_code_only(cls, value: Any) -> Any:
        # A loaded document must not be able to flip a client-side tool into
        # a locally "executed" one.
        if value is not None and not callable(value):
            raise ValueError(
                "bound_execute is code-only (a callable); documents must not set it."
            )
        return value

    @field_validator("input", "output", mode="before")
    @classmethod
    def _coerce_model_schema(cls, value: Any) -> Any:
        if _is_model_class(value):
            return {"schema": value.model_json_schema()}
        if isinstance(value, dict) and _is_model_class(value.get("schema")):
            return {**value, "schema": value["schema"].model_json_schema()}
        return value

    @model_validator(mode="after")
    def _validate_schemas(self) -> "PromptTool":
        self.input_schema()  # compile eagerly so config errors surface at load
        self.output_schema()
        return self

    def input_schema(self) -> Optional[dict[str, Any]]:
        return _tool_schema(self.input, label="input")

    def output_schema(self) -> Optional[dict[str, Any]]:
        """The declared result schema — interface documentation and typing
        data for consumers; not enforced against handler return values."""
        return _tool_schema(self.output, label="tool output")


def _tool_schema(
    config: Optional[dict[str, Any]], *, label: str
) -> Optional[dict[str, Any]]:
    if config is None:
        return None
    if "schema" in config:
        schema = config["schema"]
        if not isinstance(schema, dict):
            raise PromptError(
                f"Tool {label} schema must be an object; got "
                f"{type(schema).__name__}. (A shorthand field named 'schema' "
                "needs the full JSON Schema form.)"
            )
        return schema
    return _compile_schema_shorthand(config, label=label)


ToolChoiceConfig = Union[Literal["auto", "none", "required"], dict[str, Any]]


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    spec_version: Literal["pai.prompt.v1"] = Field(
        default=PROMPT_SPEC_VERSION, alias="specVersion"
    )
    name: str
    version: Optional[Union[int, str]] = None
    description: Optional[str] = None
    model: Optional[str] = None  # "provider/model-id" string
    params: dict[str, Any] = Field(default_factory=dict)
    input: Optional[PromptInput] = None
    output: Optional[PromptOutput] = None
    tools: dict[str, PromptTool] = Field(default_factory=dict)
    tool_choice: Optional[ToolChoiceConfig] = Field(default=None, alias="toolChoice")
    max_steps: Optional[int] = Field(default=None, ge=1, alias="maxSteps")
    messages: list[PromptMessage] = Field(default_factory=list)
    skills: dict[str, PromptSkill] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_simple_form(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # Runtime Tool values (e.g. tool(fn)) compile into tool interfaces;
        # their execute functions bind as default handlers.
        tools_config = data.get("tools")
        if isinstance(tools_config, dict) and any(
            isinstance(value, Tool) for value in tools_config.values()
        ):
            converted: dict[str, Any] = {}
            for name, value in tools_config.items():
                if not isinstance(value, Tool):
                    converted[name] = value
                    continue
                entry: dict[str, Any] = {"input": {"schema": value.json_schema()}}
                if value.description is not None:
                    entry["description"] = value.description
                output_schema = value.output_json_schema()
                if output_schema is not None:
                    entry["output"] = {"schema": output_schema}
                if value.strict is not None:
                    entry["strict"] = value.strict
                if value.execute is not None:
                    entry["bound_execute"] = value.execute
                converted[name] = entry
            data["tools"] = converted

        # input:/output: Pydantic model class or field-type shorthand ->
        # full JSON Schema (keeping the source class for typed parsing).
        input_config = data.get("input")
        if _is_model_class(input_config):
            data["input"] = {
                "schema": input_config.model_json_schema(),
                "source_model": input_config,
            }
        elif isinstance(input_config, dict):
            schema_value = input_config.get("schema")
            if _is_model_class(schema_value):
                data["input"] = {
                    **input_config,
                    "schema": schema_value.model_json_schema(),
                    "source_model": schema_value,
                }
            elif "schema" not in input_config:
                data["input"] = {"schema": compile_input_shorthand(input_config)}

        output = data.get("output")
        if _is_model_class(output):
            data["output"] = {
                "schema": _strictify_schema(output.model_json_schema()),
                "source_model": output,
            }
        elif isinstance(output, dict):
            schema_value = output.get("schema")
            if _is_model_class(schema_value):
                data["output"] = {
                    **output,
                    "schema": _strictify_schema(schema_value.model_json_schema()),
                    "source_model": schema_value,
                }
            elif "schema" not in output:
                data["output"] = {"schema": compile_output_shorthand(output)}

        # top-level system:/user: -> messages list
        system = data.pop("system", None)
        user = data.pop("user", None)
        if system is not None or user is not None:
            if data.get("messages"):
                raise ValueError(
                    "Use either top-level system/user or messages:, not both."
                )
            messages: list[dict[str, Any]] = []
            if system is not None:
                entry = system if isinstance(system, dict) else {"template": system}
                messages.append(
                    {
                        "role": "system",
                        "id": "system",
                        **entry,
                    }
                )
            if user is not None:
                entry = user if isinstance(user, dict) else {"template": user}
                messages.append({"role": "user", "id": "user", **entry})
            data["messages"] = messages
        return data

    @model_validator(mode="after")
    def _validate_document(self) -> "Prompt":
        if not self.messages:
            raise ValueError(
                "A prompt needs messages — top-level system:/user: or a messages: list."
            )
        for name in self.skills:
            if not _SKILL_NAME_PATTERN.fullmatch(name):
                raise ValueError(
                    f"Invalid skill name '{name}' (letters, digits, '-', '_' only)."
                )
        for key in self.params:
            if key in _PARAM_SNAKE_HINTS:
                raise ValueError(
                    f"Unknown params key '{key}' — did you mean "
                    f"'{_PARAM_SNAKE_HINTS[key]}'? Document params use AI SDK "
                    "option names."
                )
        if isinstance(self.tool_choice, dict) and "tool_name" in self.tool_choice:
            raise ValueError(
                "tool choice uses {'type': 'tool', 'toolName': ...} — "
                "'tool_name' is not a document key."
            )
        ids = [m.id for m in self._effective_messages() if m.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Prompt message ids must be unique (skills reserve 'skill:<name>')."
            )
        self._validate_input_schema()
        return self

    def _validate_input_schema(self) -> None:
        if self.input is None:
            return
        schema = self.input.schema_
        if schema.get("type") not in (None, "object"):
            raise ValueError("Prompt input schema must be an object schema.")
        properties = schema.get("properties")
        if properties is None:
            return
        if not isinstance(properties, dict):
            raise ValueError("Prompt input schema properties must be an object.")
        missing = [name for name in self.variables if name not in properties]
        if missing:
            raise ValueError(
                "Prompt input schema must declare template variables: "
                f"{', '.join(missing)}."
            )

    # -- introspection -------------------------------------------------------

    def _skill_message(self, name: str, skill: PromptSkill) -> PromptMessage:
        # Composition is part of the spec (spec/README.md): the description is
        # literal prose (escaped), the instructions keep their placeholders.
        template = (
            f"Skill: {name}\n"
            f"{escape_template_literals(skill.description)}\n\n"
            f"{skill.instructions}"
        )
        return PromptMessage(role="system", template=template, id=f"skill:{name}")

    def _effective_messages(self) -> list[PromptMessage]:
        """Declared messages with skills rendered in as system messages.

        Skills follow the last declared system message (or lead when there is
        none), in code-point-sorted name order — object key order is never
        semantic in a document, and the canonical hash sorts keys, so
        rendering must not depend on declaration order either. This is the
        sequence render() produces.
        """
        if not self.skills:
            return list(self.messages)
        skill_messages = [
            self._skill_message(name, self.skills[name])
            for name in sorted(self.skills)
        ]
        last_system = next(
            (
                index
                for index in range(len(self.messages) - 1, -1, -1)
                if self.messages[index].role == "system"
            ),
            None,
        )
        if last_system is None:
            return [*skill_messages, *self.messages]
        return [
            *self.messages[: last_system + 1],
            *skill_messages,
            *self.messages[last_system + 1 :],
        ]

    @property
    def variables(self) -> list[str]:
        """All template variables across messages and skills, in render order."""
        names: list[str] = []
        for message in self._effective_messages():
            for name in message.variables:
                if name not in names:
                    names.append(name)
        return names

    def content_hash(self) -> str:
        """Stable candidate identity: sha256 of the canonical document JSON,
        truncated to 16 hex chars. The algorithm is spec'd (spec/README.md) so
        Python and TypeScript agree on every hash."""
        canonical = canonical_prompt_json(self.to_dict())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude_none=True)
        # Empty containers mean "not declared" — omit them so equivalent
        # documents serialize (and hash) identically across runtimes.
        for key in ("params", "tools", "skills"):
            if not data.get(key):
                data.pop(key, None)
        return data

    def input_schema(self) -> Optional[dict[str, Any]]:
        """Return the declared structured input schema, if any."""

        return self.input.schema_ if self.input is not None else None

    def validate_inputs(self, variables: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Lightweight top-level validation for prompt input variables.

        This intentionally does not implement full JSON Schema validation.
        It enforces the parts pai-sdk relies on for stable call signatures:
        required top-level fields and `additionalProperties: false`.
        """

        data = dict(variables or {})
        if self.input is None:
            return data
        schema = self.input.schema_
        properties = schema.get("properties")
        property_names = set(properties) if isinstance(properties, dict) else None
        required = schema.get("required") or []
        missing_required = [name for name in required if name not in data]
        if missing_required:
            raise PromptError(
                f"Prompt '{self.name}' is missing required input fields: "
                f"{', '.join(missing_required)}."
            )
        if schema.get("additionalProperties") is False and property_names is not None:
            extra = sorted(set(data) - property_names)
            if extra:
                raise PromptError(
                    f"Prompt '{self.name}' received unknown input fields: "
                    f"{', '.join(extra)}."
                )
        return data

    # -- the optimization contract -------------------------------------------

    def with_template(self, message_id: str, new_template: str) -> "Prompt":
        """A new Prompt with one message's template rewritten.

        Enforces the structural optimization contract: the message must exist
        and the new template must bind exactly the same variable set
        (placeholders are data plumbing — not the optimizer's to add/remove).
        """
        index = next(
            (i for i, m in enumerate(self.messages) if m.id == message_id), None
        )
        if index is None:
            raise PromptError(f"No message with id '{message_id}'.")
        message = self.messages[index]
        if message.template is None:
            raise PromptError(f"Message '{message_id}' has literal content, not a template.")
        old_vars = set(extract_variables(message.template))
        new_vars = set(extract_variables(new_template))
        if old_vars != new_vars:
            raise PromptError(
                f"Template mutation for '{message_id}' must preserve the "
                f"variable set {sorted(old_vars)}; got {sorted(new_vars)}."
            )
        updated = message.model_copy(update={"template": new_template})
        messages = [*self.messages[:index], updated, *self.messages[index + 1 :]]
        return self.model_copy(update={"messages": messages})

    def with_tool_description(self, tool_name: str, new_description: str) -> "Prompt":
        """A new Prompt with one tool's DESCRIPTION rewritten.

        Same contract shape as with_template: the tool must exist. The tool's
        name and input/output schemas are the contract and cannot be changed
        through this method by construction.
        """
        tool_config = self.tools.get(tool_name)
        if tool_config is None:
            raise PromptError(f"No tool named '{tool_name}'.")
        tools = {
            **self.tools,
            tool_name: tool_config.model_copy(update={"description": new_description}),
        }
        return self.model_copy(update={"tools": tools})

    def with_skill_description(self, skill_name: str, new_description: str) -> "Prompt":
        """A new Prompt with one skill's when-to-apply DESCRIPTION rewritten.

        Descriptions are literal prose (no placeholders to preserve); the
        skill name and its instructions stay untouched by construction.
        """
        skill = self.skills.get(skill_name)
        if skill is None:
            raise PromptError(f"No skill named '{skill_name}'.")
        skills = {
            **self.skills,
            skill_name: skill.model_copy(update={"description": new_description}),
        }
        return self.model_copy(update={"skills": skills})

    def with_skill_instructions(self, skill_name: str, new_instructions: str) -> "Prompt":
        """A new Prompt with one skill's INSTRUCTIONS template rewritten.

        Same variable-set contract as with_template.
        """
        skill = self.skills.get(skill_name)
        if skill is None:
            raise PromptError(f"No skill named '{skill_name}'.")
        old_vars = set(extract_variables(skill.instructions))
        new_vars = set(extract_variables(new_instructions))
        if old_vars != new_vars:
            raise PromptError(
                f"Instructions mutation for skill '{skill_name}' must preserve "
                f"the variable set {sorted(old_vars)}; got {sorted(new_vars)}."
            )
        skills = {
            **self.skills,
            skill_name: skill.model_copy(update={"instructions": new_instructions}),
        }
        return self.model_copy(update={"skills": skills})

    # -- rendering & execution ------------------------------------------------

    def render(self, variables: Optional[dict[str, Any]] = None) -> list[ModelMessage]:
        """Render into typed messages (template/variables preserved on each
        message for structured traces). Skills render as system messages with
        id `skill:<name>`. Missing variables raise; extras are ignored."""
        variables = variables or {}
        missing = [n for n in self.variables if n not in variables]
        if missing:
            raise PromptError(
                f"Prompt '{self.name}' is missing variables: {', '.join(missing)}."
            )
        variables = self.validate_inputs(variables)
        return [
            self._render_prompt_message(message, variables)
            for message in self._effective_messages()
        ]

    def render_message(
        self, message_id: str, variables: Optional[dict[str, Any]] = None
    ) -> ModelMessage:
        """Render ONE message (or skill, via `skill:<name>`) from the document.

        For appending typed, trace-preserving turns to an ongoing conversation
        without re-rendering the whole prompt. Only the message's own
        variables are required.
        """
        message = next(
            (m for m in self._effective_messages() if m.id == message_id), None
        )
        if message is None:
            raise PromptError(f"No message with id '{message_id}'.")
        variables = variables or {}
        missing = [n for n in message.variables if n not in variables]
        if missing:
            raise PromptError(
                f"Message '{message_id}' is missing variables: {', '.join(missing)}."
            )
        return self._render_prompt_message(message, variables)

    def _render_prompt_message(
        self, message: PromptMessage, variables: dict[str, Any]
    ) -> ModelMessage:
        typed_cls = TYPED_MESSAGE_TYPES[message.role]
        if message.template is not None:
            bound = {n: variables[n] for n in message.variables}
            return typed_cls(
                template=message.template,
                variables=bound,
                id=message.id,
            )
        return typed_cls(
            template=escape_template_literals(message.content),
            variables={},
            id=message.id,
            content=message.content,
        )

    def _call_kwargs(
        self,
        variables: Optional[dict[str, Any]],
        model: Any,
        handlers: Optional[dict[str, Any]],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_model = model if model is not None else self.model
        if resolved_model is None:
            raise PromptError(
                f"Prompt '{self.name}' has no model — set `model:` in the "
                "config or pass model= at call time."
            )
        kwargs: dict[str, Any] = {
            **{
                _PARAM_KEY_TO_KWARG.get(key, key): value
                for key, value in self.params.items()
            },
            **overrides,
        }
        kwargs["model"] = resolved_model
        kwargs["messages"] = self.render(variables)
        if self.output is not None and "output" not in kwargs:
            kwargs["output"] = Output.object(
                # Prefer the source Pydantic class (typed result.output);
                # documents loaded from data fall back to the JSON Schema.
                schema=self.output.source_model or self.output.schema_,
                name=self.output.name,
                description=self.output.description,
            )

        bound = {
            name: t.bound_execute
            for name, t in self.tools.items()
            if t.bound_execute is not None
        }
        handlers = {**bound, **(handlers or {})}
        unknown = sorted(set(handlers) - set(self.tools))
        if unknown:
            raise PromptError(
                f"Handlers for undeclared tools: {', '.join(unknown)}. "
                f"Declared tools: {', '.join(sorted(self.tools)) or '(none)'}."
            )
        if self.tools:
            # Declared tools without a handler are client-side: calls come
            # back on result.tool_calls. An explicit tools= override wins.
            kwargs.setdefault(
                "tools",
                {
                    name: make_tool(
                        description=t.description,
                        input_schema=t.input_schema(),
                        output_schema=t.output_schema(),
                        execute=handlers.get(name),
                        strict=t.strict,
                    )
                    for name, t in self.tools.items()
                },
            )
            if self.tool_choice is not None:
                choice = self.tool_choice
                if isinstance(choice, dict):
                    # document shape {"type": "tool", "toolName": ...} ->
                    # pai-sdk's internal {"type": "tool", "tool_name": ...}
                    choice = {"type": "tool", "tool_name": choice["toolName"]}
                kwargs.setdefault("tool_choice", choice)
        if self.max_steps is not None:
            kwargs.setdefault("stop_when", step_count_is(self.max_steps))
        return kwargs

    async def generate(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        handlers: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ):
        """Render and run generate_text with this prompt's config.
        `handlers` binds execute functions to declared tool names;
        `overrides` are generate_text kwargs and win over `params`."""
        return await generate_text(**self._call_kwargs(variables, model, handlers, overrides))

    async def generate_trace(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        handlers: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ):
        """Render and run this prompt, returning a result with `.trace`.

        The trace span joins the structured `variables`, parsed outputs, usage,
        metadata, and full replayable `ModelMessage[]` history for the call.
        """
        from .trace import generate_trace

        return await generate_trace(
            self,
            variables,
            model=model,
            handlers=handlers,
            **overrides,
        )

    def stream(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        handlers: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ):
        """Render and run stream_text with this prompt's config."""
        return stream_text(**self._call_kwargs(variables, model, handlers, overrides))

    def stream_trace(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        handlers: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ):
        """Render and stream this prompt, returning a stream result with `.trace`.

        The trace is awaitable after the stream has completed:
        `trace = await result.trace`.
        """
        from .trace import stream_trace

        return stream_trace(
            self,
            variables,
            model=model,
            handlers=handlers,
            **overrides,
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _parse_text(text: str, *, suffix: str) -> dict[str, Any]:
    if suffix in YAML_EXTENSIONS:
        try:
            import yaml
        except ImportError as exc:
            raise MissingDependencyError("pyyaml", "yaml") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_prompt(source: Union[str, Path, dict[str, Any]]) -> Prompt:
    """Load a Prompt from a dict or a local .json/.yaml/.yml file."""
    if isinstance(source, dict):
        return Prompt.model_validate(source)
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix not in JSON_EXTENSIONS + YAML_EXTENSIONS:
        raise PromptError(
            f"Unsupported prompt file extension '{suffix}' "
            "(expected .json, .yaml, or .yml)."
        )
    return Prompt.model_validate(_parse_text(path.read_text(), suffix=suffix))


async def load_prompt_url(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    format: Optional[Literal["json", "yaml"]] = None,
    timeout: float = 30.0,
) -> Prompt:
    """Load a Prompt from a hosted service. Format inferred from the URL
    path/content-type (default JSON); force with format=."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    if format is None:
        content_type = response.headers.get("content-type", "")
        is_yaml = url.endswith(YAML_EXTENSIONS) or "yaml" in content_type
        format = "yaml" if is_yaml else "json"
    suffix = ".yaml" if format == "yaml" else ".json"
    return Prompt.model_validate(_parse_text(response.text, suffix=suffix))
