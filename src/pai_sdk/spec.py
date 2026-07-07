"""PromptSpec — the typed code socket a prompt document plugs into.

The prompt document (JSON) owns everything model-facing: templates, skill
prose, tool descriptions, model, params. A PromptSpec owns exactly what JSON
cannot carry: Python types (input/output Pydantic models) and executable
behavior (tool handler functions). Together they split cleanly — an external
optimization plane (e.g. Orizu) evolves the document; the app keeps one
long-lived spec and plugs each new document in:

    triage = prompt_spec(
        name="support-triage",
        input=TriageInput,
        output=TriageVerdict,
        tools={"lookup_customer": lookup_customer},
    )

    # Day 0 — author the seed text through the spec, typed end to end:
    seed = triage.document(
        model="anthropic/claude-haiku-4-5",
        system="You triage tickets for {{company}}. Be decisive.",
        user="Ticket: {{ticket}}",
    )
    seed.export("prompts/support-triage.json")     # -> what the optimizer ingests

    # Every day after — plug the optimized JSON back in:
    prompt = triage.load("prompts/support-triage.optimized.json")
    result = await prompt.generate(TriageInput(company="Acme", ticket="..."))
    result.output                                   # TriageVerdict

`bind()`/`load()` enforce the adoption contract at load time: the document's
name must match, required input fields and their types must match (the
document may add extra OPTIONAL fields), and output/tool schemas must be
compatible. Everything an optimizer produces through `apply_candidate`
satisfies this by construction, so a failed bind means a *human* broke the
contract — and the app finds out at deploy time, not mid-conversation.

A spec is optional sugar: every document also runs untyped via plain
`load_prompt`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from pydantic import BaseModel

from .output import _strictify_schema
from .prompts import (
    Prompt,
    PromptError,
    load_prompt,
    load_prompt_url,
)
from .tools import Tool, tool as make_tool

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

_IGNORED_SCHEMA_KEYS = {"title", "description"}
# Keys whose values are field-name maps (children are schemas, keys are data).
_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions"}
# Keys whose values are plain data, never schemas — no filtering inside.
_DATA_KEYS = {"enum", "const", "default", "examples"}


def _comparable_schema(schema: Any, context: str = "schema") -> Any:
    """Schemas compared structurally, ignoring prose annotation keys
    (`title`, `description`) — but only where they ARE annotations: children
    of `properties` are field names (a field may be literally named
    "description"), and enum/const/default values are data. `required` is a
    set, so its order is normalized."""
    if isinstance(schema, list):
        return [
            _comparable_schema(item, "data" if context == "data" else "schema")
            for item in schema
        ]
    if isinstance(schema, dict):
        if context == "map":
            return {
                key: _comparable_schema(value, "schema")
                for key, value in schema.items()
            }
        if context == "data":
            return {
                key: _comparable_schema(value, "data")
                for key, value in schema.items()
            }
        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key in _IGNORED_SCHEMA_KEYS:
                continue
            if key == "required" and isinstance(value, list):
                result[key] = sorted(value)
                continue
            next_context = (
                "map" if key in _MAP_KEYS else "data" if key in _DATA_KEYS else "schema"
            )
            result[key] = _comparable_schema(value, next_context)
        return result
    return schema


def _schemas_compatible(expected: Optional[dict], actual: Optional[dict]) -> bool:
    # dict equality is key-order-insensitive in Python; `required` order is
    # normalized above — key/element order is never semantic (spec/README.md).
    if expected is None or actual is None:
        return expected is None and actual is None
    return _comparable_schema(expected) == _comparable_schema(actual)


@dataclass(frozen=True, eq=False)
class PromptSpec(Generic[InputT, OutputT]):
    """Types + behavior for one named task; documents bind to it.

    Create with :func:`prompt_spec`. The spec never contains prompt text —
    text lives in documents, which the spec authors (``document()``), loads
    (``load()``/``load_url()``), and validates (``bind()``).
    """

    name: str
    input: Optional[type[InputT]] = None
    output: Optional[type[OutputT]] = None
    tools: dict[str, Tool] = field(default_factory=dict)

    # -- authoring: code -> document ---------------------------------------

    def document(self, **config: Any) -> "BoundPrompt[InputT, OutputT]":
        """Author a seed document through the spec.

        Takes the text/config keys of a prompt document (``system``, ``user``,
        ``messages``, ``skills``, ``model``, ``params``, ``toolChoice``,
        ``maxSteps``, ...); the spec contributes ``name``, the input/output
        schemas, and the tool interfaces. Returns a bound prompt —
        ``.export(path)`` writes the JSON an optimizer ingests.
        """
        reserved = {"name", "input", "output", "tools"} & set(config)
        if reserved:
            raise PromptError(
                f"document() derives {', '.join(sorted(reserved))} from the "
                "spec; pass only text/config keys."
            )
        data: dict[str, Any] = {"name": self.name, **config}
        if self.input is not None:
            data["input"] = self.input
        if self.output is not None:
            data["output"] = self.output
        if self.tools:
            data["tools"] = dict(self.tools)
        return BoundPrompt(spec=self, prompt=Prompt.model_validate(data))

    # -- binding: document -> code ------------------------------------------

    def bind(self, prompt: Union[Prompt, dict[str, Any]]) -> "BoundPrompt[InputT, OutputT]":
        """Validate a document against this spec and return a typed prompt."""
        if isinstance(prompt, dict):
            prompt = load_prompt(prompt)
        self._validate(prompt)
        return BoundPrompt(spec=self, prompt=prompt)

    def load(self, source: Union[str, Path, dict[str, Any]]) -> "BoundPrompt[InputT, OutputT]":
        """Load a document from a dict or .json/.yaml file and bind it."""
        return self.bind(load_prompt(source))

    async def load_url(self, url: str, **kwargs: Any) -> "BoundPrompt[InputT, OutputT]":
        """Load a document from a hosted service and bind it."""
        return self.bind(await load_prompt_url(url, **kwargs))

    # -- the contract ---------------------------------------------------------

    def _validate(self, prompt: Prompt) -> None:
        if prompt.name != self.name:
            raise PromptError(
                f"Document is for task '{prompt.name}', not '{self.name}' — "
                "refusing to bind the wrong task's prompt."
            )
        self._validate_input(prompt)
        self._validate_output(prompt)
        self._validate_tools(prompt)

    def _validate_input(self, prompt: Prompt) -> None:
        if self.input is None:
            return
        doc_schema = prompt.input_schema()
        if doc_schema is None:
            raise PromptError(
                f"Spec '{self.name}' declares a typed input; the document has "
                "no input schema."
            )
        spec_schema = self.input.model_json_schema()
        spec_props = spec_schema.get("properties") or {}
        doc_props = doc_schema.get("properties") or {}
        missing = sorted(set(spec_props) - set(doc_props))
        if missing:
            raise PromptError(
                f"Document input schema is missing spec fields: {', '.join(missing)}."
            )
        spec_required = set(spec_schema.get("required") or [])
        doc_required = set(doc_schema.get("required") or [])
        if spec_required != doc_required:
            raise PromptError(
                "Document input required fields "
                f"{sorted(doc_required)} do not match the spec's "
                f"{sorted(spec_required)}. (Documents may add extra OPTIONAL "
                "fields only.)"
            )
        for name in spec_props:
            if not _schemas_compatible(spec_props[name], doc_props[name]):
                raise PromptError(
                    f"Document input field '{name}' has a different type than "
                    "the spec."
                )

    def _validate_output(self, prompt: Prompt) -> None:
        if self.output is None:
            return
        doc_schema = prompt.output.schema_ if prompt.output is not None else None
        spec_schema = _strictify_schema(self.output.model_json_schema())
        if not _schemas_compatible(spec_schema, doc_schema):
            raise PromptError(
                f"Document output schema does not match spec '{self.name}' "
                f"({self.output.__name__})."
            )

    def _validate_tools(self, prompt: Prompt) -> None:
        for name, tool_def in self.tools.items():
            doc_tool = prompt.tools.get(name)
            if doc_tool is None:
                raise PromptError(
                    f"Spec '{self.name}' has a handler for tool '{name}' but "
                    "the document does not declare it."
                )
            if not _schemas_compatible(tool_def.json_schema(), doc_tool.input_schema()):
                raise PromptError(
                    f"Document tool '{name}' input schema does not match the "
                    "spec handler's."
                )


@dataclass(frozen=True, eq=False)
class BoundPrompt(Generic[InputT, OutputT]):
    """A document plugged into its spec: runnable, typed, handler-bound.

    ``generate``/``stream`` accept the spec's input model instance (or a
    plain dict) and return results whose ``output`` parses into the spec's
    output model. Anything not overridden here falls through to the wrapped
    :class:`Prompt` (``content_hash``, ``variables``, ``with_template``-style
    mutation helpers return plain Prompts — re-``bind`` to keep typing).
    """

    spec: PromptSpec[InputT, OutputT]
    prompt: Prompt

    def __getattr__(self, name: str) -> Any:
        # object.__getattribute__ avoids unbounded recursion when copy/pickle
        # probe dunder methods on a not-yet-initialized instance.
        try:
            prompt = object.__getattribute__(self, "prompt")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(prompt, name)

    def _variables(self, input: Union[InputT, dict[str, Any], None]) -> dict[str, Any]:
        if isinstance(input, BaseModel):
            return input.model_dump(mode="json")
        return dict(input or {})

    def _call_extras(self, overrides: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            name: tool_def.execute
            for name, tool_def in self.spec.tools.items()
            if tool_def.execute is not None
        }
        handlers.update(overrides.pop("handlers", None) or {})
        extras: dict[str, Any] = {"handlers": handlers or None, **overrides}
        if self.spec.output is not None and "output" not in overrides:
            from .output import Output

            output_config = self.prompt.output
            extras["output"] = Output.object(
                schema=self.spec.output,
                name=output_config.name if output_config else None,
                description=output_config.description if output_config else None,
            )
        return extras

    async def generate(
        self, input: Union[InputT, dict[str, Any], None] = None, **overrides: Any
    ):
        return await self.prompt.generate(
            self._variables(input), **self._call_extras(overrides)
        )

    async def generate_trace(
        self, input: Union[InputT, dict[str, Any], None] = None, **overrides: Any
    ):
        return await self.prompt.generate_trace(
            self._variables(input), **self._call_extras(overrides)
        )

    def stream(
        self, input: Union[InputT, dict[str, Any], None] = None, **overrides: Any
    ):
        return self.prompt.stream(
            self._variables(input), **self._call_extras(overrides)
        )

    def stream_trace(
        self, input: Union[InputT, dict[str, Any], None] = None, **overrides: Any
    ):
        return self.prompt.stream_trace(
            self._variables(input), **self._call_extras(overrides)
        )

    def render(self, input: Union[InputT, dict[str, Any], None] = None):
        return self.prompt.render(self._variables(input))

    def to_dict(self) -> dict[str, Any]:
        return self.prompt.to_dict()

    def export(self, path: Union[str, Path]) -> Path:
        return self.prompt.export(path)


def prompt_spec(
    *,
    name: str,
    input: Optional[type[InputT]] = None,
    output: Optional[type[OutputT]] = None,
    tools: Optional[dict[str, Union[Tool, Callable[..., Any]]]] = None,
) -> PromptSpec[InputT, OutputT]:
    """Define the typed socket for one named task.

    ``tools`` values are handler functions (wrapped via ``tool(fn)`` — name,
    input/output schemas inferred) or explicit ``Tool`` objects when you want
    a seed description or an explicit schema alongside the handler.
    """
    normalized: dict[str, Tool] = {}
    for tool_name, value in (tools or {}).items():
        normalized[tool_name] = value if isinstance(value, Tool) else make_tool(value)
    return PromptSpec(name=name, input=input, output=output, tools=normalized)
