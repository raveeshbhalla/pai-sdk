"""Prompt configs — prompts as data (JSON/YAML, in-repo or hosted).

A Prompt bundles a model reference, call parameters, an optional structured
output schema, and a list of message templates with `{variable}` slots. Load
it from a dict, a JSON/YAML file in the codebase, or a URL (a hosted prompt
service), then render/execute it:

    prompt = load_prompt("prompts/triage.yaml")
    result = await prompt.generate({"ticket_text": "...", "company_name": "Acme"})

Config schema (JSON-compatible; YAML needs the `yaml` extra):

    name: support-triage
    version: 3                              # optional
    model: anthropic/claude-haiku-4-5       # optional provider/model string
    params:                                 # optional generate_text kwargs
      temperature: 0.2
      max_output_tokens: 1000
    output:                                 # optional structured output
      schema: { type: object, properties: {...}, ... }   # JSON Schema
      name: triage
    messages:
      - id: instructions
        role: system
        optimize: true                      # reflection MAY rewrite this text
        template: |
          You are a support triage assistant for {company_name}. ...
      - id: ticket
        role: user
        template: "Ticket: {ticket_text}"

The optimization contract (for GEPA-style optimizers):
- `{variables}` are structurally untouchable — they are bindings, not text.
- Only messages with `optimize: true` may be rewritten, via `with_template()`,
  which also rejects any mutation that changes the template's placeholder set.
- `content_hash()` identifies a candidate; `to_dict()` persists evolved
  prompts back to JSON/YAML.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import AISDKError, MissingDependencyError
from .generate import generate_text, stream_text
from .messages import ModelMessage
from .output import Output
from .typed import TYPED_MESSAGE_TYPES, extract_variables

JSON_EXTENSIONS = (".json",)
YAML_EXTENSIONS = (".yaml", ".yml")


class PromptError(AISDKError):
    """Invalid prompt config or disallowed prompt mutation."""


class PromptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    template: Optional[str] = None
    content: Optional[str] = None  # literal text — no interpolation
    optimize: bool = False
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


class PromptOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: dict[str, Any] = Field(alias="schema")
    name: Optional[str] = None
    description: Optional[str] = None


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: Optional[Union[int, str]] = None
    description: Optional[str] = None
    model: Optional[str] = None  # "provider/model-id" string
    params: dict[str, Any] = Field(default_factory=dict)
    output: Optional[PromptOutput] = None
    messages: list[PromptMessage]

    @model_validator(mode="after")
    def _unique_ids(self) -> "Prompt":
        ids = [m.id for m in self.messages if m.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Prompt message ids must be unique.")
        return self

    # -- introspection -------------------------------------------------------

    @property
    def variables(self) -> list[str]:
        """All template variables across messages, in order of appearance."""
        names: list[str] = []
        for message in self.messages:
            for name in message.variables:
                if name not in names:
                    names.append(name)
        return names

    def optimizable_messages(self) -> list[PromptMessage]:
        """The messages a reflective optimizer may rewrite."""
        return [m for m in self.messages if m.optimize]

    def content_hash(self) -> str:
        """Stable identity for this prompt candidate (config-content hash)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    # -- the optimization contract -------------------------------------------

    def with_template(self, message_id: str, new_template: str) -> "Prompt":
        """A new Prompt with one message's template rewritten.

        Enforces the optimization contract: the message must exist, must be
        `optimize: true`, and the new template must bind exactly the same
        variable set (placeholders are data plumbing — not the optimizer's to
        add or remove).
        """
        index = next(
            (i for i, m in enumerate(self.messages) if m.id == message_id), None
        )
        if index is None:
            raise PromptError(f"No message with id '{message_id}'.")
        message = self.messages[index]
        if not message.optimize:
            raise PromptError(
                f"Message '{message_id}' is not marked optimize: true — "
                "it must not be rewritten."
            )
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

    # -- rendering & execution ------------------------------------------------

    def render(self, variables: Optional[dict[str, Any]] = None) -> list[ModelMessage]:
        """Render into typed messages (template/variables preserved on each
        message for structured traces). Missing variables raise; extras are
        ignored."""
        variables = variables or {}
        missing = [n for n in self.variables if n not in variables]
        if missing:
            raise PromptError(
                f"Prompt '{self.name}' is missing variables: {', '.join(missing)}."
            )
        rendered: list[ModelMessage] = []
        for message in self.messages:
            typed_cls = TYPED_MESSAGE_TYPES[message.role]
            if message.template is not None:
                bound = {n: variables[n] for n in message.variables}
                rendered.append(
                    typed_cls(
                        template=message.template,
                        variables=bound,
                        optimize=message.optimize,
                        id=message.id,
                    )
                )
            else:
                rendered.append(
                    typed_cls(
                        template=message.content.replace("{", "{{").replace("}", "}}"),
                        variables={},
                        optimize=message.optimize,
                        id=message.id,
                        content=message.content,
                    )
                )
        return rendered

    def _call_kwargs(
        self, variables: Optional[dict[str, Any]], model: Any, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        resolved_model = model if model is not None else self.model
        if resolved_model is None:
            raise PromptError(
                f"Prompt '{self.name}' has no model — set `model:` in the "
                "config or pass model= at call time."
            )
        kwargs: dict[str, Any] = {**self.params, **overrides}
        kwargs["model"] = resolved_model
        kwargs["messages"] = self.render(variables)
        if self.output is not None and "output" not in kwargs:
            kwargs["output"] = Output.object(
                schema=self.output.schema_,
                name=self.output.name,
                description=self.output.description,
            )
        return kwargs

    async def generate(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        **overrides: Any,
    ):
        """Render and run generate_text with this prompt's config.
        `overrides` are generate_text kwargs and win over `params`."""
        return await generate_text(**self._call_kwargs(variables, model, overrides))

    def stream(
        self,
        variables: Optional[dict[str, Any]] = None,
        *,
        model: Any = None,
        **overrides: Any,
    ):
        """Render and run stream_text with this prompt's config."""
        return stream_text(**self._call_kwargs(variables, model, overrides))


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
