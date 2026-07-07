"""Tool definitions — the Python analog of the AI SDK `tool()` helper.

A tool's input schema can be a JSON Schema dict or a Pydantic model class
(the Pydantic class plays the role Zod plays in TypeScript: it both produces
the JSON schema sent to the provider and validates/parses the model's input
before `execute` runs).

`tool(fn, description=...)` is the code-first shorthand: the tool name and
input/output schemas are inferred from the function signature and compile to
the same plain JSON Schema that prompt configs carry — the description stays
explicit because it is prompt text.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union, get_type_hints

from pydantic import BaseModel, ConfigDict, Field as PydanticField, TypeAdapter, create_model

from .errors import InvalidToolInputError
from .messages import (
    ContentOutput,
    ErrorJsonOutput,
    ErrorTextOutput,
    JsonOutput,
    TextOutput,
    ToolResultOutput,
)

InputSchema = Union[dict[str, Any], type[BaseModel], None]
OutputSchema = Union[dict[str, Any], type[BaseModel], None]
ExecuteFn = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    """A tool the model can call.

    If `execute` is provided, generate_text/stream_text run it automatically
    in the multi-step loop. Without `execute`, tool calls are returned to the
    caller (client-side tools) and the loop stops.
    """

    description: Optional[str] = None
    input_schema: InputSchema = None
    output_schema: OutputSchema = None
    execute: Optional[ExecuteFn] = None
    to_model_output: Optional[Callable[[Any], ToolResultOutput]] = None
    strict: Optional[bool] = None
    provider_options: Optional[dict[str, dict[str, Any]]] = None
    # Set from the dict key when passed to generate_text(tools={name: tool}).
    name: Optional[str] = field(default=None)

    def json_schema(self) -> dict[str, Any]:
        """The JSON Schema for this tool's input, as sent to providers."""
        if self.input_schema is None:
            return {"type": "object", "properties": {}, "additionalProperties": False}
        if isinstance(self.input_schema, dict):
            return self.input_schema
        return self.input_schema.model_json_schema()

    def output_json_schema(self) -> Optional[dict[str, Any]]:
        """The declared JSON Schema for this tool's result, if any.

        Output schemas are interface documentation (and prompt-config data);
        they are not enforced against `execute` return values at run time.
        """
        if self.output_schema is None:
            return None
        if isinstance(self.output_schema, dict):
            return self.output_schema
        return self.output_schema.model_json_schema()

    def parse_input(self, raw_input: Any) -> Any:
        """Validate raw model input. Returns a Pydantic instance when the
        schema is a Pydantic model, otherwise the input unchanged."""
        if isinstance(self.input_schema, type) and issubclass(
            self.input_schema, BaseModel
        ):
            try:
                return self.input_schema.model_validate(raw_input or {})
            except Exception as exc:
                raise InvalidToolInputError(self.name or "?", raw_input, exc) from exc
        return raw_input

    async def run(self, parsed_input: Any, options: "ToolCallOptions") -> Any:
        assert self.execute is not None
        sig = inspect.signature(self.execute)
        kwargs: dict[str, Any] = {}
        if "options" in sig.parameters:
            kwargs["options"] = options
        result = self.execute(parsed_input, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


@dataclass
class ToolCallOptions:
    """Second argument available to a tool's `execute` function."""

    tool_call_id: str
    messages: list[Any]  # list[ModelMessage] — conversation so far


def tool(
    func: Optional[ExecuteFn] = None,
    /,
    *,
    description: Optional[str] = None,
    input_schema: InputSchema = None,
    output_schema: OutputSchema = None,
    execute: Optional[ExecuteFn] = None,
    to_model_output: Optional[Callable[[Any], ToolResultOutput]] = None,
    strict: Optional[bool] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
) -> Tool:
    """Define a tool.

    `tool(description=..., input_schema=..., execute=...)` mirrors the AI SDK
    helper. `tool(fn, description=...)` is the code-first shorthand: the name
    comes from the function, and the input/output schemas are inferred from
    its signature (and compiled to plain JSON Schema when serialized into a
    prompt config). The description stays explicit — it is prompt text.
    """
    if func is not None:
        if execute is not None:
            raise TypeError("Pass either a function or execute=, not both.")
        return _tool_from_function(
            func,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            to_model_output=to_model_output,
            strict=strict,
            provider_options=provider_options,
        )
    return Tool(
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        to_model_output=to_model_output,
        strict=strict,
        provider_options=provider_options,
    )


def _schema_from_annotation(annotation: Any) -> Optional[dict[str, Any]]:
    if annotation is inspect.Signature.empty:
        return None
    if annotation is None:
        return {"type": "null"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    return TypeAdapter(annotation).json_schema()


def _input_model_from_function(fn: ExecuteFn) -> type[BaseModel]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 — fall back to raw annotations
        hints = getattr(fn, "__annotations__", {})
    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in inspect.signature(fn).parameters.items():
        if param_name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"Tool function '{getattr(fn, '__name__', fn)}' cannot use "
                "*args or **kwargs."
            )
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, PydanticField(default))
    name = getattr(fn, "__name__", "tool")
    return create_model(
        f"{name}_input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _tool_from_function(
    fn: ExecuteFn,
    *,
    description: Optional[str],
    input_schema: InputSchema,
    output_schema: OutputSchema,
    to_model_output: Optional[Callable[[Any], ToolResultOutput]],
    strict: Optional[bool],
    provider_options: Optional[dict[str, dict[str, Any]]],
) -> Tool:
    if not callable(fn):
        raise TypeError("tool(...) expects a callable.")
    if output_schema is None:
        output_schema = _schema_from_annotation(
            inspect.signature(fn).return_annotation
        )

    if input_schema is not None:
        # Explicit schema: single-argument convention, like execute=.
        execute = fn
    else:
        input_schema = _input_model_from_function(fn)

        def execute(parsed_input: Any) -> Any:
            data = (
                parsed_input.model_dump(mode="python")
                if isinstance(parsed_input, BaseModel)
                else dict(parsed_input or {})
            )
            return fn(**data)

    return Tool(
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        to_model_output=to_model_output,
        strict=strict,
        provider_options=provider_options,
        name=getattr(fn, "__name__", None),
    )


def output_to_model_output(tool_def: Optional[Tool], output: Any) -> ToolResultOutput:
    """Convert a tool's return value into a ToolResultOutput."""
    if tool_def is not None and tool_def.to_model_output is not None:
        return tool_def.to_model_output(output)
    if isinstance(
        output, (TextOutput, JsonOutput, ErrorTextOutput, ErrorJsonOutput, ContentOutput)
    ):
        return output
    if isinstance(output, str):
        return TextOutput(value=output)
    if isinstance(output, BaseModel):
        return JsonOutput(value=output.model_dump(mode="json"))
    return JsonOutput(value=output)


ToolSet = dict[str, Tool]
