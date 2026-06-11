"""Tool definitions — the Python analog of the AI SDK `tool()` helper.

A tool's input schema can be a JSON Schema dict or a Pydantic model class
(the Pydantic class plays the role Zod plays in TypeScript: it both produces
the JSON schema sent to the provider and validates/parses the model's input
before `execute` runs).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import BaseModel

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
    *,
    description: Optional[str] = None,
    input_schema: InputSchema = None,
    execute: Optional[ExecuteFn] = None,
    to_model_output: Optional[Callable[[Any], ToolResultOutput]] = None,
    strict: Optional[bool] = None,
    provider_options: Optional[dict[str, dict[str, Any]]] = None,
) -> Tool:
    """Define a tool (mirrors the AI SDK `tool()` helper)."""
    return Tool(
        description=description,
        input_schema=input_schema,
        execute=execute,
        to_model_output=to_model_output,
        strict=strict,
        provider_options=provider_options,
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
