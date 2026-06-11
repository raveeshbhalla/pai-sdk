"""Structured output — the AI SDK `Output` API plus generate_object/stream_object.

`Output.text()` and `Output.object(schema, ...)` mirror the AI SDK `Output`
namespace. An object spec produces the `response_format` dict for CallOptions
and parses/validates the model's text into a typed object (Pydantic instance
when the schema was a Pydantic class, otherwise a plain dict).

`parse_partial_json` is the analog of the AI SDK `parsePartialJson`: a
best-effort repair of incomplete JSON used to surface partial objects while
streaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Optional, Sequence, Union

from pydantic import BaseModel

from .errors import NoObjectGeneratedError
from .results import CallWarning, FinishReason, ResponseMetadata, Usage

ObjectSchema = Union[type[BaseModel], dict[str, Any]]

ParseState = str  # "successful-parse" | "repaired-parse" | "failed-parse"


def _strictify_schema(node: Any) -> Any:
    """Recursively prepare a Pydantic-generated JSON Schema for providers'
    strict modes: every object gets additionalProperties=false and all of its
    properties marked required (optionality stays expressible via null
    unions, which Pydantic already emits for Optional fields)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node.setdefault("additionalProperties", False)
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify_schema(value)
    elif isinstance(node, list):
        for value in node:
            _strictify_schema(value)
    return node


# ---------------------------------------------------------------------------
# Output specs
# ---------------------------------------------------------------------------


@dataclass
class TextOutputSpec:
    """`Output.text()` — plain text output (no response_format)."""

    type: str = "text"

    def response_format(self) -> Optional[dict[str, Any]]:
        return None

    def parse(self, text: str) -> str:
        return text


@dataclass
class ObjectOutputSpec:
    """`Output.object(schema, ...)` — JSON object output."""

    schema: ObjectSchema
    name: Optional[str] = None
    description: Optional[str] = None
    type: str = "object"

    def _json_schema(self) -> dict[str, Any]:
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # Providers' strict structured-output modes (OpenAI strict,
            # Anthropic output_config) require additionalProperties: false and
            # every property listed in `required`. We own the Pydantic->JSON
            # Schema conversion, so normalize it; raw dict schemas are the
            # caller's responsibility and pass through untouched.
            return _strictify_schema(self.schema.model_json_schema())
        return self.schema

    def response_format(self) -> dict[str, Any]:
        fmt: dict[str, Any] = {"type": "json", "schema": self._json_schema()}
        if self.name is not None:
            fmt["name"] = self.name
        if self.description is not None:
            fmt["description"] = self.description
        return fmt

    def parse(self, text: str) -> Any:
        """JSON-load and validate. Returns a Pydantic instance when the schema
        was a Pydantic class, else the parsed dict."""
        try:
            value = json.loads(text)
        except Exception as exc:  # noqa: BLE001 — surfaced as NoObjectGeneratedError
            raise NoObjectGeneratedError(
                "Could not parse the response as JSON.", text=text, cause=exc
            ) from exc
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            try:
                return self.schema.model_validate(value)
            except Exception as exc:  # noqa: BLE001
                raise NoObjectGeneratedError(
                    "Response did not match the schema.", text=text, cause=exc
                ) from exc
        return value


class Output:
    """The AI SDK `Output` namespace."""

    @staticmethod
    def text() -> TextOutputSpec:
        return TextOutputSpec()

    @staticmethod
    def object(
        schema: ObjectSchema,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ObjectOutputSpec:
        return ObjectOutputSpec(schema=schema, name=name, description=description)


# ---------------------------------------------------------------------------
# parse_partial_json — best-effort repair of incomplete JSON
# ---------------------------------------------------------------------------


def parse_partial_json(text: str) -> tuple[Any, ParseState]:
    """Parse possibly-incomplete JSON, AI SDK parsePartialJson analog.

    Returns (value_or_None, state) where state is one of
    "successful-parse" | "repaired-parse" | "failed-parse".
    """
    if text is None:
        return None, "failed-parse"

    try:
        return json.loads(text), "successful-parse"
    except Exception:  # noqa: BLE001 — try to repair
        pass

    repaired = _repair_json(text)
    if repaired is not None:
        try:
            return json.loads(repaired), "repaired-parse"
        except Exception:  # noqa: BLE001
            pass
    return None, "failed-parse"


def _repair_json(text: str) -> Optional[str]:
    """Close unterminated strings/arrays/objects and strip dangling tokens so
    the prefix of a JSON document becomes valid JSON."""
    stack: list[str] = []  # "{" / "[" expected closers tracked as "}" / "]"
    in_string = False
    escaped = False
    # Index just past the last structurally-complete value, used to roll back
    # over a dangling key/comma/partial literal at the very end.
    result: list[str] = []

    for ch in text:
        if in_string:
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
        elif ch == "{":
            stack.append("}")
            result.append(ch)
        elif ch == "[":
            stack.append("]")
            result.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            result.append(ch)
        else:
            result.append(ch)

    repaired = "".join(result)

    # Close an unterminated string.
    if in_string:
        if escaped:
            repaired = repaired[:-1]
        repaired += '"'

    # Strip trailing whitespace.
    repaired = repaired.rstrip()

    # Roll back over a dangling colon / partial literal / dangling key that
    # can't be completed, then strip trailing commas. Iterate because removing
    # one dangling piece can expose another.
    while repaired:
        stripped = repaired.rstrip()
        last = stripped[-1] if stripped else ""

        if last == ",":
            repaired = stripped[:-1]
            continue
        if last == ":":
            # Drop the key + colon (no value yet) back to the preceding token.
            repaired = _drop_last_key(stripped)
            continue
        # Dangling partial literal (number/true/false/null fragment) right
        # before a closer is needed — handled by attempting the close below.
        break

    # Append the needed closers.
    for closer in reversed(stack):
        repaired += closer

    # Final cleanup: if after closing there is still a trailing comma issue or
    # a fragment, try trimming partial bare literals.
    if not repaired:
        return None
    return repaired


def _drop_last_key(text: str) -> str:
    """Given text ending in `"key":`, drop the key and colon (and a leading
    comma if present) so the container is left empty or value-complete."""
    # text ends with ':'
    body = text[:-1].rstrip()
    # body should now end with the closing quote of the key; walk back to its
    # opening quote (one not preceded by a backslash).
    if body.endswith('"'):
        i = len(body) - 2
        while i >= 0:
            if body[i] == '"' and (i == 0 or body[i - 1] != "\\"):
                break
            i -= 1
        body = body[:i].rstrip()
    # drop a dangling comma now exposed
    if body.endswith(","):
        body = body[:-1].rstrip()
    return body


# ---------------------------------------------------------------------------
# generate_object / stream_object — thin wrappers over generate_text/stream_text
# ---------------------------------------------------------------------------


@dataclass
class GenerateObjectResult:
    """The result of generate_object() (AI SDK GenerateObjectResult)."""

    object: Any
    usage: Usage
    total_usage: Usage
    finish_reason: FinishReason
    response: ResponseMetadata
    warnings: list[CallWarning] = field(default_factory=list)
    provider_metadata: Optional[dict[str, dict[str, Any]]] = None


async def generate_object(
    *,
    model: Any,
    schema: ObjectSchema,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> GenerateObjectResult:
    """Generate a structured object — the AI SDK generateObject().

    A thin wrapper over generate_text() with output=Output.object(...).
    """
    from .generate import generate_text

    spec = Output.object(schema, name=name, description=description)
    result = await generate_text(model=model, output=spec, **kwargs)
    return GenerateObjectResult(
        object=result.output,
        usage=result.usage,
        total_usage=result.total_usage,
        finish_reason=result.finish_reason,
        response=result.response,
        warnings=result.warnings,
        provider_metadata=result.provider_metadata,
    )


class StreamObjectResult:
    """The result of stream_object() (AI SDK StreamObjectResult).

    A thin wrapper over a StreamTextResult configured with an object output
    spec. Exposes `partial_object_stream` plus awaitables `object`, `usage`,
    `finish_reason`, and `response`.
    """

    def __init__(self, stream_result: Any) -> None:
        self._result = stream_result

    @property
    def partial_object_stream(self) -> AsyncIterator[Any]:
        return self._result.partial_output_stream

    @property
    def object(self) -> Awaitable[Any]:
        return self._result.output

    @property
    def usage(self) -> Awaitable[Usage]:
        return self._result.usage

    @property
    def finish_reason(self) -> Awaitable[FinishReason]:
        return self._result.finish_reason

    @property
    def response(self) -> Awaitable[ResponseMetadata]:
        return self._result.response


def stream_object(
    *,
    model: Any,
    schema: ObjectSchema,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> StreamObjectResult:
    """Stream a structured object — the AI SDK streamObject().

    A thin wrapper over stream_text() with output=Output.object(...).
    """
    from .generate import stream_text

    spec = Output.object(schema, name=name, description=description)
    result = stream_text(model=model, output=spec, **kwargs)
    return StreamObjectResult(result)


# ---------------------------------------------------------------------------
# Unified entry points — dispatch on what the call expects
# ---------------------------------------------------------------------------


async def generate(
    *,
    model: Any,
    schema: Optional[ObjectSchema] = None,
    **kwargs: Any,
):
    """One entry point for text and structured generation.

    With `schema=`, behaves as generate_object() and returns a
    GenerateObjectResult (validated `.object`). Without it, behaves as
    generate_text() and returns a GenerateTextResult (`.text`, tools, steps —
    pass `output=Output.object(...)` for structured output WITH the full
    text-result surface).
    """
    if schema is not None:
        return await generate_object(model=model, schema=schema, **kwargs)
    from .generate import generate_text

    return await generate_text(model=model, **kwargs)


def stream(
    *,
    model: Any,
    schema: Optional[ObjectSchema] = None,
    **kwargs: Any,
):
    """Streaming twin of generate(): schema= -> stream_object()
    (StreamObjectResult with partial_object_stream), else stream_text()
    (StreamTextResult with text_stream/full_stream)."""
    if schema is not None:
        return stream_object(model=model, schema=schema, **kwargs)
    from .generate import stream_text

    return stream_text(model=model, **kwargs)
