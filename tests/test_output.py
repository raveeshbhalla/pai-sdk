from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from conftest import FakeModel
from model_message import (
    ProviderResult,
    ResponseMetadata,
    TextPart,
    Usage,
)
from model_message.errors import NoObjectGeneratedError
from model_message.generate import generate_text, stream_text
from model_message.output import (
    GenerateObjectResult,
    Output,
    StreamObjectResult,
    generate_object,
    parse_partial_json,
    stream_object,
)


class Person(BaseModel):
    name: str
    age: int


DICT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def _json_step(payload: dict) -> ProviderResult:
    return ProviderResult(
        content=[TextPart(text=json.dumps(payload))],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetadata(id="resp_1", model_id="fake-1"),
    )


# ---------------------------------------------------------------------------
# Output specs / response_format
# ---------------------------------------------------------------------------


def test_text_output_spec_has_no_response_format():
    spec = Output.text()
    assert spec.response_format() is None
    assert spec.parse("hello") == "hello"


def test_object_spec_response_format_pydantic():
    spec = Output.object(Person, name="person")
    fmt = spec.response_format()
    assert fmt["type"] == "json"
    assert fmt["name"] == "person"
    assert fmt["schema"]["properties"]["name"]["type"] == "string"


def test_object_spec_response_format_dict_schema():
    spec = Output.object(DICT_SCHEMA)
    fmt = spec.response_format()
    assert fmt["type"] == "json"
    assert fmt["schema"] is DICT_SCHEMA
    assert "name" not in fmt


@pytest.mark.asyncio
async def test_response_format_reaches_call_options():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    await generate_text(model=model, prompt="x", output=Output.object(Person, name="p"))
    assert len(model.calls) == 1
    fmt = model.calls[0].response_format
    assert fmt is not None
    assert fmt["type"] == "json"
    assert fmt["name"] == "p"


@pytest.mark.asyncio
async def test_no_output_means_no_response_format():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = await generate_text(model=model, prompt="x")
    assert model.calls[0].response_format is None
    assert result.output is None


# ---------------------------------------------------------------------------
# parse / validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pydantic_schema_produces_typed_instance():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = await generate_text(model=model, prompt="x", output=Output.object(Person))
    assert isinstance(result.output, Person)
    assert result.output.name == "Ada"
    assert result.output.age == 36


@pytest.mark.asyncio
async def test_dict_schema_produces_dict():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = await generate_text(
        model=model, prompt="x", output=Output.object(DICT_SCHEMA)
    )
    assert result.output == {"name": "Ada", "age": 36}
    assert isinstance(result.output, dict)


@pytest.mark.asyncio
async def test_malformed_json_raises_no_object_generated():
    step = ProviderResult(
        content=[TextPart(text="not json at all")],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        response=ResponseMetadata(id="r", model_id="fake-1"),
    )
    model = FakeModel(steps=[step])
    with pytest.raises(NoObjectGeneratedError) as exc:
        await generate_text(model=model, prompt="x", output=Output.object(Person))
    assert exc.value.text == "not json at all"
    assert exc.value.finish_reason == "stop"
    assert exc.value.usage is not None


@pytest.mark.asyncio
async def test_schema_validation_failure_raises():
    # Valid JSON, wrong type for age.
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": "old"})])
    with pytest.raises(NoObjectGeneratedError):
        await generate_text(model=model, prompt="x", output=Output.object(Person))


# ---------------------------------------------------------------------------
# parse_partial_json
# ---------------------------------------------------------------------------


def test_parse_partial_json_full():
    value, state = parse_partial_json('{"a": 1}')
    assert state == "successful-parse"
    assert value == {"a": 1}


def test_parse_partial_json_failed():
    value, state = parse_partial_json("")
    assert state == "failed-parse"
    assert value is None


def test_parse_partial_json_truncation_points():
    full = '{"name": "Ada", "age": 36, "tags": ["x", "y"], "meta": {"ok": true}}'
    seen = []
    for i in range(1, len(full) + 1):
        value, state = parse_partial_json(full[:i])
        if state in ("successful-parse", "repaired-parse"):
            assert isinstance(value, dict)
            seen.append(value)
        else:
            assert value is None
    # the last (complete) parse must equal the full object
    final_value, final_state = parse_partial_json(full)
    assert final_state == "successful-parse"
    assert final_value == json.loads(full)
    # partials must have been observed and grow toward the full object
    assert {"name": "Ada"} in seen
    assert any(v.get("age") == 36 for v in seen)
    assert any(v.get("tags") == ["x", "y"] for v in seen)


def test_parse_partial_json_unterminated_string():
    value, state = parse_partial_json('{"name": "Ad')
    assert state == "repaired-parse"
    assert value == {"name": "Ad"}


def test_parse_partial_json_trailing_comma_and_dangling_key():
    value, state = parse_partial_json('{"a": 1, "b": 2, ')
    assert state == "repaired-parse"
    assert value == {"a": 1, "b": 2}

    value, state = parse_partial_json('{"a": 1, "b":')
    assert state == "repaired-parse"
    assert value == {"a": 1}


def test_parse_partial_json_nested_brackets():
    value, state = parse_partial_json('{"items": [1, 2, {"k": "v"')
    assert state == "repaired-parse"
    assert value == {"items": [1, 2, {"k": "v"}]}


# ---------------------------------------------------------------------------
# streaming partial output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_output_stream_yields_growing_partials():
    payload = {"name": "Ada", "age": 36}
    model = FakeModel(steps=[_json_step(payload)])
    result = stream_text(model=model, prompt="x", output=Output.object(Person))

    partials = [p async for p in result.partial_output_stream]
    assert partials  # at least one partial yielded
    # all are dicts (not validated against the schema)
    assert all(isinstance(p, dict) for p in partials)
    # consecutive partials are distinct (only yield on change)
    for a, b in zip(partials, partials[1:]):
        assert a != b
    # final partial equals the full object
    assert partials[-1] == payload


@pytest.mark.asyncio
async def test_stream_output_awaitable_typed():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = stream_text(model=model, prompt="x", output=Output.object(Person))
    obj = await result.output
    assert isinstance(obj, Person)
    assert obj.name == "Ada"


@pytest.mark.asyncio
async def test_stream_output_malformed_raises():
    step = ProviderResult(
        content=[TextPart(text="garbage")],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        response=ResponseMetadata(id="r", model_id="fake-1"),
    )
    model = FakeModel(steps=[step])
    result = stream_text(model=model, prompt="x", output=Output.object(Person))
    with pytest.raises(NoObjectGeneratedError):
        await result.output


# ---------------------------------------------------------------------------
# generate_object / stream_object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_object_happy_path():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = await generate_object(model=model, schema=Person, prompt="x")
    assert isinstance(result, GenerateObjectResult)
    assert isinstance(result.object, Person)
    assert result.object.name == "Ada"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 15
    assert result.total_usage.total_tokens == 15
    # response_format propagated
    assert model.calls[0].response_format["type"] == "json"


@pytest.mark.asyncio
async def test_generate_object_dict_schema():
    model = FakeModel(steps=[_json_step({"name": "Ada", "age": 36})])
    result = await generate_object(model=model, schema=DICT_SCHEMA, prompt="x")
    assert result.object == {"name": "Ada", "age": 36}


@pytest.mark.asyncio
async def test_stream_object_happy_path():
    payload = {"name": "Ada", "age": 36}
    model = FakeModel(steps=[_json_step(payload)])
    result = stream_object(model=model, schema=Person, prompt="x")
    assert isinstance(result, StreamObjectResult)

    partials = [p async for p in result.partial_object_stream]
    assert partials[-1] == payload
    obj = await result.object
    assert isinstance(obj, Person)
    assert obj.age == 36
    assert (await result.finish_reason) == "stop"
    assert (await result.usage).total_tokens == 15
