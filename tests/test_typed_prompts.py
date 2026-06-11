"""Typed messages + Prompt configs (the GEPA substrate)."""

import json

import pytest

from pai_sdk import generate_text
from pai_sdk.prompts import Prompt, PromptError, load_prompt
from pai_sdk.serialize import dump_messages, load_messages
from pai_sdk.typed import (
    TemplateError,
    TypedSystemMessage,
    TypedUserMessage,
    extract_variables,
    render_template,
)

from conftest import FakeModel, text_step

# ---------------------------------------------------------------------------
# Template engine
# ---------------------------------------------------------------------------


def test_extract_variables_ordered_deduped():
    assert extract_variables("{a} then {b} then {a}") == ["a", "b"]
    assert extract_variables("no vars") == []
    assert extract_variables("escaped {{not_a_var}} but {real}") == ["real"]


@pytest.mark.parametrize(
    "bad", ["{}", "{0}", "{a.b}", "{a[0]}", "{a:>10}", "{a!r}", "{unclosed"]
)
def test_invalid_templates_rejected(bad):
    with pytest.raises(TemplateError):
        extract_variables(bad)


def test_render_template():
    assert render_template("Hi {name}!", {"name": "Ada", "extra": "ok"}) == "Hi Ada!"
    assert render_template("{{literal}} {x}", {"x": 1}) == "{literal} 1"
    with pytest.raises(TemplateError, match="name"):
        render_template("Hi {name}!", {})


# ---------------------------------------------------------------------------
# Typed messages
# ---------------------------------------------------------------------------


def test_typed_message_renders_and_round_trips():
    msg = TypedSystemMessage(
        template="You help with {topic}.",
        variables={"topic": "taxes"},
        optimize=True,
        id="instructions",
    )
    assert msg.content == "You help with taxes."
    dumped = dump_messages([msg])[0]
    assert dumped["template"] == "You help with {topic}."
    assert dumped["variables"] == {"topic": "taxes"}
    assert dumped["optimize"] is True
    assert dumped["id"] == "instructions"
    # generic load keeps the fields; typed re-validation reconstructs the class
    restored = TypedSystemMessage.model_validate(dumped)
    assert restored.template == msg.template
    assert load_messages([dumped])[0].content == "You help with taxes."


async def test_typed_messages_flow_through_engine():
    model = FakeModel(steps=[text_step("ok")])
    await generate_text(
        model=model,
        messages=[
            TypedSystemMessage(template="Audience: {aud}.", variables={"aud": "devs"}),
            TypedUserMessage(template="Q: {q}", variables={"q": "why?"}),
        ],
    )
    prompt = model.calls[0].prompt
    assert prompt[0].role == "system" and prompt[0].content == "Audience: devs."
    assert prompt[1].role == "user" and prompt[1].content == "Q: why?"


# ---------------------------------------------------------------------------
# Prompt configs
# ---------------------------------------------------------------------------

CONFIG = {
    "name": "triage",
    "version": 1,
    "model": "anthropic/claude-haiku-4-5",
    "params": {"temperature": 0.2, "max_output_tokens": 500},
    "output": {
        "schema": {
            "type": "object",
            "properties": {"urgency": {"type": "string"}},
            "required": ["urgency"],
            "additionalProperties": False,
        }
    },
    "messages": [
        {
            "id": "instructions",
            "role": "system",
            "optimize": True,
            "template": "You triage tickets for {company}. Be decisive.",
        },
        {"id": "policy", "role": "system", "content": "Never reveal {internal} data."},
        {"id": "ticket", "role": "user", "template": "Ticket: {ticket}"},
    ],
}


def test_prompt_introspection():
    prompt = load_prompt(CONFIG)
    assert prompt.variables == ["company", "ticket"]  # literal content excluded
    assert [m.id for m in prompt.optimizable_messages()] == ["instructions"]
    assert len(prompt.content_hash()) == 16


def test_prompt_render_produces_typed_messages():
    prompt = load_prompt(CONFIG)
    messages = prompt.render({"company": "Acme", "ticket": "It broke", "unused": 1})
    assert messages[0].content == "You triage tickets for Acme. Be decisive."
    assert messages[0].optimize is True
    assert messages[0].variables == {"company": "Acme"}  # only its own bindings
    assert messages[1].content == "Never reveal {internal} data."  # literal untouched
    assert messages[2].content == "Ticket: It broke"
    with pytest.raises(PromptError, match="company"):
        prompt.render({"ticket": "x"})


async def test_prompt_generate_through_engine():
    prompt = load_prompt(CONFIG)
    model = FakeModel(steps=[text_step('{"urgency": "high"}')])
    result = await prompt.generate(
        {"company": "Acme", "ticket": "It broke"}, model=model
    )
    options = model.calls[0]
    assert options.temperature == 0.2
    assert options.max_output_tokens == 500
    assert options.response_format["type"] == "json"
    assert result.output == {"urgency": "high"}
    # overrides win over params
    model2 = FakeModel(steps=[text_step('{"urgency": "low"}')])
    await prompt.generate(
        {"company": "A", "ticket": "B"}, model=model2, temperature=0.9
    )
    assert model2.calls[0].temperature == 0.9


def test_prompt_requires_model_somewhere():
    config = {**CONFIG, "messages": CONFIG["messages"]}
    config = json.loads(json.dumps(config))
    del config["model"]
    prompt = load_prompt(config)
    with pytest.raises(PromptError, match="no model"):
        prompt._call_kwargs({"company": "A", "ticket": "B"}, None, None, {})


# ---------------------------------------------------------------------------
# The optimization contract
# ---------------------------------------------------------------------------


def test_with_template_valid_mutation():
    prompt = load_prompt(CONFIG)
    evolved = prompt.with_template(
        "instructions",
        "You are {company}'s expert triage agent. Rank urgency precisely.",
    )
    assert evolved.content_hash() != prompt.content_hash()
    assert prompt.messages[0].template.endswith("Be decisive.")  # original untouched
    rendered = evolved.render({"company": "Acme", "ticket": "x"})
    assert "expert triage agent" in rendered[0].content


def test_with_template_rejects_variable_changes():
    prompt = load_prompt(CONFIG)
    with pytest.raises(PromptError, match="preserve the variable set"):
        prompt.with_template("instructions", "You triage tickets. Be decisive.")
    with pytest.raises(PromptError, match="preserve the variable set"):
        prompt.with_template(
            "instructions", "You triage for {company} at {severity} level."
        )


def test_with_template_rejects_non_optimizable_and_unknown():
    prompt = load_prompt(CONFIG)
    with pytest.raises(PromptError, match="not marked optimize"):
        prompt.with_template("ticket", "Changed: {ticket}")
    with pytest.raises(PromptError, match="No message with id"):
        prompt.with_template("nope", "x")


def test_prompt_round_trips_to_dict():
    prompt = load_prompt(CONFIG)
    assert load_prompt(prompt.to_dict()).content_hash() == prompt.content_hash()


# ---------------------------------------------------------------------------
# File / format loading
# ---------------------------------------------------------------------------


def test_load_prompt_files(tmp_path):
    json_path = tmp_path / "p.json"
    json_path.write_text(json.dumps(CONFIG))
    assert load_prompt(json_path).name == "triage"

    yaml = pytest.importorskip("yaml")
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(yaml.safe_dump(CONFIG))
    assert load_prompt(yaml_path).content_hash() == load_prompt(json_path).content_hash()

    with pytest.raises(PromptError, match="extension"):
        load_prompt(tmp_path / "p.txt")


def test_config_validation_errors():
    with pytest.raises(Exception, match="exactly one"):
        load_prompt({"name": "x", "messages": [{"role": "user"}]})
    with pytest.raises(Exception, match="unique"):
        load_prompt(
            {
                "name": "x",
                "messages": [
                    {"id": "a", "role": "user", "content": "1"},
                    {"id": "a", "role": "user", "content": "2"},
                ],
            }
        )
    with pytest.raises(Exception):  # extra="forbid" catches typos
        load_prompt({"name": "x", "mesages": [], "messages": []})


# ---------------------------------------------------------------------------
# Live: YAML prompt -> structured output -> mutate -> re-run (the GEPA loop)
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_prompt_config_live_gepa_loop(tmp_path):
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("requires ANTHROPIC_API_KEY")
    yaml = pytest.importorskip("yaml")

    config = {
        "name": "sentiment",
        "model": "anthropic/claude-haiku-4-5",
        "params": {"max_output_tokens": 300},
        "output": {
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"},
                },
                "required": ["sentiment", "confidence"],
                "additionalProperties": False,
            }
        },
        "messages": [
            {
                "id": "instructions",
                "role": "system",
                "optimize": True,
                "template": "Classify the sentiment of {kind} feedback.",
            },
            {"id": "input", "role": "user", "template": "Feedback: {text}"},
        ],
    }
    path = tmp_path / "sentiment.yaml"
    path.write_text(yaml.safe_dump(config))
    prompt = load_prompt(path)

    variables = {"kind": "customer", "text": "This product is fantastic, I love it!"}
    result = await prompt.generate(variables)
    assert result.output["sentiment"] == "positive"

    # reflective mutation: rewrite instructions (variables preserved), re-run
    evolved = prompt.with_template(
        "instructions",
        "You are an expert analyst of {kind} feedback. Classify sentiment "
        "carefully and report your confidence honestly.",
    )
    assert evolved.content_hash() != prompt.content_hash()
    result2 = await evolved.generate(variables)
    assert result2.output["sentiment"] == "positive"
    # the trace records which instructions produced this rollout
    trace = dump_messages(result2.response.messages)
    rendered = evolved.render(variables)
    assert "expert analyst" in rendered[0].content


# ---------------------------------------------------------------------------
# Simple-form configs (top-level system/user, output type shorthand)
# ---------------------------------------------------------------------------

from pai_sdk.prompts import compile_output_shorthand  # noqa: E402


def test_output_shorthand_compiles():
    schema = compile_output_shorthand(
        {
            "urgency": ["low", "medium", "high"],
            "summary": "string",
            "score": "number",
            "count": "integer",
            "done": "boolean",
            "tags": "string[]",
            "untyped": None,
            "user": {"name": "string", "id": "integer"},
        }
    )
    props = schema["properties"]
    assert props["urgency"] == {"enum": ["low", "medium", "high"]}
    assert props["summary"] == {"type": "string"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["untyped"] == {"type": "string"}
    assert props["user"]["properties"]["id"] == {"type": "integer"}
    assert props["user"]["additionalProperties"] is False
    assert schema["required"] == list(props.keys())
    assert schema["additionalProperties"] is False
    with pytest.raises(PromptError, match="Unknown output field type"):
        compile_output_shorthand({"x": "strang"})


def test_simple_form_config():
    prompt = load_prompt(
        {
            "name": "triage",
            "model": "anthropic/claude-haiku-4-5",
            "output": {"urgency": ["low", "high"]},
            "system": "You triage tickets for {company}. Be decisive.",
            "user": "Ticket: {ticket}",
        }
    )
    # system/user become messages; system is optimizable by default
    assert [m.id for m in prompt.messages] == ["system", "user"]
    assert prompt.messages[0].optimize is True
    assert prompt.messages[1].optimize is False
    # output shorthand compiled to a strict JSON Schema
    assert prompt.output.schema_["properties"]["urgency"] == {"enum": ["low", "high"]}
    assert prompt.output.schema_["additionalProperties"] is False
    # the optimization contract works on the simple form
    evolved = prompt.with_template("system", "Best triager for {company} ever.")
    assert "Best triager" in evolved.messages[0].template
    with pytest.raises(PromptError, match="not marked optimize"):
        prompt.with_template("user", "changed {ticket}")


def test_simple_form_dict_control_and_conflicts():
    prompt = load_prompt(
        {
            "name": "x",
            "system": {"template": "Frozen {a}.", "optimize": False},
            "user": "Q: {q}",
        }
    )
    assert prompt.messages[0].optimize is False
    with pytest.raises(Exception, match="not both"):
        load_prompt(
            {
                "name": "x",
                "system": "s",
                "messages": [{"role": "user", "content": "u"}],
            }
        )
    with pytest.raises(Exception, match="needs messages"):
        load_prompt({"name": "x"})


# ---------------------------------------------------------------------------
# Unified generate()/stream() dispatch
# ---------------------------------------------------------------------------


async def test_unified_generate_and_stream_dispatch():
    from pydantic import BaseModel

    from pai_sdk import generate, stream

    class Out(BaseModel):
        answer: str

    model = FakeModel(steps=[text_step('{"answer": "42"}')])
    result = await generate(model=model, schema=Out, prompt="q")
    assert result.object.answer == "42"  # GenerateObjectResult path

    model = FakeModel(steps=[text_step("plain")])
    result = await generate(model=model, prompt="q")
    assert result.text == "plain"  # GenerateTextResult path

    model = FakeModel(steps=[text_step('{"answer": "x"}')])
    s = stream(model=model, schema=Out, prompt="q")
    assert (await s.object).answer == "x"

    model = FakeModel(steps=[text_step("hello")])
    s = stream(model=model, prompt="q")
    assert await s.text == "hello"


# ---------------------------------------------------------------------------
# The prompt-config JSON Schema (for editor validation of customer YAML)
# ---------------------------------------------------------------------------

from pai_sdk.prompts import PROMPT_CONFIG_SCHEMA  # noqa: E402

jsonschema = pytest.importorskip("jsonschema")

SIMPLE_FORM = {
    "name": "triage",
    "model": "anthropic/claude-haiku-4-5",
    "params": {"max_output_tokens": 500},
    "output": {"urgency": ["low", "high"], "tags": "string[]", "user": {"id": "integer"}},
    "system": "You triage tickets for {company}.",
    "user": "Ticket: {ticket}",
}


def _validates(config) -> bool:
    try:
        jsonschema.validate(config, PROMPT_CONFIG_SCHEMA)
        return True
    except jsonschema.ValidationError:
        return False


def test_schema_accepts_what_loader_accepts():
    # simple form, general form, and full-JSON-Schema output all validate
    assert _validates(SIMPLE_FORM)
    assert _validates(CONFIG)  # the general-form fixture from above
    assert _validates(
        {
            "name": "x",
            "output": {"schema": {"type": "object", "properties": {}}},
            "system": {"template": "Hi {a}", "optimize": False},
        }
    )
    # and the loader agrees on all three
    for config in (SIMPLE_FORM, CONFIG):
        load_prompt(json.loads(json.dumps(config)))


def test_schema_rejects_what_loader_rejects():
    bad_configs = [
        {"mesages": [], "name": "x"},                       # typo'd key
        {"name": "x", "messages": [{"role": "user"}]},      # no template/content
        {"name": "x", "messages": [
            {"role": "user", "template": "a", "content": "b"}  # both bodies
        ]},
        {"name": "x", "output": {"urgency": "strang"}, "system": "s"},  # bad type
        {"messages": [{"role": "user", "content": "hi"}]},  # missing name
    ]
    for config in bad_configs:
        assert not _validates(config), f"schema wrongly accepted: {config}"
        with pytest.raises(Exception):
            load_prompt(json.loads(json.dumps(config)))


def test_schema_file_ships_with_package():
    from pai_sdk.prompts import PROMPT_CONFIG_SCHEMA_PATH

    assert PROMPT_CONFIG_SCHEMA_PATH.exists()
    assert PROMPT_CONFIG_SCHEMA["title"] == "pai-sdk prompt config"


# ---------------------------------------------------------------------------
# Tools in prompt configs
# ---------------------------------------------------------------------------

TOOL_CONFIG = {
    "name": "weather-helper",
    "model": "anthropic/claude-haiku-4-5",
    "system": "Answer using tools when needed.",
    "user": "Question: {q}",
    "tools": {
        "get_weather": {
            "description": "Get current weather. Call when asked about conditions.",
            "optimize": True,
            "input": {"city": "string"},
        },
        "search_docs": {
            "description": "Search documentation.",
            "input": {"schema": {"type": "object", "properties": {"query": {"type": "string"}},
                                  "required": ["query"], "additionalProperties": False}},
        },
    },
    "tool_choice": "auto",
    "max_steps": 3,
}


def test_prompt_tool_parsing_and_schemas():
    prompt = load_prompt(TOOL_CONFIG)
    weather = prompt.tools["get_weather"]
    assert weather.input_schema()["properties"]["city"] == {"type": "string"}
    assert weather.input_schema()["additionalProperties"] is False
    docs = prompt.tools["search_docs"]
    assert docs.input_schema()["properties"]["query"] == {"type": "string"}  # full-schema form
    assert [*prompt.optimizable_tools()] == ["get_weather"]
    with pytest.raises(Exception, match="Unknown output field type"):
        load_prompt({**TOOL_CONFIG, "tools": {"x": {"input": {"a": "strang"}}}})


async def test_prompt_tools_reach_engine_and_loop():
    calls = []

    def get_weather(input):
        calls.append(input)
        return f"72F in {input['city']}"

    from conftest import tool_step

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("It is 72F in Paris."),
        ]
    )
    prompt = load_prompt(TOOL_CONFIG)
    result = await prompt.generate(
        {"q": "Weather in Paris?"}, model=model, handlers={"get_weather": get_weather}
    )
    # tool specs reached CallOptions
    options = model.calls[0]
    assert {s.name for s in options.tools} == {"get_weather", "search_docs"}
    weather_spec = next(s for s in options.tools if s.name == "get_weather")
    assert "Call when asked" in weather_spec.description
    assert weather_spec.input_schema["properties"]["city"] == {"type": "string"}
    assert options.tool_choice == "auto"
    # the loop ran (max_steps=3 allowed the second step) and the handler fired
    assert calls == [{"city": "Paris"}]
    assert result.text == "It is 72F in Paris."
    assert len(result.steps) == 2


async def test_prompt_tools_client_side_without_handler():
    from conftest import tool_step

    model = FakeModel(steps=[tool_step("get_weather", tool_input={"city": "Oslo"})])
    prompt = load_prompt(TOOL_CONFIG)
    result = await prompt.generate({"q": "Weather?"}, model=model)  # no handlers
    assert result.finish_reason == "tool-calls"
    assert result.tool_calls[0].tool_name == "get_weather"
    assert result.tool_results == []  # client-side: nothing executed
    assert len(model.calls) == 1


async def test_prompt_handlers_for_undeclared_tool_rejected():
    prompt = load_prompt(TOOL_CONFIG)
    with pytest.raises(PromptError, match="undeclared tools: get_wether"):
        await prompt.generate({"q": "x"}, model=FakeModel(steps=[text_step("ok")]),
                              handlers={"get_wether": lambda i: "?"})


def test_with_tool_description_contract():
    prompt = load_prompt(TOOL_CONFIG)
    evolved = prompt.with_tool_description(
        "get_weather", "Fetch live weather; always call before answering weather questions."
    )
    assert "always call" in evolved.tools["get_weather"].description
    assert prompt.tools["get_weather"].description.startswith("Get current")  # original untouched
    assert evolved.content_hash() != prompt.content_hash()
    # schema/name unchanged by construction
    assert evolved.tools["get_weather"].input_schema() == prompt.tools["get_weather"].input_schema()
    with pytest.raises(PromptError, match="not marked optimize"):
        prompt.with_tool_description("search_docs", "x")
    with pytest.raises(PromptError, match="No tool named"):
        prompt.with_tool_description("nope", "x")


def test_tools_round_trip_and_schema_agreement():
    prompt = load_prompt(TOOL_CONFIG)
    dumped = prompt.to_dict()
    assert dumped["tools"]["get_weather"]["input"] == {"city": "string"}  # shorthand preserved
    assert load_prompt(dumped).content_hash() == prompt.content_hash()
    # the packaged config schema accepts it...
    assert _validates(TOOL_CONFIG)
    assert _validates(dumped)
    # ...and rejects malformed tool configs that the loader also rejects
    bad = [
        {**TOOL_CONFIG, "tools": {"x": {"inputs": {}}}},        # typo'd key
        {**TOOL_CONFIG, "tool_choice": "always"},               # bad enum
        {**TOOL_CONFIG, "max_steps": 0},                        # below minimum
    ]
    for config in bad:
        assert not _validates(config), f"schema wrongly accepted {config}"
        with pytest.raises(Exception):
            load_prompt(json.loads(json.dumps(config)))


@pytest.mark.live
async def test_prompt_config_tool_loop_live():
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("requires ANTHROPIC_API_KEY")

    calls = []

    def get_weather(input):
        calls.append(input.get("city"))
        return f"It is 72F and sunny in {input.get('city')}."

    prompt = load_prompt(
        {
            "name": "weather-live",
            "model": "anthropic/claude-haiku-4-5",
            "params": {"max_output_tokens": 1000},
            "system": "Use the get_weather tool to answer weather questions.",
            "user": "What's the weather in {city}? Use the tool.",
            "tools": {
                "get_weather": {
                    "description": "Get the current weather for a city. "
                    "Call whenever asked about weather conditions.",
                    "optimize": True,
                    "input": {"city": "string"},
                }
            },
            "max_steps": 3,
        }
    )
    result = await prompt.generate(
        {"city": "Berlin"}, handlers={"get_weather": get_weather}
    )
    assert calls and "berlin" in calls[0].lower()
    assert "72" in result.text
    assert len(result.steps) >= 2
    assert result.finish_reason == "stop"
