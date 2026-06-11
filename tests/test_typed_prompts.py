"""Typed messages + Prompt configs (the GEPA substrate)."""

import json

import pytest

from model_message import generate_text
from model_message.prompts import Prompt, PromptError, load_prompt
from model_message.serialize import dump_messages, load_messages
from model_message.typed import (
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
        prompt._call_kwargs({"company": "A", "ticket": "B"}, None, {})


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
