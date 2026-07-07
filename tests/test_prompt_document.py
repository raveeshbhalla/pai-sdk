"""The portable prompt document: spec version, skills, tool outputs, Pydantic
projections, per-message rendering, and optimize_anything-shaped candidates."""

import json

import pytest
from pydantic import BaseModel

from pai_sdk import (
    OptimizerTarget,
    apply_candidate,
    apply_optimizer_target,
    canonical_prompt_json,
    list_optimizer_targets,
    read_candidate,
    read_optimizer_target,
    span_feedback,
    tool,
)
from pai_sdk.prompts import PROMPT_SPEC_VERSION, Prompt, PromptError, load_prompt

from conftest import FakeModel, text_step, tool_step

# ---------------------------------------------------------------------------
# Spec version
# ---------------------------------------------------------------------------


def test_spec_version_defaults_and_round_trips():
    prompt = load_prompt({"name": "x", "user": "Q: {{q}}"})
    assert prompt.spec_version == PROMPT_SPEC_VERSION
    dumped = prompt.to_dict()
    assert dumped["specVersion"] == "pai.prompt.v1"
    assert load_prompt(dumped).content_hash() == prompt.content_hash()


def test_unknown_spec_version_rejected():
    with pytest.raises(Exception, match="pai.prompt.v1"):
        load_prompt({"name": "x", "specVersion": "pai.prompt.v2", "user": "hi"})


def test_to_dict_omits_empty_containers():
    dumped = load_prompt({"name": "x", "user": "hi"}).to_dict()
    assert "params" not in dumped
    assert "tools" not in dumped
    assert "skills" not in dumped


def test_canonical_json_is_sorted_and_compact():
    text = canonical_prompt_json({"b": 1, "a": {"y": "é", "x": [1, 2]}})
    assert text == '{"a":{"x":[1,2],"y":"é"},"b":1}'


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

SKILL_CONFIG = {
    "name": "support",
    "input": {"company": "string", "ticket": "string"},
    "system": "You support {{company}}.",
    "user": "Ticket: {{ticket}}",
    "skills": {
        "escalation": {
            "description": 'Apply when legal or refunds come up (e.g. {"legal": true}).',
            "instructions": "Escalate to a human. Mention {{company}} policy first.",
        },
        "tone": {
            "description": "Always applies.",
            "instructions": "Stay warm and concise.",
        },
    },
}


def test_skills_render_after_last_system_message():
    prompt = load_prompt(SKILL_CONFIG)
    messages = prompt.render({"company": "Acme", "ticket": "Sue you!"})
    assert [m.role for m in messages] == ["system", "system", "system", "user"]
    assert [m.id for m in messages] == ["system", "skill:escalation", "skill:tone", "user"]
    escalation = messages[1]
    assert escalation.content.startswith("Skill: escalation\n")
    # description braces render literally; instructions variables bind
    assert '{"legal": true}' in escalation.content
    assert "Mention Acme policy first." in escalation.content
    assert escalation.variables == {"company": "Acme"}


def test_skills_lead_when_no_system_message():
    prompt = load_prompt(
        {
            "name": "x",
            "user": "Q: {{q}}",
            "skills": {"s": {"description": "d", "instructions": "i"}},
        }
    )
    rendered = prompt.render({"q": "hi"})
    assert [m.id for m in rendered] == ["skill:s", "user"]


def test_skill_variables_join_the_input_contract():
    prompt = load_prompt(SKILL_CONFIG)
    assert prompt.variables == ["company", "ticket"]
    # input schema must cover skill variables too
    with pytest.raises(Exception, match="declare template variables"):
        load_prompt(
            {
                "name": "x",
                "input": {"q": "string"},
                "user": "Q: {{q}}",
                "skills": {
                    "s": {"description": "d", "instructions": "Use {{undeclared}}."}
                },
            }
        )


def test_skill_name_and_id_collisions_rejected():
    with pytest.raises(Exception, match="Invalid skill name"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "skills": {"bad name!": {"description": "d", "instructions": "i"}},
            }
        )
    with pytest.raises(Exception, match="unique"):
        load_prompt(
            {
                "name": "x",
                "messages": [
                    {"id": "skill:s", "role": "system", "content": "collide"},
                    {"id": "u", "role": "user", "content": "hi"},
                ],
                "skills": {"s": {"description": "d", "instructions": "i"}},
            }
        )


def test_skill_mutations_follow_the_contract():
    prompt = load_prompt(SKILL_CONFIG)
    evolved = prompt.with_skill_description("tone", "Apply on angry tickets.")
    assert evolved.skills["tone"].description == "Apply on angry tickets."
    assert prompt.skills["tone"].description == "Always applies."  # non-destructive
    assert evolved.content_hash() != prompt.content_hash()

    evolved = prompt.with_skill_instructions(
        "escalation", "Hand off to a person; cite {{company}} policy."
    )
    assert "Hand off" in evolved.skills["escalation"].instructions
    with pytest.raises(PromptError, match="preserve the variable set"):
        prompt.with_skill_instructions("escalation", "Hand off to a person.")
    with pytest.raises(PromptError, match="No skill named"):
        prompt.with_skill_description("nope", "x")


def test_skills_round_trip():
    prompt = load_prompt(SKILL_CONFIG)
    dumped = prompt.to_dict()
    assert dumped["skills"]["escalation"]["instructions"].startswith("Escalate")
    assert load_prompt(dumped).content_hash() == prompt.content_hash()


# ---------------------------------------------------------------------------
# Tool output schemas
# ---------------------------------------------------------------------------


def test_tool_output_schema_forms_and_round_trip():
    prompt = load_prompt(
        {
            "name": "x",
            "user": "Q: {{q}}",
            "tools": {
                "get_weather": {
                    "description": "Get weather.",
                    "input": {"city": "string"},
                    "output": {"temp_f": "number", "conditions": "string"},
                },
                "search": {
                    "description": "Search.",
                    "input": {"query": "string"},
                    "output": {"schema": {"type": "array", "items": {"type": "string"}}},
                },
            },
        }
    )
    weather = prompt.tools["get_weather"].output_schema()
    assert weather["properties"]["temp_f"] == {"type": "number"}
    assert prompt.tools["search"].output_schema() == {
        "type": "array",
        "items": {"type": "string"},
    }
    dumped = prompt.to_dict()
    assert dumped["tools"]["get_weather"]["output"] == {
        "temp_f": "number",
        "conditions": "string",
    }
    assert load_prompt(dumped).content_hash() == prompt.content_hash()


# ---------------------------------------------------------------------------
# Pydantic models as schemas (code-first projection)
# ---------------------------------------------------------------------------


class TriageInput(BaseModel):
    company: str
    ticket: str


class TriageOutput(BaseModel):
    urgency: str
    summary: str


class WeatherInput(BaseModel):
    city: str


class WeatherReport(BaseModel):
    temp_f: float


def test_pydantic_models_compile_into_the_document():
    prompt = Prompt(
        name="triage",
        input=TriageInput,
        output=TriageOutput,
        tools={"get_weather": {"description": "w", "input": WeatherInput, "output": WeatherReport}},
        messages=[
            {"id": "system", "role": "system", "template": "Support {{company}}."},
            {"id": "user", "role": "user", "template": "Ticket: {{ticket}}"},
        ],
    )
    dumped = prompt.to_dict()
    # the document carries plain JSON Schema only
    assert dumped["input"]["schema"]["properties"]["company"] == {
        "title": "Company",
        "type": "string",
    }
    assert dumped["output"]["schema"]["additionalProperties"] is False
    assert set(dumped["output"]["schema"]["required"]) == {"urgency", "summary"}
    assert dumped["tools"]["get_weather"]["input"]["schema"]["properties"]["city"][
        "type"
    ] == "string"
    assert json.dumps(dumped)  # JSON-able all the way down
    assert load_prompt(json.loads(json.dumps(dumped))).content_hash() == prompt.content_hash()


async def test_pydantic_output_model_parses_result():
    prompt = Prompt(
        name="triage",
        output=TriageOutput,
        messages=[{"id": "user", "role": "user", "template": "Ticket: {{ticket}}"}],
    )
    model = FakeModel(steps=[text_step('{"urgency": "high", "summary": "ok"}')])
    result = await prompt.generate({"ticket": "It broke"}, model=model)
    assert isinstance(result.output, TriageOutput)
    assert result.output.urgency == "high"
    # loaded-from-data documents still parse to dicts
    loaded = load_prompt(prompt.to_dict())
    model2 = FakeModel(steps=[text_step('{"urgency": "low", "summary": "s"}')])
    result2 = await loaded.generate({"ticket": "x"}, model=model2)
    assert result2.output == {"urgency": "low", "summary": "s"}


# ---------------------------------------------------------------------------
# render_message
# ---------------------------------------------------------------------------


def test_render_message_single_turn():
    prompt = load_prompt(SKILL_CONFIG)
    message = prompt.render_message("user", {"ticket": "New issue"})
    assert message.role == "user"
    assert message.content == "Ticket: New issue"
    assert message.id == "user"
    skill = prompt.render_message("skill:tone")
    assert skill.role == "system"
    assert "Stay warm" in skill.content
    with pytest.raises(PromptError, match="missing variables"):
        prompt.render_message("user")
    with pytest.raises(PromptError, match="No message with id"):
        prompt.render_message("nope")


# ---------------------------------------------------------------------------
# tool(fn) — code-first tool projection
# ---------------------------------------------------------------------------


def test_tool_function_form_infers_schemas():
    def get_weather(city: str, units: str = "F") -> WeatherReport:
        return WeatherReport(temp_f=72.0)

    t = tool(get_weather, description="Get weather.")
    assert t.name == "get_weather"
    schema = t.json_schema()
    assert schema["properties"]["city"] == {"title": "City", "type": "string"}
    assert "city" in schema["required"] and "units" not in schema.get("required", [])
    assert t.output_json_schema()["properties"]["temp_f"]["type"] == "number"
    assert t.description == "Get weather."


def test_tool_function_form_rejects_var_args():
    with pytest.raises(TypeError, match="args"):
        tool(lambda *args: None)  # noqa: ARG005


async def test_tool_function_form_executes_in_the_loop():
    calls = []

    def get_weather(city: str) -> str:
        calls.append(city)
        return f"72F in {city}"

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("Done."),
        ]
    )
    from pai_sdk import generate_text, step_count_is

    result = await generate_text(
        model=model,
        prompt="weather?",
        tools={"get_weather": tool(get_weather, description="Get weather.")},
        stop_when=step_count_is(3),
    )
    assert calls == ["Paris"]
    assert result.text == "Done."


async def test_tool_function_form_with_explicit_schema_gets_parsed_input():
    seen = []

    def get_weather(input: WeatherInput) -> str:
        seen.append(input)
        return f"72F in {input.city}"

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Oslo"}),
            text_step("Done."),
        ]
    )
    from pai_sdk import generate_text, step_count_is

    await generate_text(
        model=model,
        prompt="weather?",
        tools={
            "get_weather": tool(
                get_weather, description="w", input_schema=WeatherInput
            )
        },
        stop_when=step_count_is(3),
    )
    assert isinstance(seen[0], WeatherInput) and seen[0].city == "Oslo"


async def test_tool_fn_placed_directly_in_a_prompt():
    calls = []

    def get_weather(city: str) -> str:
        calls.append(city)
        return f"72F in {city}"

    prompt = Prompt(
        name="weather",
        tools={"get_weather": tool(get_weather, description="Get weather.")},
        messages=[{"id": "user", "role": "user", "template": "Weather in {{city}}?"}],
        max_steps=3,
    )
    # the interface compiled into the document; behavior stayed out of it
    dumped = prompt.to_dict()
    assert dumped["tools"]["get_weather"]["input"]["schema"]["properties"]["city"][
        "type"
    ] == "string"
    assert dumped["tools"]["get_weather"]["output"] == {"schema": {"type": "string"}}
    assert "bound_execute" not in json.dumps(dumped)
    assert load_prompt(json.loads(json.dumps(dumped))).content_hash() == prompt.content_hash()

    # the execute function auto-binds as the handler
    model = FakeModel(
        steps=[tool_step("get_weather", tool_input={"city": "Paris"}), text_step("Done.")]
    )
    result = await prompt.generate({"city": "Paris"}, model=model)
    assert calls == ["Paris"]
    assert result.text == "Done."

    # call-time handlers still win
    model2 = FakeModel(
        steps=[tool_step("get_weather", tool_input={"city": "Oslo"}), text_step("Done.")]
    )
    override_calls = []
    await prompt.generate(
        {"city": "Oslo"},
        model=model2,
        handlers={"get_weather": lambda input: override_calls.append(input) or "n/a"},
    )
    assert override_calls == [{"city": "Oslo"}] and calls == ["Paris"]


# ---------------------------------------------------------------------------
# optimize_anything-shaped candidates
# ---------------------------------------------------------------------------

CANDIDATE_CONFIG = {
    "name": "triage",
    "system": "Triage for {{company}}.",
    "user": "Ticket: {{ticket}}",
    "tools": {"lookup": {"description": "Look up the customer.", "input": {"id": "string"}}},
    "skills": {"refunds": {"description": "Refund asks.", "instructions": "Refund per {{company}} policy."}},
}


def test_target_addresses_round_trip():
    for target in [
        OptimizerTarget.message_template("system"),
        OptimizerTarget.tool_description("lookup"),
        OptimizerTarget.skill_description("refunds"),
        OptimizerTarget.skill_instructions("refunds"),
    ]:
        assert OptimizerTarget.from_address(target.address) == target
    with pytest.raises(PromptError, match="Invalid"):
        OptimizerTarget.from_address("skill:refunds")
    with pytest.raises(PromptError, match="Invalid"):
        OptimizerTarget.from_address("bogus:x")


def test_list_targets_includes_skills():
    prompt = load_prompt(CANDIDATE_CONFIG)
    addresses = {t.address for t in list_optimizer_targets(prompt)}
    assert addresses == {
        "message:system",
        "message:user",
        "tool:lookup",
        "skill:refunds.description",
        "skill:refunds.instructions",
    }


def test_read_and_apply_candidate_multi_target():
    prompt = load_prompt(CANDIDATE_CONFIG)
    targets = ["message:system", "tool:lookup", "skill:refunds.instructions"]
    seed = read_candidate(prompt, targets)
    assert seed == {
        "message:system": "Triage for {{company}}.",
        "tool:lookup": "Look up the customer.",
        "skill:refunds.instructions": "Refund per {{company}} policy.",
    }

    evolved = apply_candidate(
        prompt,
        {
            "message:system": "You are {{company}}'s decisive triage lead.",
            "tool:lookup": "Look up the customer BEFORE answering account questions.",
            "skill:refunds.instructions": "Refund within {{company}} policy limits; escalate above.",
        },
    )
    assert evolved.content_hash() != prompt.content_hash()
    assert prompt.to_dict() != evolved.to_dict()  # non-destructive
    # the optimized document is plain data, ready to persist and reload
    reloaded = load_prompt(json.loads(json.dumps(evolved.to_dict())))
    assert reloaded.content_hash() == evolved.content_hash()

    # contract violations surface per-address
    with pytest.raises(PromptError, match="preserve the variable set"):
        apply_candidate(prompt, {"message:system": "No variables here."})
    with pytest.raises(PromptError, match="must be a string"):
        apply_candidate(prompt, {"message:system": 42})


def test_single_target_helpers_accept_addresses():
    prompt = load_prompt(CANDIDATE_CONFIG)
    assert read_optimizer_target(prompt, "skill:refunds.description") == "Refund asks."
    evolved = apply_optimizer_target(prompt, "skill:refunds.description", "Money-back asks.")
    assert evolved.skills["refunds"].description == "Money-back asks."


# ---------------------------------------------------------------------------
# span_feedback — trace-derived ASI for reflective optimizers
# ---------------------------------------------------------------------------


async def test_span_feedback_from_generated_trace():
    prompt = load_prompt(
        {
            "name": "weather",
            "user": "Weather in {{city}}?",
            "tools": {"get_weather": {"description": "w", "input": {"city": "string"}}},
            "max_steps": 3,
        }
    )

    def get_weather(input):
        raise RuntimeError("service down")

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Oslo"}),
            text_step("Could not fetch the weather."),
        ]
    )
    result = await prompt.generate_trace(
        {"city": "Oslo"}, model=model, handlers={"get_weather": get_weather}
    )
    feedback = span_feedback(result.trace.spans[0])
    assert feedback["finish_reason"] == "stop"
    assert "service down" in feedback["tool_errors"]
    assert feedback["output"] == "Could not fetch the weather."
    assert "transcript" in feedback and "Oslo" in feedback["transcript"]
    assert all(isinstance(value, str) for value in feedback.values())

    compact = span_feedback(result.trace.spans[0], include_transcript=False)
    assert "transcript" not in compact


# ---------------------------------------------------------------------------
# Security review regressions
# ---------------------------------------------------------------------------


def test_source_model_not_settable_from_documents():
    benign = {"type": "object", "properties": {}, "additionalProperties": False}
    hostile = {"type": "object", "properties": {"x": {"type": "string"}}}
    with pytest.raises(Exception, match="code-only"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "output": {"schema": benign, "source_model": hostile},
            }
        )
    with pytest.raises(Exception, match="code-only"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "input": {"schema": benign, "source_model": hostile},
            }
        )


def test_bound_execute_not_settable_from_documents():
    with pytest.raises(Exception, match="code-only"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "tools": {
                    "lookup": {"input": {"id": "string"}, "bound_execute": "anything"}
                },
            }
        )


def test_skill_names_are_fully_anchored():
    with pytest.raises(Exception, match="Invalid skill name"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "skills": {"esc\n": {"description": "d", "instructions": "i"}},
            }
        )


def test_tool_schema_must_be_an_object():
    with pytest.raises(Exception, match="must be an object"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "tools": {"t": {"input": {"schema": "string"}}},
            }
        )
    with pytest.raises(Exception, match="must be an object"):
        load_prompt(
            {
                "name": "x",
                "user": "hi",
                "tools": {"t": {"output": {"schema": "string"}}},
            }
        )


def test_skill_declaration_order_is_not_semantic():
    base = {
        "name": "x",
        "system": "Base.",
        "user": "Q: {{q}}",
    }
    ab = load_prompt({**base, "skills": {
        "alpha": {"description": "a", "instructions": "A"},
        "beta": {"description": "b", "instructions": "B"},
    }})
    ba = load_prompt({**base, "skills": {
        "beta": {"description": "b", "instructions": "B"},
        "alpha": {"description": "a", "instructions": "A"},
    }})
    assert ab.content_hash() == ba.content_hash()
    render_ab = [(m.id, m.content) for m in ab.render({"q": "hi"})]
    render_ba = [(m.id, m.content) for m in ba.render({"q": "hi"})]
    assert render_ab == render_ba  # equal hash implies identical rendering
    assert [m_id for m_id, _ in render_ab] == ["system", "skill:alpha", "skill:beta", "user"]


def test_canonical_numbers_match_ecmascript():
    assert canonical_prompt_json(
        {"a": 0.00001, "b": 1e-7, "c": 1e21, "d": 1.0, "e": 123.456, "f": -0.0}
    ) == '{"a":0.00001,"b":1e-7,"c":1000000000000000000000,"d":1,"e":123.456,"f":0}'


def test_tool_fn_single_pydantic_param_gets_parsed_instance():
    seen = []

    def get_weather(req: WeatherInput) -> str:
        seen.append(req)
        return f"72F in {req.city}"

    t = tool(get_weather, description="w")
    assert t.input_schema is WeatherInput
    parsed = t.parse_input({"city": "Oslo"})
    assert isinstance(parsed, WeatherInput)


def test_tool_fn_rejects_unsupported_signatures():
    with pytest.raises(TypeError, match="positional-only"):
        def pos_only(x: str, /) -> str:
            return x
        tool(pos_only)

    with pytest.raises(TypeError, match="unbound method"):
        class Svc:
            def lookup(self, q: str) -> str:
                return q
        tool(Svc.lookup)


def test_redact_trace_content_scrubs_headers():
    from pai_sdk import redact_trace_content
    from pai_sdk.trace import Span, Trace

    span = Span(
        id="s", root_span_id="s", inputs={}, outputs={},
        messages=[],
        metadata={"response": {"headers": {"set-cookie": "secret"}}},
    )
    redacted = redact_trace_content(Trace(id="s", spans=[span]))
    assert redacted["spans"][0]["metadata"]["response"]["headers"] == {"redacted": True}
