"""PromptSpec: the typed code socket a prompt document plugs into."""

import json

import pytest
from pydantic import BaseModel

from pai_sdk import PromptError, load_prompt, prompt_spec, tool
from pai_sdk.optimization import apply_candidate

from conftest import FakeModel, text_step, tool_step


class TriageInput(BaseModel):
    company: str
    ticket: str


class TriageVerdict(BaseModel):
    urgency: str
    summary: str


def lookup_customer(customer_email: str) -> dict:
    return {"plan": "pro"}


def make_spec():
    return prompt_spec(
        name="support-triage",
        input=TriageInput,
        output=TriageVerdict,
        tools={
            "lookup_customer": tool(
                lookup_customer, description="Look up the customer's plan."
            )
        },
    )


def make_seed(spec):
    return spec.document(
        model="anthropic/claude-haiku-4-5",
        params={"maxOutputTokens": 500},
        system="You triage tickets for {{company}}. Be decisive.",
        user="Ticket: {{ticket}}",
        skills={"refunds": {"description": "Money-back asks.", "instructions": "Escalate refunds for {{company}}."}},
        maxSteps=3,
    )


# -- authoring: code -> document ---------------------------------------------


def test_document_derives_contract_sections_from_spec(tmp_path):
    spec = make_spec()
    seed = make_seed(spec)
    doc = seed.to_dict()
    assert doc["name"] == "support-triage"
    assert set(doc["input"]["schema"]["required"]) == {"company", "ticket"}
    assert set(doc["output"]["schema"]["required"]) == {"urgency", "summary"}
    assert doc["tools"]["lookup_customer"]["description"] == "Look up the customer's plan."
    assert "bound_execute" not in json.dumps(doc)

    path = seed.export(tmp_path / "triage.json")
    reloaded = load_prompt(path)
    assert reloaded.content_hash() == seed.content_hash()

    with pytest.raises(PromptError, match="derives"):
        spec.document(name="other", user="hi")


# -- binding: document -> code -------------------------------------------------


def test_load_binds_optimizer_output_and_stays_typed(tmp_path):
    spec = make_spec()
    seed = make_seed(spec)
    seed.export(tmp_path / "triage.json")

    # the optimizer's whole job: mutate text, preserve the contract
    optimized = apply_candidate(
        load_prompt(tmp_path / "triage.json"),
        {"message:system": "You are {{company}}'s decisive triage lead."},
    )
    optimized.export(tmp_path / "triage.optimized.json")

    bound = spec.load(tmp_path / "triage.optimized.json")
    assert bound.content_hash() == optimized.content_hash()


async def test_bound_generate_is_typed_and_handler_bound():
    spec = make_spec()
    bound = make_seed(spec)

    calls = []

    def handler_probe(customer_email: str) -> dict:
        calls.append(customer_email)
        return {"plan": "pro"}

    model = FakeModel(
        steps=[
            tool_step("lookup_customer", tool_input={"customer_email": "j@x.co"}),
            text_step('{"urgency": "high", "summary": "double charge"}'),
        ]
    )
    result = await bound.generate(
        TriageInput(company="Acme", ticket="Refund me $700."),
        model=model,
        handlers={"lookup_customer": lambda input: handler_probe(**input)},
    )
    assert calls == ["j@x.co"]  # call-time handler wins over spec handler
    assert isinstance(result.output, TriageVerdict)
    assert result.output.urgency == "high"

    # spec handlers auto-bind when none passed
    model2 = FakeModel(
        steps=[
            tool_step("lookup_customer", tool_input={"customer_email": "a@b.co"}),
            text_step('{"urgency": "low", "summary": "ok"}'),
        ]
    )
    result2 = await bound.generate(
        {"company": "Acme", "ticket": "hi"}, model=model2
    )
    assert isinstance(result2.output, TriageVerdict)


# -- the contract, enforced at bind time ---------------------------------------


def test_bind_rejects_wrong_task():
    spec = make_spec()
    other = load_prompt({"name": "other-task", "user": "Q: {{ticket}}"})
    with pytest.raises(PromptError, match="wrong task"):
        spec.bind(other)


def test_bind_rejects_contract_breaks_but_allows_extra_optional_fields():
    spec = make_spec()
    doc = make_seed(spec).to_dict()

    # extra OPTIONAL input field: allowed (Orizu may attach metadata fields)
    with_optional = json.loads(json.dumps(doc))
    with_optional["input"]["schema"]["properties"]["trace_tag"] = {"type": "string"}
    spec.bind(with_optional)

    # missing spec field: rejected
    broken = json.loads(json.dumps(doc))
    del broken["input"]["schema"]["properties"]["company"]
    broken["input"]["schema"]["required"] = ["ticket"]
    broken["messages"] = [m for m in broken["messages"]]
    broken["messages"][0]["template"] = "You triage tickets. Be decisive."
    broken["skills"]["refunds"]["instructions"] = "Escalate refunds."
    with pytest.raises(PromptError, match="missing spec fields: company"):
        spec.bind(broken)

    # extra REQUIRED input field: rejected (would change the call signature)
    stricter = json.loads(json.dumps(doc))
    stricter["input"]["schema"]["properties"]["tenant"] = {"type": "string"}
    stricter["input"]["schema"]["required"] = ["company", "ticket", "tenant"]
    with pytest.raises(PromptError, match="required fields"):
        spec.bind(stricter)

    # output schema drift: rejected
    drifted = json.loads(json.dumps(doc))
    drifted["output"]["schema"]["properties"]["urgency"] = {"type": "integer"}
    with pytest.raises(PromptError, match="output schema"):
        spec.bind(drifted)

    # tool the spec has a handler for is gone: rejected
    toolless = json.loads(json.dumps(doc))
    del toolless["tools"]
    with pytest.raises(PromptError, match="does not declare it"):
        spec.bind(toolless)


def test_bind_ignores_optimizable_prose():
    spec = make_spec()
    doc = make_seed(spec).to_dict()
    prose = json.loads(json.dumps(doc))
    prose["tools"]["lookup_customer"]["description"] = "Totally rewritten by Orizu."
    prose["skills"]["refunds"]["description"] = "Rewritten too."
    spec.bind(prose)  # descriptions are optimizer territory, not contract


def test_contract_comparison_is_schema_aware_and_order_insensitive():
    class Doc(BaseModel):
        description: str  # a data field literally named "description"
        urgency: str

    spec = prompt_spec(name="t", output=Doc)
    doc = spec.document(user="Q: {{q}}").to_dict()

    drifted = json.loads(json.dumps(doc))
    drifted["output"]["schema"]["properties"]["description"] = {"type": "integer"}
    with pytest.raises(PromptError, match="output schema"):
        spec.bind(drifted)

    reordered = json.loads(json.dumps(doc))
    schema = reordered["output"]["schema"]
    reordered["output"]["schema"] = {
        "additionalProperties": schema["additionalProperties"],
        "required": list(reversed(schema["required"])),
        "properties": dict(reversed(list(schema["properties"].items()))),
        "type": "object",
    }
    spec.bind(reordered)  # order is never semantic


def test_bound_prompt_survives_copy_and_is_identity_hashable():
    import copy

    bound = make_seed(make_spec())
    shallow = copy.copy(bound)
    assert shallow.content_hash() == bound.content_hash()
    assert copy.deepcopy(bound).content_hash() == bound.content_hash()
    assert len({bound, make_spec()}) == 2  # identity hash, no TypeError
    with pytest.raises(AttributeError):
        bound.no_such_attribute


def test_malformed_tool_choice_rejected_at_load():
    for bad in [{"type": "tool"}, {"type": "auto"}, {"type": "tool", "toolName": 3}]:
        with pytest.raises(Exception, match="toolName"):
            load_prompt(
                {
                    "name": "x",
                    "user": "hi",
                    "tools": {"t": {"input": {"q": "string"}}},
                    "toolChoice": bad,
                }
            )
