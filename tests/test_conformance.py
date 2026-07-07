"""Run the shared spec/conformance fixtures.

These fixtures are the cross-language contract: structured-ai-sdk (TypeScript)
runs the exact same files. A change that breaks a fixture is a spec change and
needs a deliberate decision (and possibly a specVersion bump), not a casual
edit. See spec/README.md.
"""

import json
from pathlib import Path

import pytest

from pai_sdk.prompts import load_prompt

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "spec" / "conformance"
FIXTURES = sorted(FIXTURES_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_conformance_fixture(path):
    fixture = _load(path)

    for invalid in fixture.get("invalid", []):
        with pytest.raises(Exception):
            load_prompt(json.loads(json.dumps(invalid)))

    if "document" not in fixture:
        return
    prompt = load_prompt(json.loads(json.dumps(fixture["document"])))

    expect = fixture.get("expect", {})
    if "variables" in expect:
        assert prompt.variables == expect["variables"]
    if "messageIds" in expect:
        rendered_ids = [m.id for m in prompt._effective_messages()]
        assert rendered_ids == expect["messageIds"]
    if "contentHash" in expect:
        assert prompt.content_hash() == expect["contentHash"]
    if expect.get("roundTrip"):
        reloaded = load_prompt(json.loads(json.dumps(prompt.to_dict())))
        assert reloaded.content_hash() == prompt.content_hash()

    for case in fixture.get("cases", []):
        if case.get("error"):
            with pytest.raises(Exception):
                prompt.render(case.get("variables", {}))
            continue
        rendered = prompt.render(case.get("variables", {}))
        actual = [
            {"role": m.role, "id": m.id, "content": m.content} for m in rendered
        ]
        assert actual == case["messages"]


def test_fixture_directory_is_nonempty():
    assert FIXTURES, f"no conformance fixtures found in {FIXTURES_DIR}"
