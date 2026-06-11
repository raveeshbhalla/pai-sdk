"""Tests for the Agent (ToolLoopAgent) abstraction in model_message.agent."""

import pytest
from pydantic import BaseModel

from model_message import step_count_is, tool
from model_message.agent import Agent

from conftest import FakeModel, text_step, tool_step


# ---------------------------------------------------------------------------
# 1. Defaults flow through to the provider CallOptions
# ---------------------------------------------------------------------------


async def test_defaults_flow_through_to_call_options():
    """system, temperature, and tools set on Agent must reach the provider."""

    class Input(BaseModel):
        q: str

    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(
        model=model,
        system="be concise",
        temperature=0.3,
        tools={"search": tool(description="web search", input_schema=Input)},
    )
    await agent.generate(prompt="hi")

    options = model.calls[0]
    # system message should be first in the prompt
    assert options.prompt[0].role == "system"
    assert options.prompt[0].content == "be concise"
    # temperature forwarded
    assert options.temperature == 0.3
    # tool spec forwarded
    assert len(options.tools) == 1
    assert options.tools[0].name == "search"


# ---------------------------------------------------------------------------
# 2. Default stop_when enables multi-step (no explicit stop_when needed)
# ---------------------------------------------------------------------------


async def test_default_stop_when_allows_multi_step():
    """Agent default stop_when=step_count_is(20): tool step + text step runs 2
    steps without the caller passing stop_when."""

    class CityInput(BaseModel):
        city: str

    seen: dict = {}

    def get_weather(input: CityInput) -> str:
        seen["city"] = input.city
        return f"72F in {input.city}"

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("It's 72F in Paris."),
        ]
    )
    agent = Agent(
        model=model,
        tools={
            "get_weather": tool(
                description="weather",
                input_schema=CityInput,
                execute=get_weather,
            )
        },
        # No stop_when — should default to step_count_is(20)
    )
    result = await agent.generate(prompt="Weather in Paris?")

    assert seen.get("city") == "Paris"
    assert len(result.steps) == 2
    assert result.steps[0].finish_reason == "tool-calls"
    assert result.text == "It's 72F in Paris."


# ---------------------------------------------------------------------------
# 3. Per-call overrides win over constructor defaults
# ---------------------------------------------------------------------------


async def test_per_call_temperature_overrides_default():
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model, temperature=0.5)

    await agent.generate(prompt="hi", temperature=0.9)

    assert model.calls[0].temperature == 0.9


async def test_per_call_system_overrides_default():
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model, system="default system")

    await agent.generate(
        messages=[{"role": "user", "content": "hi"}],
        system="override system",
    )

    assert model.calls[0].prompt[0].content == "override system"


async def test_per_call_tools_replaces_default():
    """Per-call tools dict replaces default entirely (no merging)."""

    class Input(BaseModel):
        x: int

    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(
        model=model,
        tools={"tool_a": tool(description="default tool")},
    )
    await agent.generate(
        prompt="hi",
        tools={"tool_b": tool(description="override tool", input_schema=Input)},
    )

    names = [s.name for s in model.calls[0].tools]
    assert names == ["tool_b"]
    assert "tool_a" not in names


# ---------------------------------------------------------------------------
# 4. provider_options shallow-merge by provider key, per-call wins
# ---------------------------------------------------------------------------


async def test_provider_options_merge():
    """Base and per-call provider_options merge by provider key; per-call wins."""
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(
        model=model,
        provider_options={
            "anthropic": {"thinking": {"type": "basic"}},
            "openai": {"reasoning": {"effort": "low"}},
        },
    )
    await agent.generate(
        prompt="hi",
        provider_options={"anthropic": {"thinking": {"type": "adaptive"}}},
    )

    po = model.calls[0].provider_options
    # per-call anthropic key wins
    assert po["anthropic"]["thinking"]["type"] == "adaptive"
    # base openai key is preserved
    assert po["openai"]["reasoning"]["effort"] == "low"


async def test_provider_options_empty_base_uses_override():
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model)
    await agent.generate(
        prompt="hi",
        provider_options={"google": {"thinking_config": {"thinking_level": "high"}}},
    )

    po = model.calls[0].provider_options
    assert po["google"]["thinking_config"]["thinking_level"] == "high"


# ---------------------------------------------------------------------------
# 5. stream() works and respects defaults
# ---------------------------------------------------------------------------


async def test_stream_respects_defaults():
    """stream() delegates to stream_text with the same merged options."""
    model = FakeModel(steps=[text_step("Streamed response.")])
    agent = Agent(model=model, system="streaming system", temperature=0.1)

    result = agent.stream(prompt="stream me")
    text = await result.text

    assert text == "Streamed response."
    assert model.calls[0].temperature == 0.1
    assert model.calls[0].prompt[0].role == "system"
    assert model.calls[0].prompt[0].content == "streaming system"


async def test_stream_multi_step_default():
    """stream() also defaults to multi-step (step_count_is(20))."""

    def noop_tool(_input) -> str:
        return "result"

    model = FakeModel(
        steps=[
            tool_step("noop", call_id="c1"),
            text_step("done"),
        ]
    )
    agent = Agent(
        model=model,
        tools={"noop": tool(execute=noop_tool)},
    )
    result = agent.stream(prompt="go")
    steps = await result.all_steps
    assert len(steps) == 2
    assert await result.text == "done"


async def test_stream_per_call_override():
    """Per-call override is honoured in stream() as well."""
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model, temperature=0.2)

    result = agent.stream(prompt="hi", temperature=0.8)
    await result.consume_stream()

    assert model.calls[0].temperature == 0.8


# ---------------------------------------------------------------------------
# 6. Unknown override key raises TypeError
# ---------------------------------------------------------------------------


async def test_unknown_override_key_raises_type_error():
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model)

    with pytest.raises(TypeError, match="unknown override key"):
        await agent.generate(prompt="hi", totally_invalid_kwarg=42)


async def test_unknown_override_key_in_stream_raises_type_error():
    model = FakeModel(steps=[text_step("ok")])
    agent = Agent(model=model)

    with pytest.raises(TypeError, match="unknown override key"):
        agent.stream(prompt="hi", another_bad_kwarg="oops")


# ---------------------------------------------------------------------------
# 7. on_step_finish callback is invoked
# ---------------------------------------------------------------------------


async def test_on_step_finish_default_callback():
    collected = []
    model = FakeModel(steps=[text_step("done")])
    agent = Agent(model=model, on_step_finish=lambda step: collected.append(step))

    await agent.generate(prompt="go")

    assert len(collected) == 1
    assert collected[0].text == "done"


async def test_on_step_finish_per_call_override():
    default_collected = []
    override_collected = []

    model = FakeModel(steps=[text_step("done")])
    agent = Agent(
        model=model,
        on_step_finish=lambda step: default_collected.append(step),
    )
    await agent.generate(
        prompt="go",
        on_step_finish=lambda step: override_collected.append(step),
    )

    # Per-call override wins; default callback should NOT be called.
    assert len(override_collected) == 1
    assert len(default_collected) == 0
