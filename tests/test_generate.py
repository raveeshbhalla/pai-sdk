import pytest
from pydantic import BaseModel

from model_message import generate_text, step_count_is, tool
from model_message.messages import ToolModelMessage

from conftest import FakeModel, text_step, tool_step


async def test_simple_generation():
    model = FakeModel(steps=[text_step("Paris.")])
    result = await generate_text(model=model, prompt="Capital of France?")
    assert result.text == "Paris."
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert len(result.steps) == 1
    # one assistant message generated
    assert [m.role for m in result.response.messages] == ["assistant"]


async def test_system_and_messages_reach_provider():
    model = FakeModel(steps=[text_step("ok")])
    await generate_text(
        model=model,
        system="be terse",
        messages=[{"role": "user", "content": "hi"}],
    )
    prompt = model.calls[0].prompt
    assert prompt[0].role == "system"
    assert prompt[1].role == "user"


async def test_tool_loop_executes_and_continues():
    class WeatherInput(BaseModel):
        city: str

    seen = {}

    def get_weather(input: WeatherInput) -> str:
        seen["city"] = input.city
        return f"72F in {input.city}"

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("It's 72F in Paris."),
        ]
    )
    result = await generate_text(
        model=model,
        prompt="Weather in Paris?",
        tools={
            "get_weather": tool(
                description="weather", input_schema=WeatherInput, execute=get_weather
            )
        },
        stop_when=step_count_is(5),
    )
    assert seen["city"] == "Paris"
    assert result.text == "It's 72F in Paris."
    assert len(result.steps) == 2
    assert result.steps[0].finish_reason == "tool-calls"
    assert result.steps[0].tool_results[0].output == "72F in Paris"
    # second call must include assistant tool-call message + tool result message
    second_prompt = model.calls[1].prompt
    assert [m.role for m in second_prompt] == ["user", "assistant", "tool"]
    tool_message = second_prompt[2]
    assert isinstance(tool_message, ToolModelMessage)
    assert tool_message.content[0].output.value == "72F in Paris"
    # aggregates
    assert result.total_usage.output_tokens == 13
    assert [m.role for m in result.response.messages] == ["assistant", "tool", "assistant"]


async def test_default_single_step():
    """Without stop_when, the loop stops after one step (AI SDK default)."""
    model = FakeModel(
        steps=[tool_step("get_thing"), text_step("never reached")]
    )
    result = await generate_text(
        model=model,
        prompt="go",
        tools={"get_thing": tool(execute=lambda _input: "thing")},
    )
    assert len(result.steps) == 1
    assert result.finish_reason == "tool-calls"
    assert len(model.calls) == 1


async def test_client_side_tool_stops_loop():
    model = FakeModel(steps=[tool_step("client_tool")])
    result = await generate_text(
        model=model,
        prompt="go",
        tools={"client_tool": tool(description="no execute")},
        stop_when=step_count_is(5),
    )
    assert len(result.steps) == 1
    assert result.tool_calls[0].tool_name == "client_tool"
    assert result.tool_results == []


async def test_tool_error_becomes_error_result():
    def boom(_input):
        raise RuntimeError("kaput")

    model = FakeModel(
        steps=[tool_step("boom"), text_step("recovered")]
    )
    result = await generate_text(
        model=model,
        prompt="go",
        tools={"boom": tool(execute=boom)},
        stop_when=step_count_is(3),
    )
    assert result.steps[0].tool_results[0].is_error
    assert result.steps[0].tool_results[0].model_output.type == "error-text"
    assert "kaput" in result.steps[0].tool_results[0].model_output.value
    assert result.text == "recovered"


async def test_tool_specs_sent_to_provider():
    class Input(BaseModel):
        q: str

    model = FakeModel(steps=[text_step("ok")])
    await generate_text(
        model=model,
        prompt="go",
        tools={"search": tool(description="search the web", input_schema=Input)},
        tool_choice="required",
    )
    options = model.calls[0]
    assert options.tools[0].name == "search"
    assert options.tools[0].input_schema["properties"]["q"]["type"] == "string"
    assert options.tool_choice == "required"


async def test_active_tools_filter():
    model = FakeModel(steps=[text_step("ok")])
    await generate_text(
        model=model,
        prompt="go",
        tools={"a": tool(), "b": tool()},
        active_tools=["b"],
    )
    assert [s.name for s in model.calls[0].tools] == ["b"]


async def test_on_step_finish_callback():
    collected = []
    model = FakeModel(steps=[text_step("done")])
    await generate_text(
        model=model, prompt="go", on_step_finish=lambda step: collected.append(step)
    )
    assert len(collected) == 1
    assert collected[0].text == "done"
