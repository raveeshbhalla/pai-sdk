import asyncio

from model_message import step_count_is, stream_text, tool

from conftest import FakeModel, text_step, tool_step


async def test_text_stream():
    model = FakeModel(steps=[text_step("Hello world")])
    result = stream_text(model=model, prompt="hi")
    chunks = [c async for c in result.text_stream]
    assert "".join(chunks) == "Hello world"
    assert len(chunks) >= 2  # FakeModel splits into multiple deltas
    assert await result.text == "Hello world"
    assert (await result.usage).output_tokens == 5
    assert await result.finish_reason == "stop"


async def test_full_stream_framing():
    model = FakeModel(steps=[text_step("Hi")])
    result = stream_text(model=model, prompt="hi")
    types = [part.type async for part in result.full_stream]
    assert types[0] == "start"
    assert types[1] == "start-step"
    assert "text-start" in types and "text-end" in types
    assert types[-2] == "finish-step"
    assert types[-1] == "finish"


async def test_stream_tool_loop():
    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("72F in Paris."),
        ]
    )
    result = stream_text(
        model=model,
        prompt="weather?",
        tools={"get_weather": tool(execute=lambda i: "72F")},
        stop_when=step_count_is(5),
    )
    types = [part.type async for part in result.full_stream]
    assert types.count("start-step") == 2
    assert types.count("finish-step") == 2
    assert "tool-input-start" in types
    assert "tool-call" in types
    assert "tool-result" in types
    assert await result.text == "72F in Paris."
    steps = await result.all_steps
    assert len(steps) == 2
    assert (await result.total_usage).output_tokens == 13
    messages = (await result.response).messages
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]


async def test_multiple_concurrent_consumers():
    model = FakeModel(steps=[text_step("Hello world")])
    result = stream_text(model=model, prompt="hi")

    async def consume_text():
        return "".join([c async for c in result.text_stream])

    async def get_usage():
        return await result.usage

    text_a, text_b, usage = await asyncio.gather(
        consume_text(), consume_text(), get_usage()
    )
    assert text_a == text_b == "Hello world"
    assert usage.output_tokens == 5


async def test_stream_error_surfaces():
    class ExplodingModel(FakeModel):
        async def do_stream(self, options):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    result = stream_text(model=ExplodingModel(), prompt="hi")
    parts = [part async for part in result.full_stream]
    assert parts[-1].type == "error"
    try:
        await result.text
        raised = False
    except RuntimeError:
        raised = True
    assert raised


async def test_on_chunk_and_on_finish_callbacks():
    seen = {"chunks": 0, "finished": False}
    model = FakeModel(steps=[text_step("Hi")])
    result = stream_text(
        model=model,
        prompt="hi",
        on_chunk=lambda part: seen.__setitem__("chunks", seen["chunks"] + 1),
        on_finish=lambda res: seen.__setitem__("finished", True),
    )
    await result.consume_stream()
    assert seen["chunks"] > 0
    assert seen["finished"]
