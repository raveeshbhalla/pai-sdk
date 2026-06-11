from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from model_message import (
    CallOptions,
    ProviderResult,
    ResponseMetadata,
    TextPart,
    ToolCallPart,
    Usage,
    generate_text,
    step_count_is,
    stream_text,
    tool,
)
from model_message.errors import AbortError, GenerationTimeoutError
from model_message.generate import PrepareStepResult
from model_message.stream import ProviderStreamPart, TextDelta, TextEnd, TextStart
from model_message.transforms import smooth_stream

from conftest import FakeModel, text_step, tool_step


# ---------------------------------------------------------------------------
# prepare_step
# ---------------------------------------------------------------------------


async def test_prepare_step_receives_args():
    seen = []
    model = FakeModel(steps=[tool_step("t"), text_step("done")])

    def prepare(*, model, step_number, steps, messages):
        seen.append((step_number, len(steps), len(messages)))
        return None

    await generate_text(
        model=model,
        prompt="go",
        tools={"t": tool(execute=lambda i: "ok")},
        stop_when=step_count_is(5),
        prepare_step=prepare,
    )
    # called before each step (including the first)
    assert seen[0][0] == 0 and seen[0][1] == 0
    assert seen[1][0] == 1 and seen[1][1] == 1
    # messages grow between steps
    assert seen[1][2] > seen[0][2]


async def test_prepare_step_overrides_reach_call_options():
    model = FakeModel(steps=[text_step("ok")])

    def prepare(**kwargs):
        return PrepareStepResult(tool_choice="required", active_tools=["b"])

    await generate_text(
        model=model,
        prompt="go",
        tools={"a": tool(), "b": tool()},
        prepare_step=prepare,
    )
    opts = model.calls[0]
    assert opts.tool_choice == "required"
    assert [s.name for s in opts.tools] == ["b"]


async def test_prepare_step_model_override():
    other = FakeModel(steps=[text_step("from other")])
    base = FakeModel(steps=[text_step("from base")])

    def prepare(**kwargs):
        return PrepareStepResult(model=other)

    result = await generate_text(model=base, prompt="go", prepare_step=prepare)
    assert result.text == "from other"
    assert len(other.calls) == 1
    assert len(base.calls) == 0


async def test_prepare_step_messages_override_is_per_step():
    model = FakeModel(steps=[tool_step("t"), text_step("done")])

    override_msgs = [{"role": "user", "content": "OVERRIDE"}]

    def prepare(*, step_number, **kwargs):
        if step_number == 0:
            return PrepareStepResult(messages=override_msgs)
        return None

    await generate_text(
        model=model,
        prompt="original",
        tools={"t": tool(execute=lambda i: "ok")},
        stop_when=step_count_is(5),
        prepare_step=prepare,
    )
    # step 0 saw only the overridden message
    first = model.calls[0].prompt
    assert len(first) == 1 and first[0].content == "OVERRIDE"
    # step 1 resumes from canonical history ("original" user msg) + appended
    second = model.calls[1].prompt
    assert second[0].content == "original"
    assert [m.role for m in second] == ["user", "assistant", "tool"]


async def test_prepare_step_async():
    model = FakeModel(steps=[text_step("ok")])

    async def prepare(**kwargs):
        await asyncio.sleep(0)
        return PrepareStepResult(tool_choice="none")

    await generate_text(model=model, prompt="go", tools={"a": tool()}, prepare_step=prepare)
    assert model.calls[0].tool_choice == "none"


# ---------------------------------------------------------------------------
# repair_tool_call
# ---------------------------------------------------------------------------


class _Schema(BaseModel):
    city: str


async def test_repair_tool_call_fixes_bad_input():
    executed = {}

    def get_weather(input: _Schema) -> str:
        executed["city"] = input.city
        return "ok"

    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"wrong_key": 1}),  # fails schema
            text_step("done"),
        ]
    )

    def repair(*, tool_call, tools, error, messages):
        return ToolCallPart(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            input={"city": "Paris"},
        )

    result = await generate_text(
        model=model,
        prompt="go",
        tools={"get_weather": tool(input_schema=_Schema, execute=get_weather)},
        stop_when=step_count_is(5),
        repair_tool_call=repair,
    )
    assert executed["city"] == "Paris"
    assert not result.steps[0].tool_results[0].is_error
    # history shows the fixed call
    fixed = [p for p in result.steps[0].content if isinstance(p, ToolCallPart)][0]
    assert fixed.input == {"city": "Paris"}


async def test_repair_tool_call_none_falls_back_to_error():
    model = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"wrong_key": 1}),
            text_step("done"),
        ]
    )

    def repair(**kwargs):
        return None

    result = await generate_text(
        model=model,
        prompt="go",
        tools={"get_weather": tool(input_schema=_Schema, execute=lambda i: "ok")},
        stop_when=step_count_is(5),
        repair_tool_call=repair,
    )
    assert result.steps[0].tool_results[0].is_error
    assert result.steps[0].tool_results[0].model_output.type == "error-text"


async def test_repair_tool_call_unknown_tool_becomes_error_when_set():
    model = FakeModel(steps=[tool_step("missing"), text_step("done")])

    def repair(**kwargs):
        return None  # cannot repair

    result = await generate_text(
        model=model,
        prompt="go",
        tools={"real": tool(execute=lambda i: "ok")},
        stop_when=step_count_is(5),
        repair_tool_call=repair,
    )
    # unknown tool with repair set -> NoSuchTool error result (not client-side)
    assert result.steps[0].tool_results[0].is_error
    assert "missing" in result.steps[0].tool_results[0].model_output.value


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


async def test_generate_abort_preset_event_raises():
    model = FakeModel(steps=[text_step("never")])
    event = asyncio.Event()
    event.set()
    with pytest.raises(AbortError):
        await generate_text(model=model, prompt="go", abort_signal=event)
    assert len(model.calls) == 0


async def test_stream_abort_preset_emits_abort_part_and_callback():
    model = FakeModel(steps=[text_step("never")])
    event = asyncio.Event()
    event.set()
    fired = {"abort": False}

    result = stream_text(
        model=model,
        prompt="go",
        abort_signal=event,
        on_abort=lambda r: fired.__setitem__("abort", True),
    )
    types = [p.type async for p in result.full_stream]
    assert "abort" in types
    assert "finish" not in types
    assert fired["abort"]
    with pytest.raises(AbortError):
        await result.text


async def test_stream_abort_method_stops_multi_step_run():
    @dataclass
    class GatedModel(FakeModel):
        gate: asyncio.Event = field(default_factory=asyncio.Event)
        first_done: asyncio.Event = field(default_factory=asyncio.Event)

        async def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
            n = len(self.calls)
            async for part in super().do_stream(options):
                yield part
            if n == 0:
                self.first_done.set()
                await self.gate.wait()

    model = GatedModel(steps=[tool_step("t"), text_step("second")])
    result = stream_text(
        model=model,
        prompt="go",
        tools={"t": tool(execute=lambda i: "ok")},
        stop_when=step_count_is(5),
    )

    async def consume():
        return [p.type async for p in result.full_stream]

    task = asyncio.create_task(consume())
    await model.first_done.wait()
    result.abort("user stop")
    model.gate.set()
    types = await task
    assert "abort" in types
    with pytest.raises(AbortError):
        await result.all_steps


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------


@dataclass
class SlowModel(FakeModel):
    sleep: float = 0.0

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        await asyncio.sleep(self.sleep)
        return await super().do_generate(options)


async def test_step_timeout_fires():
    model = SlowModel(steps=[text_step("ok")], sleep=0.2)
    with pytest.raises(GenerationTimeoutError) as exc:
        await generate_text(
            model=model, prompt="go", timeout={"step_ms": 20}
        )
    assert exc.value.budget == "step"


async def test_total_timeout_caps_multi_step_run():
    model = SlowModel(
        steps=[tool_step("t"), tool_step("t"), text_step("done")],
        sleep=0.05,
    )
    with pytest.raises(GenerationTimeoutError) as exc:
        await generate_text(
            model=model,
            prompt="go",
            tools={"t": tool(execute=lambda i: "ok")},
            stop_when=step_count_is(10),
            timeout={"total_ms": 80},
        )
    assert exc.value.budget == "total"


async def test_float_timeout_is_total():
    model = SlowModel(steps=[text_step("ok")], sleep=0.2)
    with pytest.raises(GenerationTimeoutError):
        await generate_text(model=model, prompt="go", timeout=0.02)


# ---------------------------------------------------------------------------
# stream transforms / smooth_stream
# ---------------------------------------------------------------------------


async def _drive(stream):
    parts = []
    async for p in stream:
        parts.append(p)
    return parts


async def test_smooth_stream_rechunks_into_words():
    src_parts = [
        TextStart(id="0"),
        TextDelta(id="0", text="Hello wor"),
        TextDelta(id="0", text="ld foo"),
        TextEnd(id="0"),
    ]

    async def src():
        for p in src_parts:
            yield p

    tf = smooth_stream(delay_in_ms=None, chunking="word")
    out = await _drive(tf(src()))
    deltas = [p.text for p in out if isinstance(p, TextDelta)]
    assert deltas == ["Hello ", "world ", "foo"]


async def test_smooth_stream_passes_tool_parts_through():
    call = ToolCallPart(tool_call_id="c1", tool_name="t", input={})
    src_parts = [
        TextStart(id="0"),
        TextDelta(id="0", text="hi there"),
        call,
        TextEnd(id="0"),
    ]

    async def src():
        for p in src_parts:
            yield p

    tf = smooth_stream(delay_in_ms=None)
    out = await _drive(tf(src()))
    assert call in out
    # text flushed by TextEnd
    assert "".join(p.text for p in out if isinstance(p, TextDelta)) == "hi there"


async def test_transform_list_composition():
    def upper(stream):
        async def gen():
            async for p in stream:
                if isinstance(p, TextDelta):
                    yield TextDelta(id=p.id, text=p.text.upper())
                else:
                    yield p

        return gen()

    model = FakeModel(steps=[text_step("hello world")])
    result = stream_text(
        model=model,
        prompt="go",
        transform=[smooth_stream(delay_in_ms=None), upper],
    )
    text = "".join([c async for c in result.text_stream])
    assert text == "HELLO WORLD"
    # aggregate (from raw parts) stays untransformed
    assert await result.text == "hello world"


async def test_smooth_stream_in_stream_text():
    model = FakeModel(steps=[text_step("Hello world foo")])
    result = stream_text(
        model=model, prompt="go", transform=smooth_stream(delay_in_ms=None)
    )
    chunks = [c async for c in result.text_stream]
    assert "".join(chunks) == "Hello world foo"
