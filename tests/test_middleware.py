from __future__ import annotations

import dataclasses
import logging

from pai_sdk import (
    CallOptions,
    Finish,
    ProviderResult,
    ReasoningPart,
    ResponseMetadata,
    TextDelta,
    TextEnd,
    TextPart,
    TextStart,
    ToolCallPart,
    Usage,
    generate_text,
    step_count_is,
    stream_text,
    tool,
)
from pai_sdk.middleware import (
    LanguageModelMiddleware,
    default_settings_middleware,
    extract_reasoning_middleware,
    logging_middleware,
    simulate_streaming_middleware,
    wrap_language_model,
)
from pai_sdk.stream import (
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
)

from conftest import FakeModel, text_step, tool_step


# ---------------------------------------------------------------------------
# Helpers: a model that records the options it received
# ---------------------------------------------------------------------------


def _opts(prompt="hi"):
    from pai_sdk import UserModelMessage

    return CallOptions(prompt=[UserModelMessage(content=prompt)])


async def _collect(model, options):
    return [p async for p in model.do_stream(options)]


# ---------------------------------------------------------------------------
# transform_params ordering
# ---------------------------------------------------------------------------


async def test_transform_params_order_first_runs_first():
    order: list[str] = []

    def make(name, temp):
        def transform_params(options, *, type, model):
            order.append(name)
            return dataclasses.replace(options, temperature=temp)

        return LanguageModelMiddleware(transform_params=transform_params)

    base = FakeModel(steps=[text_step("ok")])
    wrapped = wrap_language_model(base, [make("a", 0.1), make("b", 0.9)])
    await wrapped.do_generate(_opts())

    # First middleware ("a") runs first, second ("b") runs last and wins.
    assert order == ["a", "b"]
    assert base.calls[0].temperature == 0.9


async def test_transform_params_async_supported():
    async def transform_params(options, *, type, model):
        return dataclasses.replace(options, top_p=0.5)

    base = FakeModel(steps=[text_step("ok")])
    wrapped = wrap_language_model(
        base, LanguageModelMiddleware(transform_params=transform_params)
    )
    await wrapped.do_generate(_opts())
    assert base.calls[0].top_p == 0.5


async def test_transform_params_receives_type():
    seen: list[str] = []

    def transform_params(options, *, type, model):
        seen.append(type)
        return options

    base = FakeModel(steps=[text_step("ok"), text_step("ok")])
    wrapped = wrap_language_model(
        base, LanguageModelMiddleware(transform_params=transform_params)
    )
    await wrapped.do_generate(_opts())
    await _collect(wrapped, _opts())
    assert seen == ["generate", "stream"]


# ---------------------------------------------------------------------------
# wrap_generate composition (outermost first)
# ---------------------------------------------------------------------------


async def test_wrap_generate_composition_outermost_first():
    order: list[str] = []

    def make(name):
        async def wrap_generate(do_generate, options, model):
            order.append(f"{name}:before")
            result = await do_generate()
            order.append(f"{name}:after")
            return result

        return LanguageModelMiddleware(wrap_generate=wrap_generate)

    base = FakeModel(steps=[text_step("ok")])
    wrapped = wrap_language_model(base, [make("outer"), make("inner")])
    await wrapped.do_generate(_opts())

    assert order == [
        "outer:before",
        "inner:before",
        "inner:after",
        "outer:after",
    ]


async def test_transform_params_runs_before_wrap_generate():
    order: list[str] = []

    def transform_params(options, *, type, model):
        order.append("transform")
        return options

    async def wrap_generate(do_generate, options, model):
        order.append("wrap")
        return await do_generate()

    base = FakeModel(steps=[text_step("ok")])
    mw = LanguageModelMiddleware(
        transform_params=transform_params, wrap_generate=wrap_generate
    )
    wrapped = wrap_language_model(base, mw)
    await wrapped.do_generate(_opts())
    assert order == ["transform", "wrap"]


async def test_wrap_generate_sees_transformed_options_all_mws_first():
    """transform_params for ALL middlewares runs before any wrap_* body."""
    order: list[str] = []

    def make_transform(name):
        def transform_params(options, *, type, model):
            order.append(f"transform:{name}")
            return options

        async def wrap_generate(do_generate, options, model):
            order.append(f"wrap:{name}")
            return await do_generate()

        return LanguageModelMiddleware(
            transform_params=transform_params, wrap_generate=wrap_generate
        )

    base = FakeModel(steps=[text_step("ok")])
    wrapped = wrap_language_model(base, [make_transform("a"), make_transform("b")])
    await wrapped.do_generate(_opts())
    assert order == ["transform:a", "transform:b", "wrap:a", "wrap:b"]


# ---------------------------------------------------------------------------
# default_settings_middleware
# ---------------------------------------------------------------------------


async def test_default_settings_fills_only_none_fields():
    base = FakeModel(steps=[text_step("ok")])
    mw = default_settings_middleware(
        {"temperature": 0.7, "max_output_tokens": 100}
    )
    wrapped = wrap_language_model(base, mw)

    opts = _opts()
    opts.temperature = 0.2  # explicit caller value should win
    await wrapped.do_generate(opts)

    assert base.calls[0].temperature == 0.2  # caller value preserved
    assert base.calls[0].max_output_tokens == 100  # default filled


async def test_default_settings_provider_options_merge():
    base = FakeModel(steps=[text_step("ok")])
    mw = default_settings_middleware(
        {
            "provider_options": {
                "anthropic": {"thinking": {"type": "adaptive"}, "effort": "low"},
                "openai": {"reasoning": "x"},
            }
        }
    )
    wrapped = wrap_language_model(base, mw)

    opts = _opts()
    opts.provider_options = {"anthropic": {"effort": "high"}}
    await wrapped.do_generate(opts)

    po = base.calls[0].provider_options
    # existing wins on conflict; defaults fill the rest; other providers added
    assert po["anthropic"]["effort"] == "high"
    assert po["anthropic"]["thinking"] == {"type": "adaptive"}
    assert po["openai"] == {"reasoning": "x"}


# ---------------------------------------------------------------------------
# extract_reasoning_middleware — generate
# ---------------------------------------------------------------------------


async def test_extract_reasoning_generate_mixed():
    result = ProviderResult(
        content=[
            TextPart(text="before<think>secret thoughts</think>after")
        ],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        response=ResponseMetadata(id="r", model_id="fake-1"),
    )

    base = FakeModel(steps=[result])
    wrapped = wrap_language_model(base, extract_reasoning_middleware())
    out = await wrapped.do_generate(_opts())

    kinds = [(type(p).__name__, getattr(p, "text", None)) for p in out.content]
    assert ("ReasoningPart", "secret thoughts") in kinds
    texts = [p.text for p in out.content if isinstance(p, TextPart)]
    assert "before" in texts[0] and "after" in texts[0]


async def test_extract_reasoning_generate_start_with_reasoning():
    result = ProviderResult(
        content=[TextPart(text="thinking here</think>visible")],
        finish_reason="stop",
        usage=Usage(),
        response=ResponseMetadata(),
    )
    base = FakeModel(steps=[result])
    wrapped = wrap_language_model(
        base, extract_reasoning_middleware(start_with_reasoning=True)
    )
    out = await wrapped.do_generate(_opts())
    reasoning = [p for p in out.content if isinstance(p, ReasoningPart)]
    text = [p for p in out.content if isinstance(p, TextPart)]
    assert reasoning[0].text == "thinking here"
    assert text[0].text == "visible"


# ---------------------------------------------------------------------------
# extract_reasoning_middleware — stream
# ---------------------------------------------------------------------------


class _ScriptedStream(FakeModel):
    """A model whose do_stream yields a fixed list of text deltas."""

    def __init__(self, deltas):
        super().__init__(steps=[text_step("ignored")])
        self._deltas = deltas

    async def do_stream(self, options):
        self.calls.append(options)
        yield TextStart(id="0")
        for d in self._deltas:
            yield TextDelta(id="0", text=d)
        yield TextEnd(id="0")
        yield Finish(finish_reason="stop", total_usage=Usage())


async def test_extract_reasoning_stream_tag_split_across_deltas():
    # "<think>" split across two deltas; "ought" continues the reasoning.
    model = _ScriptedStream(
        deltas=["abc<thi", "nk>th", "ought</think>xyz"]
    )
    wrapped = wrap_language_model(model, extract_reasoning_middleware())
    parts = await _collect(wrapped, _opts())

    reasoning_text = "".join(
        p.text for p in parts if isinstance(p, ReasoningDelta)
    )
    text = "".join(p.text for p in parts if isinstance(p, TextDelta))
    assert reasoning_text == "thought"
    assert text == "abcxyz"
    # reasoning block was opened and closed
    assert any(isinstance(p, ReasoningStart) for p in parts)
    assert any(isinstance(p, ReasoningEnd) for p in parts)


async def test_extract_reasoning_stream_partial_tag_suffix_held_back():
    # A trailing "<" that looks like a partial tag must be held back, then
    # flushed as plain text at stream end (it was never a real tag).
    model = _ScriptedStream(deltas=["hello <"])
    wrapped = wrap_language_model(model, extract_reasoning_middleware())
    parts = await _collect(wrapped, _opts())

    text = "".join(p.text for p in parts if isinstance(p, TextDelta))
    assert text == "hello <"
    assert not any(isinstance(p, ReasoningDelta) for p in parts)


async def test_extract_reasoning_stream_start_with_reasoning():
    model = _ScriptedStream(deltas=["deep thoughts</think>answer"])
    wrapped = wrap_language_model(
        model, extract_reasoning_middleware(start_with_reasoning=True)
    )
    parts = await _collect(wrapped, _opts())
    reasoning_text = "".join(
        p.text for p in parts if isinstance(p, ReasoningDelta)
    )
    text = "".join(p.text for p in parts if isinstance(p, TextDelta))
    assert reasoning_text == "deep thoughts"
    assert text == "answer"


# ---------------------------------------------------------------------------
# simulate_streaming_middleware
# ---------------------------------------------------------------------------


async def test_simulate_streaming_end_to_end():
    base = FakeModel(steps=[text_step("Hello world")])
    wrapped = wrap_language_model(base, simulate_streaming_middleware())

    result = stream_text(model=wrapped, prompt="hi")
    chunks = [c async for c in result.text_stream]
    assert "".join(chunks) == "Hello world"
    assert await result.text == "Hello world"
    assert (await result.usage).output_tokens == 5
    assert await result.finish_reason == "stop"
    # simulate_streaming used the non-streaming path: do_generate was called.
    assert len(base.calls) == 1


async def test_simulate_streaming_with_reasoning_and_tool():
    pr = ProviderResult(
        content=[
            ReasoningPart(text="thinking"),
            TextPart(text="answer"),
            ToolCallPart(tool_call_id="c1", tool_name="t", input={}),
        ],
        finish_reason="tool-calls",
        usage=Usage(output_tokens=3),
        response=ResponseMetadata(id="r1", model_id="fake-1"),
    )
    base = FakeModel(steps=[pr])
    wrapped = wrap_language_model(base, simulate_streaming_middleware())
    parts = await _collect(wrapped, _opts())
    types = [p.type for p in parts]
    assert types[0] == "response-metadata"
    assert "reasoning-start" in types and "reasoning-delta" in types
    assert "text-start" in types
    assert "tool-input-start" in types and "tool-call" in types
    assert types[-1] == "finish"


# ---------------------------------------------------------------------------
# nesting, overrides
# ---------------------------------------------------------------------------


async def test_wrap_on_wrapped_model_nesting():
    order: list[str] = []

    def make(name):
        async def wrap_generate(do_generate, options, model):
            order.append(f"{name}:before")
            r = await do_generate()
            order.append(f"{name}:after")
            return r

        return LanguageModelMiddleware(wrap_generate=wrap_generate)

    base = FakeModel(steps=[text_step("ok")])
    inner = wrap_language_model(base, make("inner"))
    outer = wrap_language_model(inner, make("outer"))
    await outer.do_generate(_opts())
    assert order == [
        "outer:before",
        "inner:before",
        "inner:after",
        "outer:after",
    ]


async def test_model_id_and_provider_overrides():
    base = FakeModel(steps=[text_step("ok")])
    plain = wrap_language_model(base, LanguageModelMiddleware())
    assert plain.model_id == "fake-1"
    assert plain.provider == "fake"

    overridden = wrap_language_model(
        base,
        LanguageModelMiddleware(),
        model_id="custom-model",
        provider_id="custom-provider",
    )
    assert overridden.model_id == "custom-model"
    assert overridden.provider == "custom-provider"


async def test_duck_typed_middleware_object():
    class MyMiddleware:
        def __init__(self):
            self.seen = False

        async def wrap_generate(self, do_generate, options, model):
            self.seen = True
            return await do_generate()

    mw = MyMiddleware()
    base = FakeModel(steps=[text_step("ok")])
    wrapped = wrap_language_model(base, mw)
    await wrapped.do_generate(_opts())
    assert mw.seen is True


# ---------------------------------------------------------------------------
# logging middleware smoke
# ---------------------------------------------------------------------------


async def test_logging_middleware_generate(caplog):
    base = FakeModel(steps=[text_step("ok")])
    logger = logging.getLogger("test.mw.logging")
    wrapped = wrap_language_model(
        base, logging_middleware(logger=logger, level=logging.INFO)
    )
    with caplog.at_level(logging.INFO, logger="test.mw.logging"):
        await wrapped.do_generate(_opts())
    messages = [r.getMessage() for r in caplog.records]
    assert any("generate request" in m for m in messages)
    assert any("generate response" in m and "stop" in m for m in messages)


async def test_logging_middleware_stream(caplog):
    base = FakeModel(steps=[text_step("Hello")])
    logger = logging.getLogger("test.mw.logging.stream")
    wrapped = wrap_language_model(
        base, logging_middleware(logger=logger, level=logging.INFO)
    )
    with caplog.at_level(logging.INFO, logger="test.mw.logging.stream"):
        await _collect(wrapped, _opts())
    messages = [r.getMessage() for r in caplog.records]
    assert any("stream request" in m for m in messages)
    assert any("stream response" in m and "parts=" in m for m in messages)


# ---------------------------------------------------------------------------
# end-to-end: generate_text over a wrapped model still does the tool loop
# ---------------------------------------------------------------------------


async def test_generate_text_tool_loop_over_wrapped_model():
    base = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("72F in Paris."),
        ]
    )
    wrapped = wrap_language_model(base, logging_middleware())

    result = await generate_text(
        model=wrapped,
        prompt="weather?",
        tools={"get_weather": tool(execute=lambda i: "72F")},
        stop_when=step_count_is(5),
    )
    assert result.text == "72F in Paris."
    assert len(result.steps) == 2
    assert result.steps[0].tool_calls[0].tool_name == "get_weather"


async def test_stream_text_tool_loop_over_wrapped_model():
    base = FakeModel(
        steps=[
            tool_step("get_weather", tool_input={"city": "Paris"}),
            text_step("72F in Paris."),
        ]
    )
    wrapped = wrap_language_model(base, extract_reasoning_middleware())

    result = stream_text(
        model=wrapped,
        prompt="weather?",
        tools={"get_weather": tool(execute=lambda i: "72F")},
        stop_when=step_count_is(5),
    )
    text = "".join([c async for c in result.text_stream])
    assert "72F in Paris." in text
    assert len(result.steps) == 2
