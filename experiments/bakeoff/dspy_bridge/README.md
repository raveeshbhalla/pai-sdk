# DSPy Bridge Trace Prototype

This bakeoff artifact sketches the DSPy-bridge path for:

```python
Trace = { "id": str, "spans": list[Span] }
Span = {
    "id": str,
    "rootSpanId": str,
    "parentSpanId": str | None,
    "inputs": dict,
    "outputs": dict,
    "messages": ModelMessage[],
    "usage": dict | None,
    "metadata": dict,
}
```

The prototype uses DSPy `3.3.0b1` or newer with the normalized LM API, a fake typed LM, and a sanitized shape derived from an embedded generic review fixture. The fake LM avoids network calls and emits DSPy ChatAdapter-formatted structured fields so `dspy.Predict` still parses normal DSPy outputs.

## Files

- `dspy_bridge_trace_prototype.py`: runnable sketch.

## What It Demonstrates

- DSPy structured inputs are captured as `span.inputs`.
- DSPy parsed outputs are captured as `span.outputs`.
- Adapter-built `dspy.LMRequest.messages` are converted to pai-sdk `ModelMessage[]` and captured as `span.messages`.
- `dspy.LMResponse.usage`, `response_id`, provider data, and bridge metadata are captured on the same span.
- A second synthetic span shows the target `ToolCallPart` and `ToolResultPart` representation. This is provider-near, but not emitted by `dspy.ReAct` in the beta path.

## Current Gap

DSPy `ReAct` in `3.3.0b1` represents tool use as structured trajectory fields (`thought_0`, `tool_name_0`, `tool_args_0`, `observation_0`) instead of provider-native `LMToolCallPart` / `LMToolResultPart` messages. DSPy core types do include those parts, so the bridge mapping is straightforward, but a real ReAct bridge would need either:

- a custom adapter/module that writes tool calls into `LMRequest.messages`, or
- a postprocessor that converts ReAct trajectory fields into canonical tool-call/tool-result messages.

## Trust Verdict

Strengths:

- The trace shape is clean: DSPy gives semantic `inputs`/`outputs`, while `LMRequest` gives provider-near message history.
- A fake typed LM can validate the path without provider dependencies.
- The pai-sdk `ModelMessage` union already fits DSPy text/tool parts and serializes to stable JSON.

Gaps and risk:

- In `3.3.0b1`, `ChatAdapter._call_lm` still needs a private shim to force `forward(request: LMRequest)`.
- Adapter planning is still in motion upstream, so this should be treated as a beta integration point.
- Tool-use capture is not automatic for DSPy `ReAct`; it needs a bridge layer.

Rough code size: about 300 lines including dataclasses, conversion helpers, fake LM, and the synthetic tool span.

Optimizer/replay verdict: promising but not yet fully trustworthy on beta internals. For non-tool `Predict`/CoT-style calls, the bridge feels practical for external optimization and replay because structured inputs/outputs and request messages line up cleanly. For agent/tool replay, I would require a small explicit bridge that canonicalizes DSPy trajectories into `ModelMessage[]` before calling it production-safe.
