# Trace-backed history bakeoff

Status: exploratory

## Question

Which path can most cleanly produce a canonical trace span that joins structured
inputs, structured outputs, and the provider-near `ModelMessage[]` history?

```python
Trace = {
    "id": "...",
    "spans": [Span],
}

Span = {
    "id": "...",
    "rootSpanId": "...",
    "parentSpanId": None,
    "inputs": {...},
    "outputs": {...},
    "messages": [...],  # ModelMessage[]
    "usage": {...},
    "metadata": {...},
}
```

## Fixture

Use an embedded generic content-review judge shape. The fixture is intentionally
small and synthetic so this bakeoff can be shared safely.

Representative inputs:

- `original_question`
- `transcript`
- `draft_title`
- `draft_summary`

Representative outputs:

- `issues`: list of `{span, concern, suggestion}`
- `verdict`: `"Good" | "Requires review" | "Bad - rewrite from scratch"`

The original exploratory run used a private local fixture, but these checked-in
artifacts should not reference private names or paths.

## Candidates

### Pai-native

Prototype a thin adapter around existing `pai-sdk` prompt configs:

- `Prompt.variables` / future input schema -> `Span.inputs`
- `Prompt.output` -> `Span.outputs`
- rendered typed messages and response messages -> `Span.messages`
- result usage and prompt metadata -> `Span.usage` / `Span.metadata`

### DSPy bridge

Prototype a wrapper around DSPy calls:

- DSPy signature inputs -> `Span.inputs`
- DSPy prediction fields -> `Span.outputs`
- adapter-built `LMRequest.messages` converted to `ModelMessage[]` -> `Span.messages`
- LM usage and signature/module metadata -> `Span.usage` / `Span.metadata`

## Scoring

Score each candidate on:

- Trace fidelity: byte-faithful replay from `messages`.
- Semantic fidelity: clean typed `inputs` and `outputs`.
- Tool fidelity: tool call and tool result messages remain in `messages`.
- API trust: relies on stable public APIs vs internals.
- GEPA readiness: clear optimizer-selected targets and structural contracts.
- External trace import: can agent traces become spans/history?
- Code size and conceptual cleanliness.
- Failure modes: where hydration, parsing, replay, or optimization can break.
