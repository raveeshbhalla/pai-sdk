# RFC 0001: Trace-Backed Structured History

Status: Draft

## Summary

pai-sdk should make trace-backed structured history a first-class concept.
The core object is a trace span that joins:

- structured inputs
- structured outputs
- the canonical provider-near `ModelMessage[]` history
- usage
- metadata

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
    "messages": ModelMessage[],
    "usage": {...},
    "metadata": {...},
}
```

`Trace` is the replayable unit. A trace should contain enough structured and
provider-near information to recreate the key parts of an observed agent run.
`Trace.id` is the root trace/span id. Root spans default `rootSpanId` to the
trace id; child spans keep the same `rootSpanId` and set `parentSpanId`.
Inside each span, `messages` is the canonical replay transcript: rendered
inputs followed by assistant, tool-call, tool-result, and final assistant
messages in order. When per-step hooks change what is sent to a provider,
`metadata.step_request_messages` preserves those effective provider requests
for byte-faithful auditing. Usage and metadata improve analysis, but may be
optional when importing traces from external systems.

Prompt hydration remains useful, but it is a supporting mechanism. The main
proposal is not "structured messages" by themselves. The main proposal is a
canonical history row that combines DSPy-like `inputs -> outputs` semantics with
the actual `ModelMessage[]` that produced the output.

## Decision

Use the Pai-native path as the primary direction.

pai-sdk already owns the pieces closest to the target shape:

- `ModelMessage[]` as the provider-near primitive
- typed model messages, including tool calls and tool results
- `generate_text` and `stream_text`
- tool loop execution
- structured output parsing
- prompt configs with templates, variables, tools, and input/output schemas
- `dump_messages` and `load_messages` for replayable message serialization

DSPy remains relevant as an optional interop or optimizer bridge, but it should
not become the core runtime dependency for this feature. DSPy has strong
semantic input/output abstractions, but its provider-near message history and
tool trace story are not yet the canonical product object we need.

## Context

pai-sdk is a Python model I/O runtime modeled after the Vercel AI SDK. It
normalizes provider-specific APIs behind shared message types, generation
helpers, streaming events, tool-call behavior, structured-output helpers, and
trace serialization.

The original question was whether pai-sdk should lean on DSPy, GEPA, or its own
prompt config system for structured input/output and optimization.

The sharper distinction is:

- DSPy has structured input/output semantics, but lacks canonical
  `ModelMessage[]` history alongside each semantic history row.
- pai-sdk has canonical `ModelMessage[]` and replayable provider traces, but
  needs a first-class adapter/history object that joins those messages to
  structured inputs and outputs.

The bakeoff in `experiments/bakeoff/` tested both paths against a generic
content-review judge shape. The Pai-native prototype produced the target span
with a real local tool loop:

```text
system -> user -> assistant tool-call -> tool result -> assistant final
```

all inside one `messages` array. The DSPy bridge prototype captured `Predict`
inputs and outputs cleanly, but its tool span was synthetic in DSPy `3.3.0b1`
because ReAct-style tool use is represented as trajectory fields rather than
provider-native tool messages.

## Goals

### Structured History

Developers should be able to reason about calls as structured examples:

```python
inputs = {
    "original_question": "...",
    "transcript": "...",
    "draft_title": "...",
    "draft_summary": "...",
}

outputs = {
    "issues": [...],
    "verdict": "Requires review",
}
```

This is the DSPy-like part: application authors care about input and output
fields more than provider wire formats.

### Provider-Near Replay

Every structured history row should also preserve the exact model-facing
message history:

```python
messages = [
    SystemModelMessage(...),
    UserModelMessage(...),
    AssistantModelMessage(content=[..., ToolCallPart(...)]),
    ToolModelMessage(content=[ToolResultPart(...)]),
    AssistantModelMessage(...),
]
```

This is the pai-sdk part: traces should preserve what was actually sent to and
received from model providers.

### Optimization

Optimizers should work against explicit, safe targets supplied by the optimizer
run:

- prompt message templates selected by id or path
- tool descriptions selected by name
- future hydrators or renderers selected by id or path

GEPA can live in an external runner that consumes prompts, spans, examples,
and metrics. It should not be a pai-sdk dependency or optional extra.

### Observability And Trace Import

An observed agent trace should be convertible into a pai-sdk `Trace` when it
contains enough information. In practice, importers should recreate the key
parts of the trace even when some optional fields are missing:

- provider-near messages become `Span.messages`
- parsed or inferred task data becomes `Span.inputs` and `Span.outputs`
- usage, provider ids, prompt ids, evaluator data, and other trace annotations
  become optional `usage` and `metadata`

Byte-faithful span replay should be possible from `Span.messages` alone.
Full-trace replay should use the entire `Trace`, including span relationships,
structured inputs/outputs, messages, and whatever usage or metadata was
available at import time.

## Current API

Prompt configs are already close to a lightweight signature:

- `Prompt` bundles model metadata, call params, messages, tools, and optional
  input/output schemas.
- `template` messages use `{{variable}}` placeholders.
- `Prompt.variables` is inferred from placeholders across message templates.
- `output` declares the structured response shape as shorthand or JSON Schema.
- `Prompt.render(...)` produces typed `ModelMessage` subclasses with rendered
  content and template metadata.
- `Prompt.generate(...)` runs `generate_text` with rendered messages, tools,
  params, and structured output configuration.
- `result.response.messages` contains assistant and tool messages generated
  during the call.
- `dump_messages` and `load_messages` serialize and restore message histories.

Example:

```yaml
name: content-review-judge
input:
  original_question: string
  transcript: string
  draft_title: string
  draft_summary: string
messages:
  - id: system
    role: system
    template: |
      You are an editorial reviewer. Review the draft against the source
      conversation.
  - id: review-input
    role: user
    template: |
      original_question: {{original_question}}

      transcript:
      {{transcript}}

      draft_title:
      {{draft_title}}

      draft_summary:
      {{draft_summary}}
output:
  schema:
    type: object
    properties:
      verdict:
        enum: [Good, Requires review, Bad - rewrite from scratch]
      issues:
        type: array
        items:
          type: object
          properties:
            span: {type: string}
            concern: {type: string}
            suggestion: {type: string}
          required: [span, concern, suggestion]
          additionalProperties: false
    required: [verdict, issues]
    additionalProperties: false
```

The missing official abstraction is not rendering. The missing abstraction is
the trace-backed result that captures:

```python
full_history = rendered_input_messages + result.response.messages
span = Span(inputs=variables, outputs=result.output, messages=full_history)
```

## Proposal

Add first-class trace types and helpers.

### Types

```python
@dataclass
class Trace:
    id: str
    spans: list[Span]


@dataclass
class Span:
    id: str
    root_span_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    messages: list[ModelMessage]
    parent_span_id: str | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Names can be refined, but the structure should stay close to this shape.

### Prompt Convenience API

Add a convenience API around `Prompt.generate(...)`:

```python
result = await prompt.generate_trace(inputs, handlers=handlers)

span = result.trace.spans[0]
span.inputs
span.outputs
span.messages
span.usage
span.metadata
```

The API can also be a helper rather than a method:

```python
result = await generate_trace(prompt, inputs, handlers=handlers)
```

The helper should preserve normal `GenerateTextResult` access rather than hide
it:

```python
result.text
result.output
result.response
result.trace
```

### Message Semantics

`Span.messages` should mean the full replayable conversation for the span:

```python
span.messages = [
    *rendered_prompt_messages,
    *result.response.messages,
]
```

For multi-step tool calls, this includes the assistant tool-call message, the
tool result message, and the final assistant response. This is what makes the
span useful for replay and observability.

Generated-only messages remain available from `result.response.messages`.

### Inputs And Outputs

For v1:

- `inputs` can be the exact variables passed to `Prompt.render(...)`.
- `outputs` can include `result.output` when structured output is configured.
- `outputs` should also preserve final text, finish reason, tool calls, and tool
  results when useful.

For a future version, prompt configs should likely gain an explicit `input`
schema:

```yaml
input:
  original_question: string
  transcript: string
  draft_title: string
  draft_summary: string
```

Until then, `Prompt.variables` is the lightweight signature.

### Metadata

Span metadata should preserve non-semantic context:

- prompt name, version, and content hash
- message ids and template ids
- declared tool names
- provider and model id
- response id
- step finish reasons
- source dataset or trace ids
- provider-specific metadata

Metadata should not be required for replay. Replay should primarily depend on
`messages`.

## Replay Modes

### Byte-Faithful Span Replay

Send `Span.messages` back to the model.

This mode does not require structured inputs, templates, or parsing. It is the
reason `messages` must be canonical.

### Trace Replay

Recreate a whole run from `Trace`.

Trace replay should preserve span relationships and use each span's messages,
inputs, outputs, and available metadata. When importing traces from external
agent systems, usage and metadata may be partial or absent, but the importer
should still construct a usable `Trace` when it can recover the important
pieces: span hierarchy, structured inputs/outputs, and provider-near messages.

### Semantic Replay

Re-render `Span.inputs` through a prompt config and compare or replace the
original messages.

This mode supports prompt evolution and optimization:

```python
messages = evolved_prompt.render(span.inputs)
```

Semantic replay depends on stable variables, prompt ids, and future input
schema metadata.

### Reverse Parsing

Parse existing message text back into structured inputs only when a declared
hydrator or parser supports the format.

pai-sdk should not promise arbitrary natural-language reverse parsing.

## External Optimization Runners

Optimization should happen outside the pai-sdk package. A runner built around
GEPA, `optimize_anything`, or another search loop should target prompt configs
and spans like this:

- A dataset row provides `inputs` and expected/evaluated `outputs`.
- The prompt renders `inputs` into `messages`.
- The model run produces a `Span`.
- The metric scores the span and returns feedback.
- The external optimizer proposes mutations to optimizer-selected targets.

Prompt configs should expose addressable fields, but they should not declare
which fields an optimizer may change. The optimizer script owns that choice for
each run.

For example, an optimizer run might declare:

```python
targets = [
    {"kind": "message_template", "id": "system"},
    {"kind": "tool_description", "name": "lookup_context"},
]
```

and then apply mutations through SDK helpers that preserve the structural
contract:

- message template rewrites preserve the variable set
- tool description rewrites preserve the tool name and input schema

The adoption guarantee should remain:

- optimizer mutations preserve variable sets
- optimizer mutations preserve tool names and schemas
- callers can adopt an optimized prompt without code changes

GEPA and LiteLLM should not be dependencies or extras of pai-sdk. They belong
in the optimizer runner environment. pai-sdk's job is to be the prompt/trace
runner that an optimizer can call.

## Relationship To DSPy

DSPy remains valuable, but its main value is not the runtime boundary we want
pai-sdk to own.

DSPy strengths:

- signatures
- modules such as `Predict`, Chain-of-Thought, ReAct, and RLM
- adapters
- demos and semantic history
- optimizer ecosystem, including GEPA

DSPy gap for this RFC:

- no canonical `ModelMessage[]` stored alongside each input/output history row
- provider-near tool-call/tool-result replay is not the native history shape
- normalized `LMRequest` / `LMResponse` APIs are promising but still settling in
  the tested `3.3.0b1` beta

pai-sdk gap:

- needs official `Trace` / `Span` and adapter-result semantics
- needs first-class helpers for trace creation and replay

The bakeoff suggests that adding structured-history helpers to pai-sdk is more
direct than making DSPy the core runtime and adapting its internals into
provider-near message traces.

## Current V1 Implementation

The initial implementation adds:

- `Trace` and `Span`
- top-level prompt `input` schemas using the same shorthand/full-JSON-Schema
  forms as `output`
- `StepResult.request_messages` and trace `metadata.step_request_messages` so
  per-step provider requests remain auditable when `prepare_step` overrides the
  effective message list
- `Prompt.generate_trace(...)` and `Prompt.stream_trace(...)`
- `generate_trace(...)` and `stream_trace(...)` helpers for plain prompt/message
  calls outside prompt configs
- `dump_trace(...)`, `dump_trace_json(...)`, and `load_trace(...)`
- `span_input_messages(...)`, `span_response_messages(...)`, `replay_span(...)`,
  and `replay_trace(...)`
- run-time optimizer target helpers for message templates and tool descriptions
- `read_optimizer_target(...)` and `apply_optimizer_target(...)` for external
  optimizer candidate loops
- `system_instruction_target(...)` for the common external-runner case of
  optimizing one selected system-instruction template
- failed-call trace capture by attaching `.trace` to the original exception
- `trace_from_braintrust_rows(...)` for best-effort import of Braintrust
  project-log rows into pai-sdk `Trace` / `Span` objects

Generated pai-sdk spans include `metadata.input_message_count`, which records
the boundary between rendered inputs and generated response messages. That
lets replay helpers rerun from the input prefix while the full `span.messages`
stores the canonical replay transcript. For byte-faithful step auditing,
`metadata.step_request_messages` stores the effective `ModelMessage[]` sent to
the provider for each step after `prepare_step` overrides.

An external GEPA runner can read the selected system-instruction text with
`read_optimizer_target(...)` and pass it as the `seed_candidate`, then
reconstruct each evolved prompt with `apply_optimizer_target(...)`. That keeps
variables, tools, and output schemas under pai-sdk's structural contract while
GEPA owns the search and dataset handling outside the package.

The Braintrust importer is deliberately not customer- or app-specific. It reads
common export/SQL fields (`id`, `root_span_id`, `span_attributes`, `input`,
`output`, `metadata`, `scores`, `metrics`) and reconstructs `ModelMessage[]`
only when message-shaped content is present. Otherwise, it preserves raw
input/output and Braintrust metadata for analysis.

## Hydration

Structured hydration is still useful, but it should be introduced as support
for semantic replay and trace import, not as the core RFC concept.

A future structured input layer might look like:

```python
StructuredInput(
    value={"name": "Ada", "age": 37},
    schema=PersonInput,
    hydrator="key_value_v1",
)
```

Hydrators can render structured values into message text and, when explicitly
supported, parse rendered text back into structured values.

Possible hydrators:

- JSON
- YAML
- key-value text
- Markdown sections
- custom user-defined renderers

Hydration should never replace `ModelMessage[]`. It helps produce and interpret
messages, but the trace span still stores the rendered messages.

## Non-Goals

- Do not add GEPA or LiteLLM as package dependencies or extras.
- Do not make DSPy a required runtime dependency.
- Do not replace `ModelMessage`.
- Do not guarantee reverse parsing for arbitrary natural language.
- Do not require every caller to use prompt configs.
- Do not build a full DSPy-style module/program runtime in v1.

## Open Questions

- Should child spans be emitted for each tool step, or is the flat
  `Span.messages` array sufficient for v1?
- How should external observability imports map provider-specific span trees
  into pai-sdk `Trace` and `Span` objects when input/output boundaries are
  incomplete?
