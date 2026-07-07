# Prompt document specification

The prompt document (`specVersion: pai.prompt.v1`) is pai-sdk's portable
"prompts as data" format: a JSON-compatible document (authored as YAML, JSON,
or a Python dict) that bundles a model reference, call parameters, optional
input/output schemas, message templates with `{{variable}}` slots, tool
interfaces, and skills. It is designed to be stored in a repo, served by a
prompt service, safely rewritten by external optimizer runners under an
enforced contract, and run identically by the TypeScript sibling
(structured-ai-sdk). The cross-language rules — canonical serialization,
content hashing, rendering, conformance fixtures — live in
[spec/README.md](../spec/README.md).

The machine-readable schema ships in the package at
`pai_sdk/prompt-config.schema.json` (exported as `PROMPT_CONFIG_SCHEMA`;
path via `pai_sdk.prompts.PROMPT_CONFIG_SCHEMA_PATH`). Validate uploads and CI
against it; point `yaml-language-server` / VS Code `json.schemas` at it for
editor support.

## Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `specVersion` | `"pai.prompt.v1"` | no | Assumed v1 when absent; always emitted by `to_dict()`. Unknown versions are rejected. |
| `name` | string | yes | Identifies the prompt in logs/traces. |
| `version` | string \| int | no | Free-form version marker. |
| `description` | string | no | |
| `model` | string | no | `provider/model-id` (e.g. `anthropic/claude-haiku-4-5`). Omit to supply `model=` at call time. |
| `params` | object | no | Call options in the **AI SDK vocabulary** (camelCase: `maxOutputTokens`, `temperature`, `topP`, `providerOptions`, ...). TypeScript passes them to `generateText`/`streamText` verbatim; Python maps them 1:1 onto `generate_text` kwargs. Per-call overrides win. Snake_case keys are rejected with a did-you-mean error. |
| `input` | object | no | Structured input signature — shorthand or full JSON Schema (below). |
| `output` | object | no | Structured output — shorthand or full JSON Schema (below). |
| `system` / `user` | string \| object | no | Simple form (below). Mutually exclusive with `messages`. |
| `toolChoice` | string \| object | no | `auto` \| `none` \| `required` \| `{type: tool, toolName}`. |
| `maxSteps` | integer | no | Tool-loop budget. |
| `messages` | array | no | General form (below). |
| `skills` | object | no | Named blocks of model-facing prose (below). |

A config must yield at least one message (simple or general form).

## Simple form

```yaml
system: |
  You triage support tickets for {{company}}. Be decisive.
user: "Ticket: {{ticket}}"
```

`system`/`user` accept a string template or `{template|content, id}` for
control. They normalize to a `messages` list with ids `"system"` / `"user"`.

## General form

```yaml
messages:
  - id: instructions       # stable id — addressing for mutations
    role: system           # system | user | assistant
    template: |            # interpolated; placeholders are the contract
      You triage support tickets for {{company}}.
  - id: policy
    role: system
    content: "Never reveal internal data."   # literal — no interpolation, braces untouched
  - id: ticket
    role: user
    template: "Ticket: {{ticket}}"
```

Each message has exactly one of `template` (interpolated) or `content`
(literal). `assistant` role exists for few-shot demonstrations. Message ids
must be unique.

## Template syntax

Mustache-style `{{name}}` placeholders only — names must be Python
identifiers. Optional whitespace is allowed inside the tag, e.g. `{{ name }}`.
Format specs (`{{x:>10}}`), conversions (`{{x!r}}`), positional (`{{0}}`,
`{{}}`), and dotted/indexed access (`{{a.b}}`, `{{a[0]}}`) are rejected at
load time. Single braces are literal text, so JSON examples like
`{"answer": "yes"}` can appear in templates without escaping. This
restriction is deliberate: the same templates must render identically in
non-Python runtimes (a TypeScript implementation is a small regex
interpolator).

Rendering requires every placeholder to be bound; extra variables are ignored.
Values are stringified. (Known limitation: slots are text-only — there is no
message-level slot for splicing structured content yet.)

## Input and output schemas

`input` and `output` support the same two forms. **Shorthand** (no `schema`
key) is a field-type mapping compiled to a strict JSON Schema (all fields
required, `additionalProperties: false`):

```yaml
input:
  company: string
  ticket: string
  customer_context: string

output:
  urgency: [low, medium, high]   # list of literals -> enum
  summary: string                # string | number | integer | boolean
  tags: string[]                 # "<type>[]" -> array (nests: string[][])
  reporter:                      # nested mapping -> nested object
    name: string
    id: integer                  # null/empty value -> string
```

**Full JSON Schema** (escape hatch — presence of `schema` selects it):

```yaml
input:
  schema:
    type: object
    properties:
      company: {type: string}
      ticket: {type: string}
      customer_context: {type: string}
    required: [company, ticket]   # customer_context is optional
    additionalProperties: false
output:
  schema: { type: object, properties: {...}, required: [...], additionalProperties: false }
  name: triage          # optional
  description: ...      # optional
```

When `input` is present, every template variable must be declared as a
top-level input property. `Prompt.render()` enforces missing required fields and
`additionalProperties: false` at the top level. It intentionally does not do
full JSON Schema type validation; callers can run their validator of choice
before invoking the prompt. Optional fields are useful for trace/eval metadata
or future hydrators, but a `{{placeholder}}` still requires a value when the
template is rendered.

When `output` is present, `prompt.generate()` requests provider-strict
structured output and returns the validated object on `result.output`.

In code, a Pydantic model class works anywhere a schema does — `input`,
`output`, and tool `input`/`output` — and compiles to plain JSON Schema on
serialization, so the document stays portable while `result.output` parses
into the model class:

```python
class Triage(BaseModel):
    urgency: Literal["low", "medium", "high"]
    summary: str

prompt = Prompt(name="triage", output=Triage, messages=[...])
result = await prompt.generate(vars)      # result.output is a Triage
prompt.to_dict()["output"]["schema"]      # plain JSON Schema
```

## Tools

Tool **interfaces** are config; tool **behavior** is code. The config declares
name (the key), description, and input schema; `execute` functions bind by
name at call time. Declared tools without a handler are client-side — calls
come back on `result.tool_calls`.

```yaml
tools:
  get_weather:
    description: Get current weather. Call when asked about conditions.
    input:                # same field:type shorthand as output:
      city: string
    output:               # declared result schema (interface/typing data;
      temp_f: number      # not enforced against handler returns at run time)
      conditions: string
  search_docs:
    description: Search documentation.
    input: { schema: { type: object, properties: {...}, ... } }   # full JSON Schema
toolChoice: auto          # auto | none | required | {type: tool, toolName: ...}
maxSteps: 5               # tool-loop budget -> stop_when=step_count_is(5)
```

```python
result = await prompt.generate(variables, handlers={"get_weather": get_weather_fn})
```

In code, a runtime `Tool` (e.g. `tool(fn, description=...)`) can be placed
directly in `Prompt(tools={...})`: its interface compiles into the document
and its `execute` auto-binds as the default handler (call-time `handlers=`
win). Handlers for undeclared tool names raise `PromptError` (catches typos).
Provider server-side tools (web search etc.) are not declared here — pass
them via `provider_options`. Provider caveat: Gemini currently rejects tools
combined with JSON `output` in one call — split the tool loop and the
structured extraction into two prompts there.

## Skills

A skill is a named, addressable block of model-facing prose: `description`
says when it applies, `instructions` (a template) says how. Skills render as
system messages with id `skill:<name>` after the last declared system
message, in code-point-sorted name order (key order is never semantic);
instruction `{{variables}}` join the prompt's
input contract:

```yaml
skills:
  escalation:
    description: When a ticket mentions legal threats or refunds over $500.
    instructions: |
      Escalate to a human. Summarize the thread for {{company}} first.
```

The name is the contract; both prose fields are optimizer-addressable
(`skill:<name>.description`, `skill:<name>.instructions`).
`with_skill_description()` swaps the prose;`with_skill_instructions()`
enforces the same variable-set contract as `with_template()`.
`prompt.render_message("skill:escalation", vars)` renders one skill block for
appending to an ongoing conversation.

## The optimization contract

These rules are **enforced by the library**, not advisory — they are what make
optimizer-produced versions safe to adopt automatically:

1. **Variables are structurally untouchable.** Placeholders are bindings, not
   text. `Prompt.with_template(message_id, new_template)` rejects any mutation
   whose placeholder set differs from the original.
2. **Optimizer targets live in the optimizer script.** Prompt configs expose
   stable message ids and tool names; each optimizer run decides which ids it
   is allowed to rewrite.
3. **Tool descriptions are addressable prose; names and schemas are the
   contract.** `with_tool_description(name, text)` rewrites a tool's
   description while the name and input/output schemas remain unchanged by
   construction. (When-to-call errors are description failures —
   descriptions are a first-class optimization target.) Skills follow the
   same split: `with_skill_description()` is free prose,
   `with_skill_instructions()` preserves the variable set, and the skill name
   never changes.
4. **Mutations are non-destructive.** `with_template` returns a new `Prompt`;
   `content_hash()` (16-hex sha256 of the canonical document JSON — the
   algorithm is specified in spec/README.md so Python and TypeScript agree)
   identifies a candidate; `to_dict()` serializes it back to config form for
   persistence/promotion.

Consequence — the **adoption guarantee**: every optimizer-produced descendant
of a prompt has an identical call-site signature (same variable set, same
message ids). Consumers can adopt a new version by re-fetching the config;
no code change is ever required by an optimizer mutation. Only a *human*
edit that changes the variable set is a breaking change.

## Loading and running

```python
from pai_sdk import load_prompt, load_prompt_url, Prompt

prompt = load_prompt("prompts/triage.yaml")     # .yaml/.yml (yaml extra) or .json
prompt = load_prompt({...})                     # dict, simple or general form
prompt = await load_prompt_url(url)             # hosted service (format inferred)
prompt = Prompt(name=..., messages=[...])       # plain Pydantic constructor

prompt.variables             # ordered template variable names (the signature)
prompt.input_schema()        # declared structured input signature, if present
messages = prompt.render({"company": "Acme", "ticket": "..."})  # typed messages
result = await prompt.generate({...}, model=optional_override, **overrides)
stream = prompt.stream({...})
traced = await prompt.generate_trace({...}, model=optional_override)
traced_stream = prompt.stream_trace({...}, model=optional_override)
```

`render()` produces `TypedSystemMessage` / `TypedUserMessage` /
`TypedAssistantMessage` — message subclasses carrying `template`, `variables`,
and `id` alongside the rendered `content`. Providers only read the rendered
content; `dump_messages` traces preserve the structure, so logs record which
instructions and which bindings produced every call. New optimizer runs should
choose target ids in the optimizer script rather than encoding optimization
intent in the prompt config.

## Trace-backed generation

`Prompt.generate_trace(...)` returns a generation result wrapper with the normal
`GenerateTextResult` fields plus a replayable `Trace`. `Prompt.stream_trace(...)`
does the same for `stream_text`; its trace is awaitable after the stream
finishes:

```python
from pai_sdk import dump_trace_json, load_trace, replay_span

traced = await prompt.generate_trace({"company": "Acme", "ticket": "..."})
streamed = prompt.stream_trace({"company": "Acme", "ticket": "..."})

traced.text
traced.output
stream_trace = await streamed.trace

span = traced.trace.spans[0]
span.inputs      # variables passed to render()
span.outputs     # text/object/finish/tool summaries
span.messages    # rendered input messages + assistant/tool/final messages
span.usage       # total token usage when available
span.metadata    # prompt/model/response metadata

loaded = load_trace(dump_trace_json(traced.trace))
rerun = await replay_span(loaded.spans[0], model=alternate_model)
```

`span.messages` is the canonical replay transcript for that span: rendered
input messages followed by assistant/tool/final response messages. The whole
`Trace` is the replayable unit: imported traces may omit usage or some
metadata, but should preserve span relationships, structured inputs/outputs,
and provider-near messages when available. Semantic reruns with `replay_span`
use `metadata.input_message_count` to send only the recorded input prefix; this
boundary is recorded by pai-sdk trace helpers and can be provided by importers.
For byte-faithful step auditing, pai-sdk generated traces also include
`metadata.step_request_messages`, the effective `ModelMessage[]` sent to the
provider for each step after `prepare_step` overrides.

If generation fails after messages have been rendered, `generate_trace(...)`
and `stream_trace(...)` attach a failed `Trace` to the original exception as
`.trace`. The span includes the rendered input messages, `outputs.error`, and
`metadata.error` so failed calls remain observable and replayable from the same
input prefix.

## External optimizer runners

pai-sdk does not ship or depend on an optimizer. External runners own GEPA,
`optimize_anything`, datasets, candidate search, and sync/async orchestration.
pai-sdk provides the pieces those runners need: stable target ids,
contract-preserving candidate application, structured output, and replayable
traces.

Targets are addressed as `message:<id>`, `tool:<name>`,
`skill:<name>.description`, or `skill:<name>.instructions`. A candidate is a
`{address: text}` dict — exactly the shape GEPA's `optimize_anything`
evolves — and a span converts into the diagnostic feedback (ASI) its
reflective proposer reads:

```python
from pai_sdk import apply_candidate, read_candidate, span_feedback

targets = ["message:system", "skill:refunds.instructions", "tool:lookup"]
seed_candidate = read_candidate(prompt, targets)      # {address: current text}

def evaluate(candidate, example):                      # GEPA evaluator
    evolved = apply_candidate(prompt, candidate)       # contract enforced
    traced = await evolved.generate_trace(example["inputs"], model=model)
    score = metric(traced.output, example["expected"])
    return score, span_feedback(traced.trace.spans[0]) # score + trace ASI

# after the run, the winner is a plain JSON document — persist and adopt it:
optimized = apply_candidate(prompt, result.best_candidate)
Path("triage.optimized.json").write_text(json.dumps(optimized.to_dict()))
```

`examples/gepa_optimize_anything.py` is a complete runner in generalization
mode (dataset + valset). The single-target helpers
(`system_instruction_target`, `read_optimizer_target`,
`apply_optimizer_target`) remain for one-target runs; they accept both
`OptimizerTarget` values and address strings. Prompt YAML never marks
anything optimizable; optimizer scripts decide which stable ids to target for
each run.

## Trace integrations

pai-sdk's core trace format is `pai.trace.v1`. `dump_trace(...)` and
`dump_trace_json(...)` include `schemaVersion`, and `TRACE_WIRE_SCHEMA` exposes
a JSON Schema for validation. Use `redact_trace(...)` or
`redact_trace_content(...)` before sending traces to external systems.

OpenTelemetry/OpenLLMetry-style conversion is dependency-free and lives in the
integration namespace:

```python
from pai_sdk import redact_trace_content
from pai_sdk.integrations.otel import trace_from_otel_spans, trace_to_otel_spans

safe_trace = redact_trace_content(trace)
otel_spans = trace_to_otel_spans(safe_trace)
trace = trace_from_otel_spans(otel_spans)
```

Braintrust is a vendor-specific integration, not part of the top-level SDK API.
`trace_from_braintrust_rows(...)` converts Braintrust SQL/export rows into a
pai-sdk `Trace`. It understands common project-log fields like `id`,
`root_span_id`, `span_attributes`, `input`, `output`, `metadata`, `scores`, and
`metrics`. When `input` or `output` contains message-shaped data, the importer
reconstructs `ModelMessage[]`; otherwise it preserves the raw input/output and
Braintrust metadata for analysis.

```python
from pai_sdk.integrations.braintrust import trace_from_braintrust_rows

trace = trace_from_braintrust_rows(rows)
span = trace.spans[0]

span.messages
span.usage
span.metadata["braintrust"]["scores"]
```

This is intentionally best-effort. It is enough to turn an observed Braintrust
run into pai-sdk's structured trace shape when the Braintrust row carries the
rendered messages, and it still preserves useful metadata when privacy settings
or application logging omit message content.
