# Prompt config specification

The prompt config is pai-sdk's "prompts as data" format: a JSON-compatible
document (authored as YAML, JSON, or a Python dict) that bundles a model
reference, call parameters, optional input/output schemas, and message templates with
`{{variable}}` slots. It is designed to be stored in a repo, served by a prompt
service, and safely rewritten by external optimizer runners under an enforced
contract.

The machine-readable schema ships in the package at
`pai_sdk/prompt-config.schema.json` (exported as `PROMPT_CONFIG_SCHEMA`;
path via `pai_sdk.prompts.PROMPT_CONFIG_SCHEMA_PATH`). Validate uploads and CI
against it; point `yaml-language-server` / VS Code `json.schemas` at it for
editor support.

## Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | Identifies the prompt in logs/traces. |
| `version` | string \| int | no | Free-form version marker. |
| `description` | string | no | |
| `model` | string | no | `provider/model-id` (e.g. `anthropic/claude-haiku-4-5`). Omit to supply `model=` at call time. |
| `params` | object | no | `generate_text` kwargs applied on every call; per-call overrides win. |
| `input` | object | no | Structured input signature — shorthand or full JSON Schema (below). |
| `output` | object | no | Structured output — shorthand or full JSON Schema (below). |
| `system` / `user` | string \| object | no | Simple form (below). Mutually exclusive with `messages`. |
| `messages` | array | no | General form (below). |

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
  search_docs:
    description: Search documentation.
    input: { schema: { type: object, properties: {...}, ... } }   # full JSON Schema
tool_choice: auto         # auto | none | required | {type: tool, tool_name: ...}
max_steps: 5              # tool-loop budget -> stop_when=step_count_is(5)
```

```python
result = await prompt.generate(variables, handlers={"get_weather": get_weather_fn})
```

Handlers for undeclared tool names raise `PromptError` (catches typos).
Provider server-side tools (web search etc.) are not declared here — pass
them via `provider_options`.

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
   description while the name and input schema remain unchanged by
   construction. (When-to-call errors are description failures —
   descriptions are a first-class optimization target.)
4. **Mutations are non-destructive.** `with_template` returns a new `Prompt`;
   `content_hash()` (16-hex) identifies a candidate; `to_dict()` serializes it
   back to config form for persistence/promotion.

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
`optimize`, and `id` alongside the rendered `content`. Providers only read the
rendered content; `dump_messages` traces preserve the structure, so logs record
which instructions and which bindings produced every call.

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

`span.messages` is the byte-faithful provider transcript for that span. The
whole `Trace` is the replayable unit: imported traces may omit usage or some
metadata, but should preserve span relationships, structured inputs/outputs,
and provider-near messages when available. Semantic reruns with `replay_span`
use `metadata.input_message_count` to send only the recorded input prefix; this
boundary is recorded by pai-sdk trace helpers and can be provided by importers.

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

For example, a GEPA `optimize_anything` script can choose to optimize the
system instruction by sending only that template text as the candidate:

```python
from pai_sdk import apply_optimizer_target, system_instruction_target

target = system_instruction_target(prompt, message_id="system")
seed_candidate = next(message.template for message in prompt.messages if message.id == target.id)

async def evaluate_candidate(candidate_text, example):
    candidate_prompt = apply_optimizer_target(prompt, target, candidate_text)
    traced = await candidate_prompt.generate_trace(example["inputs"], model=model)
    return score(traced.output, example["expected"])

# The external optimizer package calls evaluate_candidate for proposals.
best_candidate = external_optimizer_result.best_candidate
optimized_prompt = apply_optimizer_target(prompt, target, best_candidate)
```

The same pattern works for a subsection of a prompt, a user-message template,
or a tool description. Prompt YAML does not need `optimize: true`; optimizer
scripts decide which stable ids to target for each run.

## Braintrust trace import

`trace_from_braintrust_rows(...)` converts Braintrust SQL/export rows into a
pai-sdk `Trace`. It understands common project-log fields like `id`,
`root_span_id`, `span_attributes`, `input`, `output`, `metadata`, `scores`, and
`metrics`. When `input` or `output` contains message-shaped data, the importer
reconstructs `ModelMessage[]`; otherwise it preserves the raw input/output and
Braintrust metadata for analysis.

```python
from pai_sdk import trace_from_braintrust_rows

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
