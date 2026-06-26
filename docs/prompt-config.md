# Prompt config specification

The prompt config is pai-sdk's "prompts as data" format: a JSON-compatible
document (authored as YAML, JSON, or a Python dict) that bundles a model
reference, call parameters, an output schema, and message templates with
`{{variable}}` slots. It is designed to be stored in a repo, served by a prompt
service, and **mutated by reflective optimizers under an enforced contract**.

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
| `output` | object | no | Structured output — shorthand or full JSON Schema (below). |
| `system` / `user` | string \| object | no | Simple form (below). Mutually exclusive with `messages`. |
| `messages` | array | no | General form (below). |

A config must yield at least one message (simple or general form).

## Simple form

```yaml
system: |                  # optimize: true by DEFAULT — this IS the instructions
  You triage support tickets for {{company}}. Be decisive.
user: "Ticket: {{ticket}}" # optimize: false — never rewritten
```

`system`/`user` accept a string template or `{template|content, optimize, id}`
for control. They normalize to a `messages` list with ids `"system"` / `"user"`.

## General form

```yaml
messages:
  - id: instructions       # stable id — addressing for mutations
    role: system           # system | user | assistant
    optimize: true         # an optimizer MAY rewrite this text
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

## Output schemas

Two forms. **Shorthand** (no `schema` key) — field-type mapping compiled to a
strict JSON Schema (all fields required, `additionalProperties: false`):

```yaml
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
output:
  schema: { type: object, properties: {...}, required: [...], additionalProperties: false }
  name: triage          # optional
  description: ...      # optional
```

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
    optimize: true        # an optimizer MAY rewrite the DESCRIPTION
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
2. **Only `optimize: true` messages may be rewritten.** Mutating any other
   message raises `PromptError`.
3. **Tool descriptions are optimizable prose; names and schemas are the
   contract.** `with_tool_description(name, text)` rewrites a tool's
   description only when that tool is `optimize: true`; the name and input
   schema cannot change through mutation by construction. (When-to-call
   errors are description failures — descriptions are a first-class
   optimization target.) `optimizable_tools()` lists the eligible tools.
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
messages = prompt.render({"company": "Acme", "ticket": "..."})  # typed messages
result = await prompt.generate({...}, model=optional_override, **overrides)
stream = prompt.stream({...})
```

`render()` produces `TypedSystemMessage` / `TypedUserMessage` /
`TypedAssistantMessage` — message subclasses carrying `template`, `variables`,
`optimize`, and `id` alongside the rendered `content`. Providers only read the
rendered content; `dump_messages` traces preserve the structure, so logs record
which instructions and which bindings produced every call.
