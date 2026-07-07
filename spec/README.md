# The pai prompt document spec (`pai.prompt.v1`)

The prompt document is the portable source of truth shared by **pai-sdk**
(Python) and **structured-ai-sdk** (TypeScript, delegating inference to the
Vercel AI SDK). It is a JSON-compatible object that carries everything
model-facing:

- a `model` reference and call `params`
- an `input` schema (the typed variable contract)
- `messages` — system/user/assistant templates with `{{variable}}` slots and
  stable ids
- an `output` schema (structured responses via provider-native strict modes)
- `tools` — name, description, input schema, output schema (behavior binds at
  call time)
- `skills` — named, addressable blocks of prose (description = when,
  instructions = how)

Code-first conveniences in either language (Pydantic models / Zod schemas /
`as const` inference, `tool(fn)`) are projections that compile INTO this
document. Nothing may exist only as code: if it cannot be serialized into the
document and re-created in the other language, it is not part of the model —
it is at most local sugar.

## Files

- The machine-readable schema is the packaged
  [`src/pai_sdk/prompt-config.schema.json`](../src/pai_sdk/prompt-config.schema.json).
  structured-ai-sdk vendors this file **byte-for-byte** as its own
  `prompt-config.schema.json`; CI in both repos should verify the copies are
  identical (compare sha256).
- [`conformance/`](./conformance) holds shared fixtures. Both runtimes run
  every fixture; a runtime conforms when all fixtures pass. Copy the directory
  (or git-subtree it) into structured-ai-sdk unchanged.

## Versioning

Documents carry `specVersion: "pai.prompt.v1"`. The field is optional on
input (assumed v1) and always emitted on output. Runtimes MUST reject a
document whose `specVersion` they do not implement. The version bumps only on
incompatible changes to the document shape **or its rendering rules** —
rendering is part of the spec, because two runtimes must produce identical
`ModelMessage[]` from the same document.

Changes from the pre-versioned format: the `optimize` boolean on messages and
tools was removed (optimizer runs select targets at run time — see
"Optimization contract"), and `skills`, `tools.<name>.output`, and
`specVersion` were added.

## Canonical serialization and content hash

`content_hash()` / `contentHash()` identifies a prompt candidate (e.g. one
point in an optimizer search). Both runtimes MUST compute:

```
hash = hex(sha256(utf8(canonical_json(document))))[0:16]
```

where `document` is the serialized form (see below) and `canonical_json` is
JSON with:

- object keys sorted lexicographically (by unicode code point), recursively
- separators `,` and `:` with no whitespace
- non-ASCII characters emitted raw (no `\uXXXX` escaping)
- numbers in ECMAScript `Number::toString` format: integral values as plain
  digits (`1.0` -> `1`, `1e21` -> `1000000000000000000000`), non-integral
  values as shortest round-trip decimal with JavaScript's exponent rules
  (`0.00001` -> `0.00001`, `1e-7` -> `1e-7`). JavaScript produces this
  natively; Python re-formats its shortest digits under the same rules.
  Integers outside ±2^53 are not interchange-safe (JSON parsers disagree on
  their value) and must not appear in documents.

Serialized form rules (what `to_dict()` emits):

- `specVersion` always present
- keys with `null`/absent values omitted
- empty `params`, `tools`, `skills` omitted
- top-level `input`/`output` shorthand is compiled to full JSON Schema at
  load and serialized in the compiled `{schema: {...}}` form; TOOL
  `input`/`output` shorthand is preserved as written
- simple-form `system:`/`user:` normalize into `messages` with ids
  `"system"` / `"user"` (the simple form is sugar; the serialized form is
  always the `messages` list)
- object key order is never semantic: hashing sorts keys, and nothing may
  render differently based on declaration order

## Template syntax

Mustache-style `{{name}}` placeholders only; names must be unicode
identifiers (XID_Start/`_` followed by XID_Continue — Python's
`str.isidentifier()`). Whitespace inside the tag is allowed (`{{ name }}`)
and is stripped using Python's whitespace set. Format specs, dotted or
indexed access, and positional tags are load-time errors. Single braces are
literal text (JSON examples need no escaping). A literal `{{` is written
`\{{`; backslashes double in front of it. Rendering requires every
placeholder bound; extra variables are ignored; values are stringified.

## Rendering rules

`render(variables)` produces the message list in document order, with each
message's rendered text as `content` and (in trace-preserving runtimes) the
`template`, bound `variables`, and `id` carried alongside.

**Skills** render as system messages, in code-point-sorted NAME order (never
declaration order — key order is not semantic and the canonical hash sorts
keys), inserted after the last declared system message (or before all
messages when there is none). The
rendered id is `skill:<name>` (message ids must not collide with these). The
rendered template composes as:

```
Skill: <name>
<description, with literal {{ escaped>

<instructions>
```

so the description is always literal prose while instructions keep their
placeholders. Skill names match `^[A-Za-z0-9][A-Za-z0-9_-]*$`.

**Input schemas**, when present, must declare every template variable
(including skill-instruction variables) as a top-level property. Runtimes
enforce required fields and top-level `additionalProperties: false` at render
time; full JSON Schema validation is the caller's choice.

## Optimization contract

Documents expose stable addressable text regions; optimizer runs (external —
e.g. GEPA `optimize_anything`; never a dependency of either runtime) choose
targets and receive/apply candidates:

| Address                       | Region                     | Contract                          |
| ----------------------------- | -------------------------- | --------------------------------- |
| `message:<id>`                | a message template         | `{{variable}}` set preserved      |
| `tool:<name>`                 | a tool description         | name + input/output schemas fixed |
| `skill:<name>.description`    | a skill's when-to-apply    | name fixed                        |
| `skill:<name>.instructions`   | a skill's how-to template  | `{{variable}}` set preserved      |

`read_candidate(prompt, addresses)` returns `{address: text}` — the
`dict[str, str]` candidate shape optimize_anything evolves.
`apply_candidate(prompt, candidate)` rebuilds a document, enforcing the
contract per address. The result of `apply_candidate(...).to_dict()` is the
optimized JSON document: persist it and load it anywhere.

Consequence — the adoption guarantee: every optimizer-produced descendant has
an identical call-site signature (same variable set, same ids, same schemas).
Consumers adopt a new version by re-fetching the document; no code changes.

## Conformance fixtures

Each `conformance/*.json` file is:

```jsonc
{
  "description": "...",
  "document": { /* a prompt document */ },
  "expect": {                    // optional document-level expectations
    "variables": ["..."],        // Prompt.variables order
    "messageIds": ["..."],       // rendered ids incl. skill:<name>
    "contentHash": "...",        // 16-hex canonical hash
    "roundTrip": true            // load(to_dict(load(document))) hash-stable
  },
  "cases": [
    {
      "variables": { /* render bindings */ },
      "messages": [ {"role": "...", "id": "...", "content": "..."} ]
    },
    { "variables": { }, "error": true }   // rendering must fail
  ],
  "invalid": [ { /* documents that must fail to load */ } ]
}
```

A conforming runtime, for every fixture: loads `document` (or rejects each
`invalid` entry), checks the `expect` fields, and for each case either renders
messages whose `(role, id, content)` triples match exactly, or errors when
`error` is true.

## Security considerations

Prompt documents are **data** and are routinely loaded from less-trusted
places (repos, databases, hosted services). The contract:

- Loading and rendering a document never executes code. Templates are plain
  substitution — no expressions, no format specs, no eval.
- Behavior binds only in code: tool handlers attach by name at call time
  (`handlers=` / runtime `tool(fn)` values). Code-only fields
  (`source_model`, `bound_execute` in pai-sdk) are rejected when a loaded
  document tries to set them — a document must never be able to smuggle in a
  schema or execution path that `to_dict()`/`content_hash()` would not
  reveal.
- Everything hash-relevant is in the serialized form: if two documents hash
  equal, they render identically (skills render in sorted-name order for
  exactly this reason).
- Runtimes must use own-property lookups for document-controlled keys (tool
  and skill names) — in JavaScript, `Object.hasOwn`, never `in` or truthiness
  on prototype-bearing objects.
- Skill names match `^[A-Za-z0-9][A-Za-z0-9_-]*$` anchored at BOTH ends
  (full match, so trailing newlines cannot forge `skill:<name>` ids).
- `load_prompt_url`/`loadPromptUrl` fetch whatever URL they are given; the
  caller owns allow-listing. Hosted services should validate uploads against
  the JSON Schema before serving them.
- Traces contain rendered prompts, model output, and (in metadata) provider
  response headers. Use `redact_trace(...)`/`redact_trace_content(...)`
  before exporting traces to external systems.
