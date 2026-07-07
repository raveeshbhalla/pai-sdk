# Changelog

## 0.5.0 — 2026-07-06

The portable prompt document. One versioned, JSON-Schema-validated document is
now the source of truth for everything model-facing; code-first conveniences
compile into it, and the TypeScript sibling (structured-ai-sdk) runs the same
documents byte-for-byte.

- `specVersion: pai.prompt.v1` on prompt documents — optional on input,
  always emitted; unknown versions rejected. Canonical serialization and the
  16-hex `content_hash()` algorithm are now specified cross-language
  (`spec/README.md`); hashes use compact separators (existing hashes change).
- **Skills**: `skills: {name: {description, instructions}}` — named,
  addressable prose blocks that render as system messages (`skill:<name>`)
  after the last declared system message. Instruction `{{variables}}` join
  the input contract. `with_skill_description()` / `with_skill_instructions()`
  mutations under the optimization contract.
- **Tool output schemas**: `tools.<name>.output` (shorthand or full JSON
  Schema) declared in documents and on `tool(..., output_schema=...)`.
- **Pydantic models as schemas**: `input=`, `output=`, and tool
  `input`/`output` accept `BaseModel` classes in code; they compile to plain
  JSON Schema on `to_dict()`, and `result.output` parses into the class.
- **`tool(fn, description=...)`**: code-first tool form — name and
  input/output schemas inferred from the function signature.
- **`render_message(id, vars)`**: render one document message (or
  `skill:<name>`) for appending typed turns to an ongoing conversation.
- **optimize_anything-shaped candidates**: target addresses (`message:<id>`,
  `tool:<name>`, `skill:<name>.description|instructions`),
  `read_candidate()` / `apply_candidate()` over `{address: text}` dicts, and
  `span_feedback()` (trace -> diagnostic ASI text). GEPA/LiteLLM remain
  external (`examples/gepa_optimize_anything.py`).
- **Conformance fixtures** in `spec/conformance/` — the cross-language
  contract structured-ai-sdk runs verbatim.
- **Removed**: the legacy `optimize:` flags on messages and tools (and
  `optimizable_messages()`/`optimizable_tools()`, the `optimize` field on
  typed messages). Optimization intent lives in optimizer scripts, never in
  documents. Documents using `optimize:` now fail validation — delete the key.
- The walked-back direction: no `Definition`/`Task`/`Signature` class DSL and
  no class-based `Tool` field markers. The document is the abstraction;
  Python sugar is Pydantic models + `tool(fn)`.
- Security/robustness hardening from adversarial review (documents are
  untrusted data): code-only fields (`source_model`, `bound_execute`) are
  rejected when set from loaded documents (a document could previously
  smuggle in a hash-invisible schema or flip a client-side tool); skill
  names are full-matched (no trailing-newline id forgery); tool `schema:`
  values must be objects (load-time error instead of call-time crash);
  skills render in sorted-name order so equal hashes imply identical
  rendering; canonical JSON numbers follow ECMAScript formatting for exact
  cross-runtime hash parity; `redact_trace_content` scrubs provider response
  headers; `tool(fn)` rejects unbound methods and positional-only params and
  passes a parsed instance to single-Pydantic-parameter functions. Spec gains
  a "Security considerations" section.

## 0.4.0 — 2026-06-11

- Tools in prompt configs: `tools:` (interface in data — description +
  input-schema shorthand; behavior binds via `prompt.generate(...,
  handlers={name: fn})`), `tool_choice:`, `max_steps:`. Declared tools
  without handlers are client-side.
- Optimization contract extended to tool descriptions:
  `with_tool_description()` / `optimizable_tools()` — descriptions are
  mutable prose when `optimize: true`; names and input schemas are
  contractual and cannot change through mutation.
- Config JSON Schema updated accordingly (loader<->schema agreement tested).

## 0.3.0 — 2026-06-11

First version intended for consumption by other codebases.

- Packaging: LICENSE (MIT), `py.typed` (PEP 561 — consumers get type checking),
  CI workflow (offline suite on Python 3.10–3.14), docs/.
- Everything below shipped during 0.1–0.2 development:
  - `ModelMessage` type family, wire-compatible with the TypeScript AI SDK.
  - `generate_text` / `stream_text` with the multi-step tool loop, stop
    conditions, `prepare_step`, `repair_tool_call`, abort/timeout, transforms.
  - Structured output: `Output`, `generate_object` / `stream_object`,
    `parse_partial_json`, partial object streaming; unified `generate`/`stream`.
  - Providers: OpenAI (Responses + Chat Completions), Anthropic, Google Gemini,
    OpenRouter, Amazon Bedrock, Google Vertex AI, Azure OpenAI — including
    reasoning round-trips (signatures / encrypted reasoning), server-side
    tools, sources, cache accounting.
  - Typed messages (`TypedSystemMessage` et al.) and prompt configs
    (`Prompt`, `load_prompt`, `load_prompt_url`) with the enforced
    optimization contract and a shipped JSON Schema for the config format.
  - Middleware (`wrap_language_model` + built-ins), provider registry,
    embeddings, lossless trace serialization (`dump_messages`/`load_messages`),
    cost estimation (`estimate_cost`, `refresh_pricing`).
  - Test suite: ~294 offline tests plus ~60 live provider tests (marker:
    `live`, keys via `.env.local`).

Status: alpha. APIs may change before 1.0. Verified on Python 3.14; 3.10–3.13
are claimed via static checks and enforced by CI once enabled.
