# Changelog

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
