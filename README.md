# pai-sdk
Python model I/O with [Vercel AI SDK](https://ai-sdk.dev)-style ergonomics: the `ModelMessage` type family, `generate_text()`, and `stream_text()` — one message format and one call interface across **OpenAI** (Chat Completions _and_ Responses API), **Anthropic** (Messages API), **Google Gemini** (`google-genai`), **OpenRouter**, **Amazon Bedrock**, **Google Vertex AI**, and **Azure OpenAI**, including multimodal input and multi-step tool calling.

> **Status: alpha.** Used as the model-IO runtime under eval/optimization
> tooling; APIs may change before 1.0. Docs: [prompt-config spec](docs/prompt-config.md)
> · [embedding in a platform](docs/embedding.md) · [CHANGELOG](CHANGELOG.md).

## Why pai-sdk: two ideas

pai-sdk is built around two ideas that together turn prompt engineering from
hand-tweaking strings in code into an offline, measurable optimization loop.
Everything else in this README is mechanics for these two.

### Idea 1 — prompts are documents, so they can be optimized offline

Most prompts live as strings in code:

```python
system = f"You triage support tickets for {company}. Be decisive."
```

Three problems hide in that one line:

- **The prose and the plumbing are fused.** "Be decisive" is an instruction
  you might want to improve; `{company}` is data flow you must never break.
  As one string, nothing — human or optimizer — can safely rewrite one
  without risking the other.
- **Improving it means editing code.** Every candidate wording is a deploy.
  There is no identity for "the prompt that ran last Tuesday", no A/B, no
  rollback.
- **You can only evaluate in production.** There is no artifact to score
  offline against a dataset.

The prompt document unfuses them:

```yaml
name: support-triage
input: {company: string, ticket: string}
output: {urgency: [low, medium, high], summary: string}
system: |
  You triage support tickets for {{company}}. Be decisive.
user: "Ticket: {{ticket}}"
```

What templating like this actually buys, each point backed by a mechanism:

- **Bindings become structural.** `{{company}}` is not text; it is plumbing.
  `with_template()` rejects any rewrite that changes a template's placeholder
  set — so arbitrary rewriting of the prose is *safe by construction*, which
  is the property that makes machine rewriting possible at all.
- **Every piece of model-facing prose is addressable.** Instructions
  (`message:system`), tool descriptions (`tool:lookup_customer`), skill text
  (`skill:refunds.instructions`) — all have stable addresses. Documents never
  mark what is optimizable; each optimizer *run* chooses its targets.
- **Prompts get identity.** `content_hash()` names a candidate — pin it, A/B
  it, record it in traces, roll it back. Hashes are byte-identical in Python
  and in the TypeScript sibling ([structured-ai-sdk](https://github.com/raveeshbhalla/structured-ai-sdk)),
  because the same JSON document runs in both.
- **Improvement becomes offline search.** `read_candidate()` extracts the
  selected prose as a `{address: text}` dict — exactly the candidate shape
  GEPA's `optimize_anything` evolves. An external optimizer proposes
  rewrites, `apply_candidate()` enforces the contract, and each candidate is
  scored against your dataset — *before* any user sees it:

```python
seed = read_candidate(prompt, ["message:system", "skill:refunds.instructions"])
# ... an external optimizer (GEPA, Orizu, ...) evolves the dict offline ...
optimized = apply_candidate(prompt, best_candidate)   # variables/ids/schemas preserved
optimized.export("support-triage.optimized.json")     # ship data, not code
```

The payoff is the **adoption guarantee**: every optimizer-produced descendant
has the same variables, the same message/tool/skill ids, and the same
schemas. Call sites cannot tell the difference — except in quality. Shipping
a better prompt is a data update, not a deploy. (`prompt_spec()` makes this a
typed, load-time-validated socket; see [PromptSpec](#promptspec-typed-code-socket-for-optimizer-produced-documents).)

### Idea 2 — history is ModelMessage[], so real traffic becomes optimization data

Offline optimization is only as good as its data, and the data you want is
what the model *actually saw and did*. A semantic `inputs -> outputs` row
cannot replay a tool-using rollout; the real unit of history is the
provider-near transcript:

```python
[{"role": "system", "content": "..."},
 {"role": "user", "content": "..."},
 {"role": "assistant", "content": [..., {"type": "tool-call", "toolName": "lookup_customer", ...}]},
 {"role": "tool", "content": [{"type": "tool-result", ...}]},
 {"role": "assistant", "content": "..."}]
```

`ModelMessage[]` is that transcript as a first-class, serializable type —
wire-compatible with the TypeScript AI SDK, resendable verbatim (tool-call
ids and reasoning signatures replay correctly). Three consequences:

- **Every call is traced — tracing is plumbing, not an API.** Connect a
  sink once and every `generate_text`/`stream_text` call (including through
  `Prompt`/`PromptSpec`) emits a `Trace` as a side effect, exactly like the
  AI SDK's telemetry integrations — no separate call path, results stay
  plain, failed calls emit too (and carry `exc.trace`):

  ```python
  from pai_sdk import configure_telemetry, otel_sink
  configure_telemetry(otel_sink(my_exporter))   # once, at startup — that's it
  result = await prompt.generate({...})          # plain result; trace emitted
  ```

  Each span joins the semantic row (`inputs`/`outputs`) with the full
  transcript (`messages`). `replay_span()` reruns any span from its recorded
  input prefix — e.g. under a candidate prompt or model — and
  `span_feedback()` turns a rollout into the diagnostic text a reflective
  optimizer reads (the text-optimization analog of a gradient).
- **Provenance rides along.** Rendered messages are typed: each carries its
  `template`, bound `variables`, and message `id`, and every span records the
  document's `content_hash`. Every rollout is attributable to exactly the
  prompt version and bindings that produced it — which is what lets an
  optimizer credit outcomes to candidates.
- **Your observability backend is already the dataset.** Because history is
  plain data, it survives round trips through logging systems. Export spans
  with `trace_to_otel_spans()` (lossless `pai.*` attributes plus standard
  `gen_ai.*` mirrors), and later — when you decide to run an optimization —
  recreate fully replayable history straight from your OTEL backend:

```python
from pai_sdk.integrations.otel import trace_from_otel_spans
from pai_sdk import replay_span, span_feedback

trace = trace_from_otel_spans(spans_from_your_collector)
span = trace.spans[0]
span.messages                                   # ModelMessage[] — replayable
rerun = await replay_span(span, model=candidate_model)
feedback = span_feedback(span)                  # optimizer-readable diagnostics
```

  Foreign observability rows work too when they carry message-shaped content:
  `trace_from_braintrust_rows()` reconstructs `ModelMessage[]` from Braintrust
  exports, and generic OpenLLMetry-style `gen_ai.*` spans import with usage
  and metadata preserved. You do not need to build a data pipeline before
  your first optimization run — the traffic you already log is the training
  set. (When you want a trace in-process — an optimizer evaluator, a test —
  `generate_trace()`/`TraceCollector` return the same object connected sinks
  receive; same pipeline, not a separate one.)

### The loop, end to end

```
  prompt document (JSON) ──render──▶ ModelMessage[] ──provider──▶ rollout
         ▲                                                           │
         │ apply_candidate()                    telemetry sinks / OTEL import
         │ (contract-enforced)                                       ▼
  external optimizer ◀──── score + span_feedback() ◀────────── Trace / Span
  (GEPA optimize_anything, Orizu, ...)
```

pai-sdk deliberately does **not** ship the optimizer: GEPA, LiteLLM,
datasets, and search loops stay outside the package. pai-sdk is the runtime
on both edges of the loop — rendering documents into transcripts, and
turning transcripts back into candidates, scores, and shippable JSON
(`examples/gepa_optimize_anything.py` is a complete runner).

## AI SDK and DSPy relationship

| Capability | pai-sdk | Vercel AI SDK | DSPy |
|---|---|---|---|
| Provider-near messages | Python `ModelMessage` classes with AI-SDK-compatible JSON | Native `ModelMessage` primitive | Not the primary history representation |
| Text/stream generation | `generate_text()` / `stream_text()` with a shared provider contract | Source inspiration and close API parity | LM calls are owned by DSPy modules/adapters |
| Tool loop | Multi-step tool calling, tool results in `response.messages`, per-step traces | Similar loop and response-message continuation model | ReAct-style trajectories, but not canonical provider-message replay |
| Structured input | Prompt `input` schema plus template variables | Usually app-owned before messages are built | Signature input fields |
| Structured output | `Output.object(...)` and prompt `output` schema | Structured output helpers | Signature output fields |
| Prompt-as-data | YAML/JSON/code `Prompt` configs with stable message and tool ids | Not a prompt-config system | Signatures/modules are the main abstraction |
| Semantic history | `Trace` / `Span` store inputs, outputs, usage, metadata | App or middleware owned | Built-in examples/history center on input/output rows |
| Provider transcript history | `Trace` / `Span.messages` preserve `ModelMessage[]`, including tool calls/results | Response messages can be appended manually | Gap: not a native message-array history alongside each row |
| Replay | `replay_span()` / `replay_trace()` from stored messages; semantic replay when structured values exist | Possible manually by resending messages | Demos/history are semantic, not provider-near replay |
| Optimizer support | Target read/apply helpers for external runners; no optimizer dependency | Not an optimizer framework | Optimizer ecosystem, including GEPA |
| Observability | Versioned trace wire format, redaction, OpenTelemetry/OpenLLMetry-style conversion, Braintrust import integration | Middleware hooks | `inspect_history`, MLflow integrations |
| Program runtime | Intentionally small: runner, prompts, traces, helpers | Runtime/tooling library | Full module/program abstraction |

```bash
pip install "pai-sdk[all]"        # all providers
pip install "pai-sdk[anthropic]"  # or pick: openai / anthropic / google / bedrock / vertex

# From a checkout (not yet on PyPI):
pip install -e ".[all]"   # run from the repo root
# or pin by git tag once a remote exists:
pip install "pai-sdk[anthropic] @ git+https://github.com/raveeshbhalla/pai-sdk@v0.3.0"
```
## Quick start
```python
import asyncio
from pai_sdk import generate_text
from pai_sdk.providers import anthropic

async def main():
    result = await generate_text(
        model=anthropic("claude-opus-4-8"),
        prompt="What is the capital of France?",
    )
    print(result.text)
    print(result.usage)

asyncio.run(main())
```

API keys come from the environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENROUTER_API_KEY`.
## Choosing a model
```python
from pai_sdk.providers import (
    openai, anthropic, google, openrouter, bedrock, vertex, azure,
)

openai("gpt-5.4")                       # OpenAI Responses API (default)
openai.chat("gpt-5.4")                  # OpenAI Chat Completions API
anthropic("claude-opus-4-8")            # Anthropic Messages API
google("gemini-2.5-flash")              # Gemini
openrouter("anthropic/claude-opus-4.6") # OpenRouter (Chat Completions shape)
# Cloud-hosted variants:
bedrock("anthropic.claude-opus-4-8")    # Claude on Amazon Bedrock
vertex("gemini-2.5-flash")              # Gemini on Google Vertex AI
vertex.anthropic("claude-opus-4-8")     # Claude on Vertex AI
azure("my-deployment")                  # Azure OpenAI (Responses API)
azure.chat("my-deployment")             # Azure OpenAI (Chat Completions)
# Or plain strings, AI SDK gateway-style:
await generate_text(model="anthropic/claude-opus-4-8", prompt="...")
await generate_text(model="openrouter/google/gemini-2.5-flash", prompt="...")
await generate_text(model="bedrock/anthropic.claude-opus-4-8", prompt="...")
await generate_text(model="vertex/gemini-2.5-flash", prompt="...")
await generate_text(model="azure/my-deployment", prompt="...")
```

The cloud providers reuse the underlying Anthropic / Gemini / OpenAI request mappings — only the SDK client differs (AWS-signed, Vertex-scoped, or Azure-deployment-scoped). Credentials come from the environment:

- **Bedrock** — `AWS_REGION` plus the standard AWS credential chain (or pass `aws_region` / `aws_access_key` / `aws_secret_key` / `aws_session_token`). Model ids carry the `anthropic.` prefix and are passed through verbatim.
  
- **Vertex** — `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (default `us-central1`); Anthropic models use the `/anthropic` subpath via `vertex.anthropic(...)`.
  
- **Azure** — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION`; the model id is the Azure _deployment_ name.
  

Install the matching extras: `pai-sdk[bedrock]` or `pai-sdk[vertex]` (Azure ships with the base `openai` extra).

Configured provider instances:

```python
from pai_sdk.providers import (
    create_openai, create_openrouter, create_bedrock, create_vertex, create_azure,
)

my_openai = create_openai(api_key="sk-...", base_url="https://proxy.internal/v1")
my_openrouter = create_openrouter(app_url="https://myapp.com", app_title="My App")
my_bedrock = create_bedrock(aws_region="us-east-1")
my_vertex = create_vertex(project="my-gcp-project", location="us-east5")
my_azure = create_azure(azure_endpoint="https://my.openai.azure.com", api_version="2024-10-21")
```
## ModelMessage
The same message union as the AI SDK — `system` / `user` / `assistant` / `tool` roles, discriminated content parts. Plain dicts and typed classes are interchangeable; serialized JSON is camelCase and wire-compatible with the TypeScript AI SDK. This is the "history is data" half of [the two ideas](#why-pai-sdk-two-ideas): because transcripts serialize losslessly (`dump_messages`/`load_messages`), any store that kept them — your database, your OTEL backend, a Braintrust export — is one import away from replayable, optimizable history.

```python
from pai_sdk import (
    SystemModelMessage, UserModelMessage, AssistantModelMessage, ToolModelMessage,
    TextPart, ImagePart, FilePart, ReasoningPart, ToolCallPart, ToolResultPart,
)

messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image": open("photo.png", "rb").read()},
        ],
    },
]
result = await generate_text(model=google("gemini-2.5-flash"), messages=messages)
```

Content parts:

| Part | Where | Fields |
|---|---|---|
| `TextPart` | user, assistant | `text` |
| `ImagePart` | user | `image` (bytes / base64 / data: URL / http URL), `media_type?` |
| `FilePart` | user, assistant | `data`, `media_type` (required), `filename?` |
| `ReasoningPart` | assistant | `text` (+ provider signatures in `provider_options`) |
| `ToolCallPart` | assistant | `tool_call_id`, `tool_name`, `input` |
| `ToolResultPart` | tool | `tool_call_id`, `tool_name`, `output` (text / json / error-text / error-json / content) |

Multimodal support per provider: images everywhere; PDFs to Anthropic, OpenAI, Gemini, and OpenRouter (`FilePart` with `media_type="application/pdf"`); audio to OpenAI Chat Completions and OpenRouter (`audio/wav`, `audio/mpeg`). Remote image URLs are passed through where the provider supports them and downloaded automatically for Gemini.
## Tools and the agent loop
```python
from pydantic import BaseModel
from pai_sdk import generate_text, tool, step_count_is
from pai_sdk.providers import openai

class WeatherInput(BaseModel):
    city: str

def get_weather(input: WeatherInput) -> str:
    return f"72°F and sunny in {input.city}"

result = await generate_text(
    model=openai("gpt-5.4"),
    prompt="What's the weather in Paris and London?",
    tools={
        "get_weather": tool(
            description="Get current weather for a city",
            input_schema=WeatherInput,   # or a raw JSON Schema dict
            execute=get_weather,         # sync or async
        ),
    },
    stop_when=step_count_is(5),  # default is a single step, like the AI SDK
)
print(result.text)
print(result.steps)        # every generation step
print(result.tool_results)
```

- A Pydantic `input_schema` validates and parses the model's arguments before `execute` runs (the Zod analog). Raw JSON Schema dicts work too.
  
- Tool execution errors are caught and fed back to the model as `error-text` results, like the AI SDK.
  
- Tools without `execute` are client-side: calls are returned on `result.tool_calls` and the loop stops.
  
- `stop_when` accepts `step_count_is(n)`, `has_tool_call(name)`, a custom `def cond(steps) -> bool`, or a list of conditions.
  

Continue a conversation by appending the generated messages:

```python
history = [*messages, *result.response.messages]
```
## Streaming
```python
from pai_sdk import stream_text
from pai_sdk.providers import anthropic

result = stream_text(
    model=anthropic("claude-opus-4-8"),
    prompt="Write a haiku about types.",
)

async for delta in result.text_stream:      # plain text deltas
    print(delta, end="", flush=True)

print(await result.usage)                   # awaitable aggregates
print(await result.finish_reason)
```

`full_stream` yields the AI SDK stream-part union (`start`, `start-step`, `text-start/-delta/-end`, `reasoning-*`, `tool-input-start/-delta/-end`, `tool-call`, `tool-result`, `tool-error`, `finish-step`, `finish`, `error`):

```python
async for part in result.full_stream:
    if part.type == "text-delta":
        print(part.text, end="")
    elif part.type == "tool-call":
        print(f"\n[calling {part.tool_name}({part.input})]")
    elif part.type == "reasoning-delta":
        ...
```

Streaming runs the same multi-step tool loop, and both streams can be consumed multiple times or concurrently. Errors surface as `error` parts on `full_stream` and re-raise when awaiting aggregates. Callbacks: `on_chunk`, `on_error`, `on_step_finish`, `on_finish`.
## Provider options
Untyped passthrough for provider-specific features, keyed by provider name — same escape hatch as the AI SDK:

```python
# Anthropic adaptive thinking + effort
await generate_text(
    model=anthropic("claude-opus-4-8"),
    prompt="Solve this step by step...",
    provider_options={"anthropic": {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }},
)
# OpenAI Responses reasoning
provider_options={"openai": {"reasoning": {"effort": "high", "summary": "auto"}}}
# Gemini thinking
provider_options={"google": {"thinking_config": {"thinking_level": "high", "include_thoughts": True}}}
# OpenRouter routing / normalized reasoning
provider_options={"openrouter": {
    "reasoning": {"effort": "high"},
    "provider": {"order": ["anthropic"], "allow_fallbacks": False},
}}
```

Reasoning output comes back as `ReasoningPart`s (and `reasoning-*` stream parts). Anthropic thinking signatures and Gemini thought signatures are preserved in `provider_options` on the parts and replayed automatically when you append `result.response.messages` to history.
## Standard parameters

`generate_text` / `stream_text` accept the AI SDK parameter set:

**Input** (an optional system prompt, plus one of the two prompt forms):

- `system` — system prompt string
- `prompt` — string or message list, _or_ `messages` — message list

**Sampling & limits:**

- `max_output_tokens`
- `temperature`, `top_p`, `top_k`
- `presence_penalty`, `frequency_penalty`
- `stop_sequences`, `seed`

**Tools:**

- `tools` — `{name: tool(...)}`
- `tool_choice` — `"auto" | "none" | "required" | {"type": "tool", "tool_name": ...}`
- `active_tools` — restrict which tools are exposed for this call

**Loop & lifecycle control:**

- `stop_when` — `step_count_is(n)`, `has_tool_call(name)`, custom, or a list
- `prepare_step` — per-step model/tools/tool_choice/messages overrides
- `repair_tool_call` — one retry for invalid tool calls
- `abort_signal` — an `asyncio.Event` (or `result.abort()` on streams)
- `timeout` — total seconds or `{"total_ms", "step_ms"}`
- `output` — structured output spec (`Output.object(...)`)

**Transport & escape hatches:**

- `max_retries`, `headers`
- `provider_options` — provider-specific passthrough

**stream_text only:**

- `transform` — e.g. `smooth_stream()`
- `include_raw_chunks` — surface raw provider events as `raw` parts
- `on_chunk` / `on_error` / `on_step_finish` / `on_finish` / `on_abort` callbacks

Parameters a provider doesn't support are reported as `CallWarning`s in `result.warnings`, mirroring AI SDK behavior.
## Structured output
```python
from pydantic import BaseModel
from pai_sdk import generate_object, stream_object, Output, generate_text

class Recipe(BaseModel):
    name: str
    ingredients: list[str]

result = await generate_object(model=openai("gpt-5.4"), schema=Recipe,
                               prompt="A simple pancake recipe.")
print(result.object.name)            # validated Recipe instance
# Streaming partials:
result = stream_object(model=openai("gpt-5.4"), schema=Recipe, prompt="...")
async for partial in result.partial_object_stream:
    print(partial)                   # growing dicts via parse_partial_json
print(await result.object)
# Or combine text + structured output on generate_text:
result = await generate_text(model=..., prompt=..., output=Output.object(schema=Recipe))
print(result.output)
```

The DSPy-module-style "it just knows" dispatch lives on **prompt configs**: a `Prompt` declares its output signature, so `prompt.generate(variables)` needs no flags — it returns text when the config has no `output:`, and a schema-validated object when it does (see [Typed messages & prompt configs](#typed-messages--prompt-configs)). The standalone unified entry points `generate(...)` / `stream(...)` are for ad-hoc calls where there is no config carrying the signature — there, `schema=` *is* the signature (Python can't see the expected return type at runtime), and its presence selects the object path:

```python
from pai_sdk import generate, stream

result = await generate(model=openai("gpt-5.4"), schema=Recipe, prompt="...")
result.object                    # validated Recipe
result = await generate(model=openai("gpt-5.4"), prompt="...")
result.text                      # plain text result
```
## Agent
```python
from pai_sdk import Agent

agent = Agent(
    model=anthropic("claude-opus-4-8"),
    system="You are a research assistant.",
    tools={"search": search_tool},
    # Agents default to stop_when=step_count_is(20), like the AI SDK
)
result = await agent.generate(prompt="Find recent papers on...")
stream = agent.stream(prompt="...", temperature=0.2)  # per-call overrides win
```
## Results
`GenerateTextResult` / awaitables on `StreamTextResult`:

- `text`, `reasoning_text`, `content` (typed parts), `output` (structured output)
  
- `tool_calls`, `tool_results` (including provider-executed server-side tools — Anthropic/OpenAI web search etc. — flagged `provider_executed`)
  
- `sources` (web-search/grounding citations as `UrlSourcePart`), `files` (`GeneratedFile` with `.bytes`/`.base64`)
  
- `finish_reason` (`stop | length | content-filter | tool-calls | error | other | unknown`), `raw_finish_reason`
  
- `usage` / `total_usage` (`input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens`, `cached_input_tokens`, plus `input_token_details` / `output_token_details` cache/reasoning breakdowns)
  
- `steps` (per-step results), `response.messages` (append to history), `warnings` (structured `CallWarning`s), `request` (echoed request body), `provider_metadata` (e.g. OpenRouter `cost`, Anthropic cache tokens)
  
## Architecture
`generate_text`/`stream_text` normalize everything into `CallOptions` and a `list[ModelMessage]`, then call a `LanguageModel` implementation (`do_generate` / `do_stream`) — the same provider contract as the AI SDK's `LanguageModelV3`. Writing a new provider means implementing those two methods; the tool loop, streaming framing, retries, and message accounting live in the core.
## Middleware, registry, embeddings, traces
Why each exists:

- **Middleware** — intercept every model call without touching call sites: log/trace rollouts, enforce default settings, cache, extract `<think>` reasoning from models that inline it, or fake streaming for non-streaming backends. It's the hook an optimizer or eval harness attaches to.
  
- **Registry & aliases** — name models once (`"aliases:smart"`) and swap the implementation (different provider, wrapped with middleware, candidate under test) in one place instead of every call site.
  
- **Embeddings** — similarity for retrieval, dedup, or clustering eval data; same provider abstraction as text models.
  
- **Telemetry** — `configure_telemetry(sink, ...)` (or the scoped
  `telemetry(...)` context manager, or per-call `telemetry=`) makes every
  `generate_text`/`stream_text` call emit a `Trace` to connected sinks:
  `otel_sink(exporter)`, `jsonl_sink(path)`, `TraceCollector()`, or any
  callable. Sinks are fire-and-forget — a raising sink never breaks
  generation.

- **Trace helpers** — `generate_trace`/`stream_trace` are the in-process
  variants (they ride the telemetry pipeline and return the same trace the
  sinks receive); both join structured inputs,
  outputs, usage, metadata, and provider-near `ModelMessage[]` transcripts;
  `dump_trace`/`load_trace` round-trip versioned `pai.trace.v1` wire data for replay and
  observability imports. Generated traces also record per-step provider request
  messages after `prepare_step` overrides. `redact_trace` lets exporters scrub
  sensitive content before leaving the process.
  

```python
from pai_sdk import (
    wrap_language_model, extract_reasoning_middleware, default_settings_middleware,
    create_provider_registry, custom_provider,
    embed_many, cosine_similarity,
    dump_messages_json, load_messages,
    generate_trace, stream_trace, dump_trace_json, load_trace, replay_span, redact_trace_content,
    apply_optimizer_target, read_optimizer_target, system_instruction_target,
)
from pai_sdk.integrations.braintrust import trace_from_braintrust_rows
from pai_sdk.integrations.otel import trace_to_otel_spans
# Middleware (AI SDK LanguageModelMiddleware): logging, defaults, reasoning
# extraction, simulated streaming — or write your own transform_params /
# wrap_generate / wrap_stream.
logged = wrap_language_model(openai("gpt-5.4"), [
    default_settings_middleware({"temperature": 0.2}),
    extract_reasoning_middleware(tag_name="think"),
])
# Registry + aliases (great for swapping wrapped/candidate models):
registry = create_provider_registry({
    "openai": openai,
    "aliases": custom_provider(language_models={"smart": logged}),
})
model = registry.language_model("aliases:smart")
# Embeddings:
result = await embed_many(model=openai.embedding("text-embedding-3-small"),
                          values=["a", "b"])
# Lossless traces — ModelMessage is the log schema. Subclasses with extra
# structured fields (templates, variable bindings) round-trip intact:
text = dump_messages_json([*messages, *result.response.messages])
history = load_messages(text)   # ready to re-send

# Whole traces keep the structured history row and provider transcript together:
traced = await generate_trace(model=model, prompt="Summarize this ticket.")
trace_text = dump_trace_json(traced.trace)
trace = load_trace(trace_text)
rerun = await replay_span(trace.spans[0], model=model)
safe_trace = redact_trace_content(trace)
otel_spans = trace_to_otel_spans(safe_trace)

# Import common Braintrust SQL/export rows through the integration namespace:
trace = trace_from_braintrust_rows(braintrust_rows)

# External optimizer runners own GEPA/optimize_anything, datasets, and search.
# pai-sdk just provides safe target selection, candidate application, and traces.
target = system_instruction_target(prompt, message_id="system")
seed_candidate = read_optimizer_target(prompt, target)
candidate_prompt = apply_optimizer_target(
    prompt,
    target,
    "You triage support tickets for {{company}}. Be concise and calibrated.",
)
traced = await candidate_prompt.generate_trace({"company": "Acme", "ticket": "..."})
```
## Typed messages & prompt configs

Prompts as data: define them in YAML or JSON (in the codebase or fetched from a service) or directly in code, with Mustache-style `{{variable}}` slots and stable ids that external optimizer runners can target. All three forms produce the same `Prompt` object. Single braces are literal text, so JSON examples can appear naturally in templates. ([Why documents at all?](#idea-1--prompts-are-documents-so-they-can-be-optimized-offline) — the short version: bindings become structural, prose becomes addressable, and improvement becomes offline search you can ship without a deploy.)

### Creating a prompt in YAML

```yaml
# prompts/triage.yaml
# yaml-language-server: $schema=https://<where-you-host-it>/prompt-config.schema.json
name: support-triage
model: anthropic/claude-haiku-4-5
params:
  maxOutputTokens: 500
output:                       # field: type shorthand, compiled to JSON Schema
  urgency: [low, medium, high]   # enum
  summary: string                # string / number / integer / boolean / string[]
input:                        # optional structured input signature
  company: string
  ticket: string
system: |
  You triage support tickets for {{company}}. Be decisive.
user: "Ticket: {{ticket}}"
skills:                       # named, addressable blocks of prose
  refunds:
    description: Apply when the customer asks for money back.
    instructions: Treat refunds for {{company}} as high urgency.
```

```python
from pai_sdk import load_prompt

prompt = load_prompt("prompts/triage.yaml")
```

Skills render as system messages (id `skill:<name>`) after the last declared
system message; instruction `{{variables}}` join the input contract.

That's the simple form. The general form is an explicit `messages:` list — use it for multiple system blocks (e.g. frozen policy text next to mutable instructions), few-shot assistant turns, or stable message ids that optimizer scripts can target at run time — and `input: {schema: {...}}` / `output: {schema: {...}}` accept full JSON Schema:

```yaml
messages:
  - id: instructions
    role: system
    template: |
      You triage support tickets for {{company}}. Be decisive.
  - id: policy
    role: system
    content: "Never reveal internal data."   # literal — never touched
  - id: ticket
    role: user
    template: "Ticket: {{ticket}}"
```

YAML needs the `yaml` extra (`pip install "pai-sdk[yaml]"`).

### Creating a prompt in JSON

The same format, JSON-encoded — natural when prompts come from a service or database rather than the repo:

```json
{
  "name": "support-triage",
  "model": "anthropic/claude-haiku-4-5",
  "params": { "maxOutputTokens": 500 },
  "output": { "urgency": ["low", "medium", "high"], "summary": "string" },
  "system": "You triage support tickets for {{company}}. Be decisive.",
  "user": "Ticket: {{ticket}}"
}
```

```python
from pai_sdk import load_prompt, load_prompt_url

prompt = load_prompt("prompts/triage.json")                    # local file
prompt = await load_prompt_url("https://prompts.internal/triage")  # hosted service
```

Both YAML and JSON are validated by the packaged config schema (`pai_sdk/prompt-config.schema.json`, exported as `PROMPT_CONFIG_SCHEMA`). Point editors at it for autocomplete and red squiggles — the `# yaml-language-server: $schema=...` header above, or VS Code's `json.schemas` for JSON — and run the same schema in CI before prompts ship. It is the file a hosted prompt service should serve.

### Creating a prompt in code

`Prompt` is a Pydantic model — construct it directly (or pass a dict to `load_prompt`); `to_dict()` writes it back out as JSON/YAML-able data:

```python
from pai_sdk import Prompt, load_prompt

prompt = Prompt(
    name="support-triage",
    model="anthropic/claude-haiku-4-5",
    params={"maxOutputTokens": 500},
    output={"schema": {"type": "object", "properties": {"urgency": {"type": "string"}},
                       "required": ["urgency"], "additionalProperties": False}},
    messages=[
        {"id": "instructions", "role": "system",
         "template": "You triage support tickets for {{company}}. Be decisive."},
        {"id": "ticket", "role": "user", "template": "Ticket: {{ticket}}"},
    ],
)
prompt = load_prompt({...})        # dicts work too, simple or general form
```

Pydantic model classes work anywhere a schema does and compile to plain JSON
Schema in `to_dict()`, so the document stays portable while `result.output`
parses into your class:

```python
class Triage(BaseModel):
    urgency: Literal["low", "medium", "high"]
    summary: str

prompt = Prompt(name="triage", output=Triage, messages=[...])
result = await prompt.generate(vars)   # result.output is a Triage instance
```

Or skip configs entirely and use the typed messages straight in `generate_text` — same trace properties, no file:

```python
from pai_sdk import TypedSystemMessage, TypedUserMessage, generate_text

result = await generate_text(
    model=anthropic("claude-haiku-4-5"),
    messages=[
        TypedSystemMessage(template="You triage tickets for {{company}}.",
                           variables={"company": "Acme"}),
        TypedUserMessage(template="Ticket: {{ticket}}", variables={"ticket": "It broke"}),
    ],
)
```

### Tools in configs

Configs can declare tool interfaces (name, description, input schema — same
shorthand as `output:`); behavior binds at call time. Optimizer scripts may
rewrite a selected tool **description** (when-to-call errors are description
failures) while the name and schema stay contractual — enforced by
`with_tool_description`, like `with_template`:

```yaml
tools:
  get_weather:
    description: Get current weather. Call when asked about conditions.
    input: { city: string }
    output: { temp_f: number }   # declared result schema (typing/interface data)
maxSteps: 5
```

```python
result = await prompt.generate(vars, handlers={"get_weather": get_weather_fn})
```

In code, `tool(fn, description=...)` infers the name and input/output schemas
from the function signature (the description stays explicit — it is prompt
text):

```python
def get_weather(city: str) -> str:
    return f"72F in {city}"

tools = {"get_weather": tool(get_weather, description="Get current weather.")}
```

### Running prompts & the optimization contract

```python
result = await prompt.generate({"company": "Acme", "ticket": "It broke"})
print(result.output)                               # validated against the schema
# The optimization contract, enforced:
evolved = prompt.with_template("instructions", "You are {{company}}'s expert...")
prompt.with_template("instructions", "no vars")    # PromptError: variable set changed
evolved.content_hash()                             # candidate identity
evolved.to_dict()                                  # persist back to JSON/YAML
```

Multi-target candidates use stable addresses (`message:<id>`, `tool:<name>`,
`skill:<name>.description`, `skill:<name>.instructions`):

```python
from pai_sdk import apply_candidate, read_candidate

seed = read_candidate(prompt, ["message:instructions", "skill:refunds.instructions"])
optimized = apply_candidate(prompt, evolved_candidate)   # contract enforced
optimized.to_dict()                                      # the optimized document
```

Rendering produces `TypedSystemMessage`/`TypedUserMessage`/`TypedAssistantMessage` — subclasses that carry `template`, `variables`, and `id` alongside the rendered `content`. Providers see plain messages; `dump_messages` traces keep the structure, so logs record _which instructions_ and _which bindings_ produced every rollout. Template syntax is plain `{{name}}` only, and rendering rules are part of the versioned spec, so the same document renders identically in structured-ai-sdk. Prompt documents never carry optimization intent; optimizer scripts choose the target addresses they want to mutate for each run.
### PromptSpec: typed code socket for optimizer-produced documents

For apps on an external optimization plane (e.g. Orizu): define the types and
handlers once in code, plug each optimized JSON document in — validated at
load, typed at the call site:

```python
triage = prompt_spec(name="support-triage", input=TriageInput,
                     output=TriageVerdict, tools={"lookup_customer": lookup_fn})
seed = triage.document(model="anthropic/claude-haiku-4-5",
                       system="You triage tickets for {{company}}.",
                       user="Ticket: {{ticket}}")
seed.export("prompts/support-triage.json")         # -> optimizer ingests this
prompt = triage.load("prompts/support-triage.optimized.json")   # <- plugs back in
result = await prompt.generate(TriageInput(company="Acme", ticket="It broke"))
result.output                                       # TriageVerdict
```

## Cost estimates
Adapters normalize every provider's cache/reasoning token accounting into `usage.input_token_details` / `usage.output_token_details`, so one formula prices all of them — uncached input, cache reads, cache writes, and text+reasoning output each at their own rate:

```python
from pai_sdk import estimate_cost, register_pricing, ModelPricing

result = await generate_text(model=anthropic("claude-haiku-4-5"), ...)
cost = estimate_cost(result.total_usage, model="claude-haiku-4-5")
print(cost.total, cost.cache_read_cost, cost.output_cost)

monthly = sum(costs, start=first_cost)        # CostEstimate supports +
register_pricing("my-finetune", ModelPricing(input=2.0, output=8.0))
```

Built-in prices are dated estimates (`pricing.PRICING_AS_OF`) for common Anthropic/OpenAI/Gemini models with substring lookup (Bedrock prefixes and dated snapshots resolve). For fresh, broad coverage, pull a server-hosted table at startup — entries merge over (and override) the built-ins:

```python
await refresh_pricing()                  # LiteLLM community table (default)
await refresh_pricing("openrouter")      # OpenRouter models API (no key)
await refresh_pricing("models.dev")      # models.dev catalog
await refresh_pricing("https://prices.internal/models.json", format="simple")
```

The "simple" format for self-hosted tables is `{model_id: {"input": per_1M, "output": per_1M, "cache_read"?, "cache_write"?}}`. Or override individual models with `register_pricing` / pass `pricing=`. OpenRouter returns authoritative cost directly in `result.provider_metadata["openrouter"]["cost"]` — prefer that when available. Azure deployments have arbitrary names, so register pricing per deployment.
## Not (yet) implemented
MCP tool loading, image/speech/transcription models, and the tool-approval _flow_ (the message types exist; the loop doesn't pause on approvals yet).
