# Embedding pai-sdk in a platform

Patterns for using pai-sdk as the model-IO runtime inside a larger system —
an eval harness, a prompt-optimization platform, a batch pipeline. pai-sdk
deliberately does **not** include an optimizer, an eval loop, or a trace
store; it provides the primitives those systems are built from.

## The generic executor pattern

Because a prompt config carries its messages, output schema, model, and
params, a runner that executes "any prompt against any row" needs no
prompt-specific code:

```python
from pai_sdk import estimate_cost, load_prompt

prompt = load_prompt(config_dict)            # arrives from your control plane
result = await prompt.generate(variables)    # variables come from the row
record = {
    "output": result.output if result.output is not None else result.text,
    "token_in": result.usage.input_tokens,
    "token_out": result.usage.output_tokens,
    "cost_usd": estimate_cost(result.usage, model=model_id).total,
}
```

The only deployment-specific code is the **row → variables mapping**. Two
options, in order of preference:

1. **Flat rows by convention** — build eval datasets whose rows are variable
   dicts (`{"context": ..., "ticket": ..., "label": ...}`). Then rows *are*
   the variables and no mapping code exists.
2. **A mapper function** — `def variables(row: dict) -> dict` for raw-trace
   datasets that can't be normalized.

Error taxonomy that has worked in practice: provider/infra failures
(`APICallError`) are run-level errors; schema-validation failures
(`NoObjectGeneratedError`, carrying the raw text) are row-level results to
record, not crashes. With provider-strict structured output the latter are
rare — there is no JSON-fence parsing anywhere in this path.

## Trace recording

Attach a middleware once; every model call in the process is observed without
touching call sites:

```python
from pai_sdk import LanguageModelMiddleware, dump_messages, wrap_language_model

def trace_middleware(sink):
    async def wrap_generate(do_generate, options, model):
        result = await do_generate()
        sink({
            "model": model.model_id,
            "messages": dump_messages(options.prompt),  # lossless, camelCase JSON
            "content": [p.model_dump(by_alias=True, exclude_none=True)
                        for p in result.content],
            "usage": result.usage.__dict__,
        })
        return result
    return LanguageModelMiddleware(wrap_generate=wrap_generate)

model = wrap_language_model(base_model, [trace_middleware(my_sink)])
```

Properties your trace store inherits: dumps are valid AI-SDK (TypeScript)
messages; typed messages keep `template`/`variables`/`optimize` in the dump,
so traces are re-renderable and attributable to a prompt candidate;
`load_messages` turns a stored trace back into typed messages that can be
re-sent verbatim (tool-call ids and reasoning signatures replay correctly —
this is live-tested against all providers).

Define your record schema in **your** package; pai-sdk stays format-agnostic.

## Model management

- `create_provider_registry({...})` + `custom_provider(language_models={...})`
  give stable aliases (`"aliases:candidate-b"`) so the platform swaps
  models/wrapped variants without touching execution code.
- `"provider/model-id"` strings (`anthropic/claude-haiku-4-5`,
  `bedrock/anthropic.claude-...`, `openrouter/google/gemini-...`) resolve
  directly — store them as data.
- `wrap_language_model` composes middleware (defaults, logging, tracing) per
  alias.

## Cost accounting

`estimate_cost(usage, model=...)` prices the *normalized* token view
(`input_token_details` / `output_token_details`), so cache reads/writes and
reasoning tokens are billed correctly across providers. `CostEstimate`
supports `+` for aggregating a run. Refresh prices from a hosted table at
startup (`await refresh_pricing()` — LiteLLM/OpenRouter/models.dev or a custom
URL with the documented "simple" schema); prefer OpenRouter's authoritative
`provider_metadata["openrouter"]["cost"]` when present.

## Sync contexts

The API is async. In a synchronous entrypoint (a subprocess runner, a CLI):

```python
import asyncio
result = asyncio.run(prompt.generate(variables))
```

One event loop per process invocation is fine. Inside Jupyter (which already
runs a loop), use `await` directly.

## Concurrency

Fan out with normal asyncio; bound with a semaphore. SDK clients are cached
per model instance, so reuse one model object across a sweep:

```python
sem = asyncio.Semaphore(8)
async def run_row(row):
    async with sem:
        return await prompt.generate(mapper(row))
results = await asyncio.gather(*(run_row(r) for r in rows))
```

## Validating configs on your write path

Use the shipped schema (`from pai_sdk import PROMPT_CONFIG_SCHEMA`) with any
JSON Schema validator before persisting customer configs; `load_prompt` (the
source of truth) enforces the cross-field rules the schema can't express
(template/content exclusivity is in both; system/user-vs-messages exclusivity
is loader-only). See docs/prompt-config.md for the format.
