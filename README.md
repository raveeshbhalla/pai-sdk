# model-message

[Vercel AI SDK](https://ai-sdk.dev) ergonomics for Python: the `ModelMessage`
type family, `generate_text()`, and `stream_text()` — one message format and
one call interface across **OpenAI** (Chat Completions *and* Responses API),
**Anthropic** (Messages API), **Google Gemini** (`google-genai`),
**OpenRouter**, **Amazon Bedrock**, **Google Vertex AI**, and
**Azure OpenAI**, including multimodal input and multi-step tool calling.

```bash
pip install "model-message[all]"        # all providers
pip install "model-message[anthropic]"  # or pick: openai / anthropic / google / bedrock / vertex
```

## Quick start

```python
import asyncio
from model_message import generate_text
from model_message.providers import anthropic

async def main():
    result = await generate_text(
        model=anthropic("claude-opus-4-8"),
        prompt="What is the capital of France?",
    )
    print(result.text)
    print(result.usage)

asyncio.run(main())
```

API keys come from the environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENROUTER_API_KEY`.

## Choosing a model

```python
from model_message.providers import (
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

The cloud providers reuse the underlying Anthropic / Gemini / OpenAI request
mappings — only the SDK client differs (AWS-signed, Vertex-scoped, or
Azure-deployment-scoped). Credentials come from the environment:

- **Bedrock** — `AWS_REGION` plus the standard AWS credential chain (or pass
  `aws_region` / `aws_access_key` / `aws_secret_key` / `aws_session_token`).
  Model ids carry the `anthropic.` prefix and are passed through verbatim.
- **Vertex** — `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (default
  `us-central1`); Anthropic models use the `/anthropic` subpath via
  `vertex.anthropic(...)`.
- **Azure** — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
  `OPENAI_API_VERSION`; the model id is the Azure *deployment* name.

Install the matching extras: `model-message[bedrock]` or
`model-message[vertex]` (Azure ships with the base `openai` extra).

Configured provider instances:

```python
from model_message.providers import (
    create_openai, create_openrouter, create_bedrock, create_vertex, create_azure,
)

my_openai = create_openai(api_key="sk-...", base_url="https://proxy.internal/v1")
my_openrouter = create_openrouter(app_url="https://myapp.com", app_title="My App")
my_bedrock = create_bedrock(aws_region="us-east-1")
my_vertex = create_vertex(project="my-gcp-project", location="us-east5")
my_azure = create_azure(azure_endpoint="https://my.openai.azure.com", api_version="2024-10-21")
```

## ModelMessage

The same message union as the AI SDK — `system` / `user` / `assistant` /
`tool` roles, discriminated content parts. Plain dicts and typed classes are
interchangeable; serialized JSON is camelCase and wire-compatible with the
TypeScript AI SDK.

```python
from model_message import (
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

Multimodal support per provider: images everywhere; PDFs to Anthropic, OpenAI,
Gemini, and OpenRouter (`FilePart` with `media_type="application/pdf"`); audio
to OpenAI Chat Completions and OpenRouter (`audio/wav`, `audio/mpeg`). Remote
image URLs are passed through where the provider supports them and downloaded
automatically for Gemini.

## Tools and the agent loop

```python
from pydantic import BaseModel
from model_message import generate_text, tool, step_count_is
from model_message.providers import openai

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

- A Pydantic `input_schema` validates and parses the model's arguments before
  `execute` runs (the Zod analog). Raw JSON Schema dicts work too.
- Tool execution errors are caught and fed back to the model as `error-text`
  results, like the AI SDK.
- Tools without `execute` are client-side: calls are returned on
  `result.tool_calls` and the loop stops.
- `stop_when` accepts `step_count_is(n)`, `has_tool_call(name)`, a custom
  `def cond(steps) -> bool`, or a list of conditions.

Continue a conversation by appending the generated messages:

```python
history = [*messages, *result.response.messages]
```

## Streaming

```python
from model_message import stream_text
from model_message.providers import anthropic

result = stream_text(
    model=anthropic("claude-opus-4-8"),
    prompt="Write a haiku about types.",
)

async for delta in result.text_stream:      # plain text deltas
    print(delta, end="", flush=True)

print(await result.usage)                   # awaitable aggregates
print(await result.finish_reason)
```

`full_stream` yields the AI SDK stream-part union (`start`, `start-step`,
`text-start/-delta/-end`, `reasoning-*`, `tool-input-start/-delta/-end`,
`tool-call`, `tool-result`, `tool-error`, `finish-step`, `finish`, `error`):

```python
async for part in result.full_stream:
    if part.type == "text-delta":
        print(part.text, end="")
    elif part.type == "tool-call":
        print(f"\n[calling {part.tool_name}({part.input})]")
    elif part.type == "reasoning-delta":
        ...
```

Streaming runs the same multi-step tool loop, and both streams can be
consumed multiple times or concurrently. Errors surface as `error` parts on
`full_stream` and re-raise when awaiting aggregates. Callbacks: `on_chunk`,
`on_error`, `on_step_finish`, `on_finish`.

## Provider options

Untyped passthrough for provider-specific features, keyed by provider name —
same escape hatch as the AI SDK:

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

Reasoning output comes back as `ReasoningPart`s (and `reasoning-*` stream
parts). Anthropic thinking signatures and Gemini thought signatures are
preserved in `provider_options` on the parts and replayed automatically when
you append `result.response.messages` to history.

## Standard parameters

`generate_text` / `stream_text` accept the AI SDK parameter set:
`max_output_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`,
`frequency_penalty`, `stop_sequences`, `seed`, `tool_choice`
(`"auto" | "none" | "required" | {"type": "tool", "tool_name": ...}`),
`active_tools`, `headers`, `max_retries`, `provider_options`, `system`,
`prompt` *or* `messages`. Parameters a provider doesn't support are reported
in `result.warnings` or ignored by that provider, mirroring AI SDK behavior.

## Results

`GenerateTextResult` / awaitables on `StreamTextResult`:

- `text`, `reasoning_text`, `content` (typed parts)
- `tool_calls`, `tool_results`
- `finish_reason` (`stop | length | content-filter | tool-calls | error | other | unknown`), `raw_finish_reason`
- `usage` / `total_usage` (`input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens`, `cached_input_tokens`)
- `steps` (per-step results), `response.messages` (append to history), `warnings`

## Architecture

`generate_text`/`stream_text` normalize everything into `CallOptions` and a
`list[ModelMessage]`, then call a `LanguageModel` implementation
(`do_generate` / `do_stream`) — the same provider contract as the AI SDK's
`LanguageModelV3`. Writing a new provider means implementing those two
methods; the tool loop, streaming framing, retries, and message accounting
live in the core.

## Not (yet) implemented

Structured output helpers (`Output.object`), tool-approval flow, sources,
provider-executed tools, telemetry, and `prepare_step` are out of scope for
this first version.
