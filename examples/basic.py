"""Examples — set one provider API key and run.

    OPENAI_API_KEY=... python examples/basic.py

If present, ~/.config/structured-ai-sdk/.env.local is loaded without
overriding existing environment variables.
"""

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from pai_sdk import generate_text, step_count_is, stream_text, tool
from pai_sdk.providers import anthropic, google, openai, openrouter  # noqa: F401


def _load_env_local() -> None:
    env_file = Path("~/.config/structured-ai-sdk/.env.local").expanduser()
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _default_model():
    if os.environ.get("OPENAI_API_KEY"):
        return openai("gpt-5.4-mini")              # Responses API
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic("claude-opus-4-8")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return google("gemini-2.5-flash")
    if os.environ.get("OPENROUTER_API_KEY"):
        return openrouter("anthropic/claude-opus-4.6")
    raise RuntimeError(
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, "
        "or OPENROUTER_API_KEY, or add one to ~/.config/structured-ai-sdk/.env.local."
    )


_load_env_local()
MODEL = _default_model()
# MODEL = openai("gpt-5.4-mini")                   # Responses API
# MODEL = openai.chat("gpt-5.4")                   # Chat Completions
# MODEL = google("gemini-2.5-flash")
# MODEL = openrouter("anthropic/claude-opus-4.6")


async def basic():
    result = await generate_text(model=MODEL, prompt="What is the capital of France?")
    print(result.text)
    print(result.usage)


async def streaming():
    result = stream_text(
        model=MODEL,
        system="You are a poet.",
        prompt="Write a haiku about static types.",
    )
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
    print("\n", await result.usage)


async def tools():
    class WeatherInput(BaseModel):
        city: str

    def get_weather(input: WeatherInput) -> str:
        return f"72°F and sunny in {input.city}"

    result = await generate_text(
        model=MODEL,
        prompt="What's the weather in Paris and in London?",
        tools={
            "get_weather": tool(
                description="Get current weather for a city",
                input_schema=WeatherInput,
                execute=get_weather,
            )
        },
        stop_when=step_count_is(5),
    )
    print(result.text)
    for step in result.steps:
        print(f"step: {step.finish_reason}, tool calls: {len(step.tool_calls)}")


async def multimodal():
    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJ0lEQVR42u3NsQkAAAjAsP7/tF7hIASyp6lTCQQCgUAgEAgEgi/BAjLD/C5w/SM9AAAAAElFTkSuQmCC"
    )
    result = await generate_text(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the dominant color of this square? Reply in one sentence.",
                    },
                    {"type": "image", "image": tiny_png},
                ],
            }
        ],
    )
    print(result.text)


async def main():
    await basic()
    await streaming()
    await tools()
    await multimodal()


if __name__ == "__main__":
    asyncio.run(main())
