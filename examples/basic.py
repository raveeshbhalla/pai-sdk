"""Examples — pick a model, set the matching API key, and run.

    ANTHROPIC_API_KEY=... python examples/basic.py
"""

import asyncio

from pydantic import BaseModel

from pai_sdk import generate_text, step_count_is, stream_text, tool
from pai_sdk.providers import anthropic, google, openai, openrouter  # noqa: F401

MODEL = anthropic("claude-opus-4-8")
# MODEL = openai("gpt-5.4")                        # Responses API
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
    result = await generate_text(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image",
                        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
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
