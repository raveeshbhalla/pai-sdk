"""Trace-backed structured history without provider API keys.

Run:

    python examples/trace_history.py

This uses a tiny scripted model so the example is deterministic and offline.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from pai_sdk import (
    CallOptions,
    Finish,
    LanguageModel,
    OptimizerTarget,
    ProviderResult,
    ResponseMetadata,
    TextDelta,
    TextEnd,
    TextPart,
    TextStart,
    Usage,
    apply_optimizer_target,
    dump_trace_json,
    load_prompt,
    load_trace,
    replay_span,
    stream_trace,
)
from pai_sdk.stream import ProviderStreamPart, ResponseMetadataPart


class ScriptedModel(LanguageModel):
    provider = "example"
    model_id = "scripted"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[CallOptions] = []

    def _next_result(self, options: CallOptions) -> ProviderResult:
        self.calls.append(options)
        index = len(self.calls) - 1
        text = self.texts[min(index, len(self.texts) - 1)]
        return ProviderResult(
            content=[TextPart(text=text)],
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            response=ResponseMetadata(id=f"resp_{index}", model_id=self.model_id),
        )

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        return self._next_result(options)

    async def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
        result = self._next_result(options)
        yield ResponseMetadataPart(id=result.response.id, model_id=result.response.model_id)
        text = result.content[0].text
        yield TextStart(id="0")
        yield TextDelta(id="0", text=text[: len(text) // 2])
        yield TextDelta(id="0", text=text[len(text) // 2 :])
        yield TextEnd(id="0")
        yield Finish(finish_reason="stop", total_usage=result.usage)


async def main() -> None:
    prompt = load_prompt(
        {
            "name": "support-triage",
            "input": {"company": "string", "ticket": "string"},
            "messages": [
                {
                    "id": "system",
                    "role": "system",
                    "template": "You triage {{company}} tickets.",
                },
                {"id": "ticket", "role": "user", "template": "Ticket: {{ticket}}"},
            ],
            "output": {"urgency": ["low", "high"], "summary": "string"},
        }
    )

    traced = await prompt.generate_trace(
        {"company": "Acme", "ticket": "Login is broken"},
        model=ScriptedModel('{"urgency":"high","summary":"Login is broken"}'),
    )

    span = traced.trace.spans[0]
    print("structured output:", traced.output)
    print("trace roles:", [message.role for message in span.messages])

    dumped = dump_trace_json(traced.trace, indent=2)
    loaded = load_trace(dumped)
    replayed = await replay_span(loaded.spans[0], model=ScriptedModel("replayed answer"))
    print("semantic replay:", replayed.text)

    streamed = stream_trace(
        model=ScriptedModel("streamed answer"),
        prompt="Summarize the incident.",
        inputs={"task": "incident-summary"},
    )
    streamed_text = "".join([delta async for delta in streamed.text_stream])
    streamed_trace = await streamed.trace
    print("streamed text:", streamed_text)
    print("stream trace inputs:", streamed_trace.spans[0].inputs)

    evolved = apply_optimizer_target(
        prompt,
        OptimizerTarget.message_template("system"),
        "You triage {{company}} tickets with concise severity labels.",
    )
    print("evolved system:", evolved.messages[0].template)


if __name__ == "__main__":
    asyncio.run(main())
