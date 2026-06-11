from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator


def _load_env_local() -> None:
    """Load KEY=VALUE pairs from .env.local (repo root) for live tests.
    Existing environment variables win."""
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_local()

from pai_sdk import (
    CallOptions,
    Finish,
    LanguageModel,
    ProviderResult,
    ResponseMetadata,
    TextDelta,
    TextEnd,
    TextPart,
    TextStart,
    ToolCallPart,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputStart,
    Usage,
)
from pai_sdk.stream import ProviderStreamPart, ResponseMetadataPart


@dataclass
class FakeModel(LanguageModel):
    """Scripted model: returns canned step results in order, records calls."""

    steps: list[ProviderResult] = field(default_factory=list)
    calls: list[CallOptions] = field(default_factory=list)
    provider: str = "fake"
    model_id: str = "fake-1"

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        self.calls.append(options)
        return self.steps[len(self.calls) - 1]

    async def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
        self.calls.append(options)
        result = self.steps[len(self.calls) - 1]
        yield ResponseMetadataPart(id=result.response.id, model_id=result.response.model_id)
        for index, part in enumerate(result.content):
            if isinstance(part, TextPart):
                yield TextStart(id=str(index))
                # split text into two deltas to exercise accumulation
                text = part.text
                mid = max(1, len(text) // 2)
                yield TextDelta(id=str(index), text=text[:mid])
                if text[mid:]:
                    yield TextDelta(id=str(index), text=text[mid:])
                yield TextEnd(id=str(index))
            elif isinstance(part, ToolCallPart):
                yield ToolInputStart(id=part.tool_call_id, tool_name=part.tool_name)
                yield ToolInputDelta(id=part.tool_call_id, delta="{}")
                yield ToolInputEnd(id=part.tool_call_id)
                yield part
        yield Finish(
            finish_reason=result.finish_reason,
            raw_finish_reason=result.raw_finish_reason,
            total_usage=result.usage,
        )


def text_step(text: str, **kwargs: Any) -> ProviderResult:
    return ProviderResult(
        content=[TextPart(text=text)],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetadata(id="resp_1", model_id="fake-1"),
        **kwargs,
    )


def tool_step(tool_name: str, call_id: str = "call_1", tool_input: Any = None) -> ProviderResult:
    return ProviderResult(
        content=[
            TextPart(text="Let me check."),
            ToolCallPart(tool_call_id=call_id, tool_name=tool_name, input=tool_input or {}),
        ],
        finish_reason="tool-calls",
        usage=Usage(input_tokens=20, output_tokens=8, total_tokens=28),
        response=ResponseMetadata(id="resp_0", model_id="fake-1"),
    )
