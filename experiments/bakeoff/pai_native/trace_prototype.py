from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from pai_sdk import (
    CallOptions,
    LanguageModel,
    Output,
    ProviderResult,
    ResponseMetadata,
    TextPart,
    ToolCallPart,
    Usage,
    dump_messages,
    generate_text,
    load_messages,
    load_prompt,
    step_count_is,
    tool,
)
from pai_sdk.messages import ModelMessage
from pai_sdk.stream import ProviderStreamPart


REVIEW_FIXTURE_ROWS: list[dict[str, Any]] = [
    {
        "id": "review_fixture_001",
        "original_question": "What should readers understand from the source conversation?",
        "transcript": [
            {
                "speaker_label": "Asker",
                "message": "Can you summarize the decision and the evidence behind it?",
                "reactions": [],
            },
            {
                "speaker_label": "Expert",
                "message": "The answer depends on the latest data and two caveats.",
                "reactions": ["+1"],
            },
            {
                "speaker_label": "Editor",
                "message": "Please avoid overstating certainty in the summary.",
                "reactions": [],
            },
        ],
        "sections": [
            {"title": "Title", "before": "Draft title that overstates certainty"},
            {
                "title": "Summary",
                "before": "Draft summary with one claim that needs source checking.",
            },
        ],
        "draft_title": "Draft title that overstates certainty",
        "draft_summary": "Draft summary with one claim that needs source checking.",
        "expected_concerns": ["summary_overstates_evidence"],
        "expected_confirmations": ["title_mentions_main_topic"],
        "expected_verdict": "Requires review",
    }
]


@dataclass
class Span:
    id: str
    root_span_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    messages: list[ModelMessage]
    parent_span_id: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rootSpanId": self.root_span_id,
            "parentSpanId": self.parent_span_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "messages": dump_messages(self.messages),
            "usage": self.usage,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    id: str
    spans: list[Span]

    def to_jsonable(self) -> dict[str, Any]:
        return {"id": self.id, "spans": [span.to_jsonable() for span in self.spans]}


class JudgeIssue(BaseModel):
    section: Literal["Title", "Summary"]
    span: str
    concern: str
    suggestion: str


class JudgeOutput(BaseModel):
    issues: list[JudgeIssue] = Field(default_factory=list)
    verdict: Literal["Good", "Requires review", "Bad - rewrite from scratch"]


class ReviewGuidanceInput(BaseModel):
    section: Literal["Title", "Summary"]


@dataclass
class ScriptedModel(LanguageModel):
    """Local provider fixture that records Pai call options and never calls a network."""

    final_payload: dict[str, Any]
    calls: list[CallOptions] = field(default_factory=list)
    provider: str = "fixture"
    model_id: str = "content-review-fixture"

    async def do_generate(self, options: CallOptions) -> ProviderResult:
        self.calls.append(options)
        if len(self.calls) == 1:
            return ProviderResult(
                content=[
                    TextPart(text="I need the editorial review rubric for the draft section."),
                    ToolCallPart(
                        tool_call_id="call_review_guidance_1",
                        tool_name="lookup_review_guidance",
                        input={"section": "Summary"},
                    ),
                ],
                finish_reason="tool-calls",
                usage=Usage(input_tokens=220, output_tokens=24, total_tokens=244),
                response=ResponseMetadata(id="fixture-step-1", model_id=self.model_id),
                provider_metadata={"fixture": {"phase": "tool_request"}},
            )

        return ProviderResult(
            content=[TextPart(text=json.dumps(self.final_payload, ensure_ascii=False))],
            finish_reason="stop",
            usage=Usage(input_tokens=310, output_tokens=74, total_tokens=384),
            response=ResponseMetadata(id="fixture-step-2", model_id=self.model_id),
            provider_metadata={"fixture": {"phase": "structured_judgment"}},
        )

    async def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
        raise NotImplementedError("This prototype only exercises generate_text.")


def _read_fixture_row(index: int) -> dict[str, Any]:
    try:
        return REVIEW_FIXTURE_ROWS[index]
    except IndexError as exc:
        raise IndexError(f"Fixture does not contain row {index}.") from exc


def _redacted_text(value: str | None) -> str:
    value = value or ""
    return f"[redacted text: {len(value)} chars]"


def _format_sanitized_transcript(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages[:6]:
        speaker = message.get("speaker_label") or "Unknown"
        text = message.get("message") or ""
        reactions = message.get("reactions") or []
        lines.append(
            f"{speaker}: [redacted message: {len(text)} chars, reactions={len(reactions)}]"
        )
    if len(messages) > 6:
        lines.append(f"[{len(messages) - 6} additional transcript messages redacted]")
    return "\n".join(lines)


def _section_before(row: dict[str, Any], title: str) -> str:
    for section in row.get("sections") or []:
        if section.get("title") == title:
            return section.get("before") or ""
    return ""


def _safe_review_inputs(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    transcript = row.get("transcript") or []
    sections = row.get("sections") or []
    return {
        "dataset": "generic_content_review_fixture",
        "row_index": row_index,
        "row_shape": {
            "has_original_question": bool(row.get("original_question")),
            "transcript_messages": len(transcript),
            "sections": [section.get("title") for section in sections],
            "expected_concerns": len(row.get("expected_concerns") or []),
            "expected_confirmations": len(row.get("expected_confirmations") or []),
            "expected_verdict": row.get("expected_verdict"),
        },
        "prompt_variables": {
            "original_question": _redacted_text(row.get("original_question")),
            "transcript": _format_sanitized_transcript(transcript),
            "draft_title": _redacted_text(
                row.get("draft_title") or _section_before(row, "Title")
            ),
            "draft_summary": _redacted_text(
                row.get("draft_summary") or _section_before(row, "Summary")
            ),
        },
    }


def _fixture_output(row: dict[str, Any]) -> dict[str, Any]:
    verdict = row.get("expected_verdict") or "Requires review"
    if verdict == "Good":
        return {"issues": [], "verdict": "Good"}
    return {
        "issues": [
            {
                "section": "Summary",
                "span": "[redacted draft span]",
                "concern": "Fixture concern derived from the presence of an expected concern.",
                "suggestion": "Re-check this section against the transcript and revise the unsupported claim.",
            }
        ],
        "verdict": verdict,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def make_prompt_config() -> dict[str, Any]:
    return {
        "name": "content-review-judge-pai-native-prototype",
        "model": "fixture/content-review-fixture",
        "params": {"temperature": 0, "max_output_tokens": 800},
        "messages": [
            {
                "id": "system",
                "role": "system",
                "template": (
                    "You are an editorial reviewer. Review a draft's Title and "
                    "Summary against the original question and transcript. "
                    "Return JSON only."
                ),
            },
            {
                "id": "review-input",
                "role": "user",
                "template": (
                    "original_question: {{original_question}}\n\n"
                    "transcript:\n{{transcript}}\n\n"
                    "draft_title:\n{{draft_title}}\n\n"
                    "draft_summary:\n{{draft_summary}}"
                ),
            },
        ],
        "output": {
            "schema": JudgeOutput.model_json_schema(),
            "name": "content_review_judge_output",
        },
        "tools": {
            "lookup_review_guidance": {
                "description": (
                    "Fetch section-specific editorial review guidance before deciding "
                    "whether a draft section needs revision."
                ),
                "input": {"section": ["Title", "Summary"]},
            }
        },
        "max_steps": 4,
    }


async def build_trace(row_index: int = 0) -> Trace:
    row = _read_fixture_row(row_index)
    safe_inputs = _safe_review_inputs(row, row_index)
    prompt = load_prompt(make_prompt_config())
    rendered_messages = prompt.render(safe_inputs["prompt_variables"])

    def lookup_review_guidance(input: ReviewGuidanceInput) -> dict[str, Any]:
        return {
            "section": input.section,
            "checks": [
                "faithful_to_transcript",
                "answers_original_question",
                "editorial_brevity",
            ],
            "source": "local_fixture_redacted",
        }

    model = ScriptedModel(final_payload=_fixture_output(row))
    result = await generate_text(
        model=model,
        messages=rendered_messages,
        output=Output.object(JudgeOutput, name="content_review_judge_output"),
        tools={
            "lookup_review_guidance": tool(
                description=prompt.tools["lookup_review_guidance"].description,
                input_schema=ReviewGuidanceInput,
                execute=lookup_review_guidance,
            )
        },
        stop_when=step_count_is(4),
        temperature=0,
        max_output_tokens=800,
    )

    full_history = [*rendered_messages, *result.response.messages]
    # Prove the provider-near history can round-trip through the public serializer.
    full_history = load_messages(dump_messages(full_history))

    root_span_id = f"span_{uuid.uuid4().hex[:12]}"
    span = Span(
        id=root_span_id,
        root_span_id=root_span_id,
        inputs={
            **safe_inputs,
            "prompt": {
                "name": prompt.name,
                "variables": prompt.variables,
                "message_ids": [message.id for message in prompt.messages],
                "declared_tools": list(prompt.tools),
                "output_schema_name": "JudgeOutput",
            },
        },
        outputs={
            "text": result.text,
            "object": _jsonable(result.output),
            "finish_reason": result.finish_reason,
            "tool_calls": [_jsonable(call) for step in result.steps for call in step.tool_calls],
            "tool_results": [_jsonable(item) for step in result.steps for item in step.tool_results],
        },
        messages=full_history,
        usage=_jsonable(result.total_usage),
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "embedded_generic_fixture",
            "raw_fixture_content": "not persisted; values redacted to lengths/counts",
            "provider": model.provider,
            "model_id": model.model_id,
            "provider_call_count": len(model.calls),
            "step_finish_reasons": [step.finish_reason for step in result.steps],
            "response_message_roles": [message.role for message in result.response.messages],
        },
    )
    return Trace(id=f"trace_{uuid.uuid4().hex[:12]}", spans=[span])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    trace = asyncio.run(build_trace(args.row_index))
    print(
        json.dumps(
            trace.to_jsonable(),
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
