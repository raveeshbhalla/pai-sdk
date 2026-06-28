"""DSPy bridge prototype for canonical Trace/Span capture.

Run from the repo root with an environment that has DSPy 3.3.0b1 or newer:

    python experiments/bakeoff/dspy_bridge/dspy_bridge_trace_prototype.py

This deliberately uses a local fake typed LM. It does not call external APIs.
The fixture row is reduced to a schema/count/length shape before it enters DSPy
so the emitted trace is safe to inspect.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import dspy
from dspy.core import types as lm_types

from pai_sdk import (
    AssistantModelMessage,
    JsonOutput,
    SystemModelMessage,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    UserModelMessage,
)
from pai_sdk.serialize import dump_messages


REVIEW_FIXTURE_ROW: dict[str, Any] = {
    "id": "review_fixture_001",
    "document_id": "document_fixture_001",
    "original_question": "What should readers understand from the source conversation?",
    "transcript": [
        {
            "index": 0,
            "message": "Can you summarize the decision and the evidence behind it?",
            "speaker_label": "Asker",
            "user_type": "requester",
            "reactions": [],
        },
        {
            "index": 1,
            "message": "The answer depends on the latest data and two caveats.",
            "speaker_label": "Expert",
            "user_type": "expert",
            "reactions": ["+1"],
        },
    ],
    "sections": [
        {"title": "Title", "before": "Draft title that overstates certainty"},
        {"title": "Summary", "before": "Draft summary with one claim to check"},
    ],
    "draft_title": "Draft title that overstates certainty",
    "draft_summary": "Draft summary with one claim to check",
    "expected_concerns": ["summary_overstates_evidence"],
    "expected_confirmations": ["title_mentions_main_topic"],
    "expected_verdict": "Requires review",
}


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_wire(self) -> dict[str, int | None]:
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
        }


@dataclass
class Span:
    id: str
    root_span_id: str
    parent_span_id: str | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    messages: list[dict[str, Any]]
    usage: Usage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rootSpanId": self.root_span_id,
            "parentSpanId": self.parent_span_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "messages": self.messages,
            "usage": self.usage.to_wire() if self.usage else None,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    id: str
    spans: list[Span]

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "spans": [span.to_wire() for span in self.spans]}


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def load_sanitized_review_shape() -> dict[str, Any]:
    """Use a generic review-row shape without carrying raw fixture content."""
    row = REVIEW_FIXTURE_ROW
    return {
        "row_ref": stable_id(row.get("id", "")),
        "document_ref": stable_id(row.get("document_id", "")),
        "original_question_chars": len(row.get("original_question", "")),
        "transcript_turn_count": len(row.get("transcript", [])),
        "transcript_turn_keys": sorted(row["transcript"][0].keys()) if row.get("transcript") else [],
        "section_count": len(row.get("sections", [])),
        "section_keys": sorted(row["sections"][0].keys()) if row.get("sections") else [],
        "draft_title_chars": len(row.get("draft_title", "")),
        "draft_summary_chars": len(row.get("draft_summary", "")),
        "expected_concern_count": len(row.get("expected_concerns", [])),
        "expected_confirmation_count": len(row.get("expected_confirmations", [])),
        "expected_verdict_label": row.get("expected_verdict"),
    }


def part_text(part: Any) -> str:
    return getattr(part, "text", "")


def lm_text(message: dspy.LMMessage) -> str:
    return "".join(part_text(part) for part in message.parts if getattr(part, "type", None) == "text")


def lm_part_to_pai_part(part: Any) -> Any:
    part_type = getattr(part, "type", None)
    if part_type == "text":
        return TextPart(text=part.text)
    if part_type == "tool_call":
        return ToolCallPart(
            tool_call_id=part.id or f"call_{stable_id(part.name)}",
            tool_name=part.name,
            input=dict(part.args),
            provider_options={"dspy": dict(part.provider_data)} if part.provider_data else None,
        )
    if part_type == "tool_result":
        text = "".join(part_text(item) for item in part.content if getattr(item, "type", None) == "text")
        output = JsonOutput(value={"text": text, "is_error": part.is_error})
        return ToolResultPart(
            tool_call_id=part.call_id or "unknown",
            tool_name=part.name or "unknown",
            output=output,
            provider_options={"dspy": dict(part.provider_data)} if part.provider_data else None,
        )
    return TextPart(text=f"[unmapped DSPy part: {part_type}]")


def lm_request_to_model_messages(request: dspy.LMRequest) -> list[Any]:
    messages: list[Any] = []
    for message in request.messages:
        parts = [lm_part_to_pai_part(part) for part in message.parts]
        content: Any = parts
        if all(isinstance(part, TextPart) for part in parts):
            content = "".join(part.text for part in parts)

        metadata = {"provider_options": {"dspy": dict(message.metadata)}} if message.metadata else {}
        if message.role == "system":
            messages.append(SystemModelMessage(content=str(content), **metadata))
        elif message.role == "assistant":
            messages.append(AssistantModelMessage(content=content, **metadata))
        elif message.role == "tool":
            tool_parts = [part for part in parts if isinstance(part, ToolResultPart)]
            messages.append(ToolModelMessage(content=tool_parts, **metadata))
        else:
            messages.append(UserModelMessage(content=content, **metadata))
    return messages


class TypedChatAdapter(dspy.ChatAdapter):
    """Temporary DSPy 3.3.0b1 shim to keep adapter-built LMRequest intact."""

    def _call_lm(self, lm: dspy.BaseLM, request: dspy.LMRequest) -> dspy.LMResponse:
        if getattr(lm, "forward_contract", "legacy") == "typed_lm":
            return lm.forward(request)
        return super()._call_lm(lm, request)


class ContentReviewSignature(dspy.Signature):
    """Judge whether a draft review needs changes from sanitized row features."""

    row_shape: dict[str, Any] = dspy.InputField()
    verdict: str = dspy.OutputField(desc="One of: ok, needs_changes")
    concerns: list[str] = dspy.OutputField()
    confirmations: list[str] = dspy.OutputField()


class FakeContentReviewLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def __init__(self) -> None:
        super().__init__(model="fake/content-review-judge")
        self.requests: list[dspy.LMRequest] = []
        self.responses: list[dspy.LMResponse] = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        user_text = next((lm_text(message) for message in reversed(request.messages) if message.role == "user"), "")
        verdict = "needs_changes" if "expected_concern_count" in user_text else "ok"
        response = dspy.LMResponse(
            model=request.model,
            outputs=[
                lm_types.LMOutput(
                    parts=[
                        lm_types.LMTextPart(
                            text=(
                                "[[ ## verdict ## ]]\n"
                                f"{verdict}\n\n"
                                "[[ ## concerns ## ]]\n"
                                '["draft-review shape indicates at least one concern bucket"]\n\n'
                                "[[ ## confirmations ## ]]\n"
                                '["adapter produced a typed LMRequest from DSPy fields"]\n\n'
                                "[[ ## completed ## ]]"
                            )
                        )
                    ],
                    finish_reason="stop",
                    provider_data={"fake_finish": "structured_fields"},
                )
            ],
            usage=lm_types.LMUsage(input_tokens=151, output_tokens=37, total_tokens=188),
            response_id="fake-content-review-response",
            provider_data={"provider": "fake-local"},
            metadata={"bridge": "dspy-typed-request", "request_message_count": len(request.messages)},
        )
        self.responses.append(response)
        return response


def usage_from_response(response: dspy.LMResponse) -> Usage | None:
    if response.usage is None:
        return None
    usage = response.usage
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def run_predict_span(row_shape: dict[str, Any]) -> Span:
    lm = FakeContentReviewLM()
    old_settings = dict(dspy.settings.config)
    dspy.configure(lm=lm, adapter=TypedChatAdapter())
    try:
        prediction = dspy.Predict(ContentReviewSignature)(row_shape=row_shape)
    finally:
        dspy.settings.configure(**old_settings)

    request = lm.requests[-1]
    response = lm.responses[-1]
    span_id = f"span_{uuid.uuid4().hex[:12]}"
    return Span(
        id=span_id,
        root_span_id=span_id,
        parent_span_id=None,
        inputs={"row_shape": row_shape},
        outputs={
            "verdict": prediction.verdict,
            "concerns": prediction.concerns,
            "confirmations": prediction.confirmations,
        },
        messages=dump_messages(lm_request_to_model_messages(request)),
        usage=usage_from_response(response),
        metadata={
            "dspy_version": dspy.__version__,
            "signature": "ContentReviewSignature",
            "lm_model": request.model,
            "response_id": response.response_id,
            "provider_data": response.provider_data,
            "response_metadata": response.metadata,
            "uses_private_adapter_shim": True,
        },
    )


def synthetic_tool_span(root_span_id: str, row_ref: str) -> Span:
    call_id = "call_fetch_review_context"
    messages = [
        SystemModelMessage(content="Use tools when structured row metadata is required."),
        UserModelMessage(content=f"Review sanitized fixture row ref {row_ref}."),
        AssistantModelMessage(
            content=[
                TextPart(text="I will fetch the sanitized row context."),
                ToolCallPart(
                    tool_call_id=call_id,
                    tool_name="fetch_review_context",
                    input={"row_ref": row_ref},
                    provider_options={"dspy": {"part_type": "LMToolCallPart"}},
                ),
            ]
        ),
        ToolModelMessage(
            content=[
                ToolResultPart(
                    tool_call_id=call_id,
                    tool_name="fetch_review_context",
                    output=JsonOutput(value={"section_count": 2, "has_expected_labels": True}),
                    provider_options={"dspy": {"part_type": "LMToolResultPart"}},
                )
            ]
        ),
    ]
    return Span(
        id=f"span_{uuid.uuid4().hex[:12]}",
        root_span_id=root_span_id,
        parent_span_id=root_span_id,
        inputs={"tool_name": "fetch_review_context", "row_ref": row_ref},
        outputs={"section_count": 2, "has_expected_labels": True},
        messages=dump_messages(messages),
        usage=None,
        metadata={
            "scenario": "synthetic provider-native tool history",
            "gap": "DSPy ReAct stores tool use in trajectory fields in 3.3.0b1; this span shows the target ModelMessage mapping.",
        },
    )


def main() -> None:
    row_shape = load_sanitized_review_shape()
    predict_span = run_predict_span(row_shape)
    tool_span = synthetic_tool_span(predict_span.root_span_id, row_shape["row_ref"])
    trace = Trace(id=f"trace_{uuid.uuid4().hex[:12]}", spans=[predict_span, tool_span])

    print(json.dumps(trace.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
