"""Tests for the AI SDK v6 core extensions (sources, generated files,
provider-executed tools, structured warnings, usage detail breakdowns,
request echo, raw chunks, file references, v6 tool-result content items,
and tool-approval types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest

from model_message import (
    AssistantModelMessage,
    CallOptions,
    CallWarning,
    ContentOutput,
    DocumentSourcePart,
    Finish,
    FileDataContentItem,
    FileIdData,
    FilePart,
    FileUrlContentItem,
    GeneratedFile,
    ImageDataContentItem,
    ImagePart,
    ImageUrlContentItem,
    InputTokenDetails,
    LanguageModel,
    OutputTokenDetails,
    ProviderResult,
    ResponseMetadata,
    SourceStreamPart,
    TextPart,
    ToolApprovalRequest,
    ToolApprovalResponse,
    ToolCallPart,
    ToolModelMessage,
    ToolResultEvent,
    ToolResultPart,
    UrlSourcePart,
    Usage,
    generate_text,
    model_message_adapter,
    stream_text,
)
from model_message.messages import ErrorTextOutput, TextContentItem
from model_message.stream import (
    Finish as StreamFinish,
    ProviderStreamPart,
    RawPart,
    ResponseMetadataPart,
)


# ---------------------------------------------------------------------------
# Helper models for streaming scenarios
# ---------------------------------------------------------------------------


@dataclass
class ScriptedStreamModel(LanguageModel):
    """Yields a fixed list of provider stream parts."""

    parts: list[Any] = field(default_factory=list)
    calls: list[CallOptions] = field(default_factory=list)
    provider: str = "fake"
    model_id: str = "fake-1"

    async def do_generate(self, options: CallOptions) -> ProviderResult:  # pragma: no cover
        raise NotImplementedError

    async def do_stream(self, options: CallOptions) -> AsyncIterator[ProviderStreamPart]:
        self.calls.append(options)
        for part in self.parts:
            yield part


def _finish() -> StreamFinish:
    return StreamFinish(
        finish_reason="stop",
        total_usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


# ---------------------------------------------------------------------------
# 1. Sources
# ---------------------------------------------------------------------------


def test_source_part_round_trip():
    msg = model_message_adapter.validate_python(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "source",
                    "sourceType": "url",
                    "id": "s1",
                    "url": "https://example.com",
                    "title": "Example",
                },
                {
                    "type": "source",
                    "sourceType": "document",
                    "id": "s2",
                    "mediaType": "application/pdf",
                    "title": "Doc",
                    "filename": "doc.pdf",
                },
            ],
        }
    )
    assert isinstance(msg.content[0], UrlSourcePart)
    assert isinstance(msg.content[1], DocumentSourcePart)
    dumped = msg.model_dump(by_alias=True, exclude_none=True)
    assert dumped["content"][0] == {
        "type": "source",
        "sourceType": "url",
        "id": "s1",
        "url": "https://example.com",
        "title": "Example",
    }
    assert dumped["content"][1]["sourceType"] == "document"


@pytest.mark.asyncio
async def test_generate_result_sources():
    from tests.conftest import FakeModel

    src = UrlSourcePart(id="s1", url="https://example.com", title="Example")
    model = FakeModel(
        steps=[
            ProviderResult(
                content=[TextPart(text="see source"), src],
                finish_reason="stop",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
            )
        ]
    )
    result = await generate_text(model=model, prompt="hi")
    assert len(result.sources) == 1
    assert isinstance(result.sources[0], UrlSourcePart)
    assert result.steps[0].sources[0].id == "s1"


@pytest.mark.asyncio
async def test_stream_result_sources():
    src = UrlSourcePart(id="s1", url="https://example.com")
    model = ScriptedStreamModel(
        parts=[
            ResponseMetadataPart(id="r1", model_id="fake-1"),
            SourceStreamPart(source=src),
            _finish(),
        ]
    )
    result = stream_text(model=model, prompt="hi")
    seen_source = False
    async for part in result.full_stream:
        if isinstance(part, SourceStreamPart):
            seen_source = True
    assert seen_source
    sources = await result.sources
    assert len(sources) == 1
    assert sources[0].id == "s1"


# ---------------------------------------------------------------------------
# 2. Generated files
# ---------------------------------------------------------------------------


def test_generated_file_accessors():
    f = GeneratedFile(data=b"abc", media_type="image/png")
    assert f.bytes == b"abc"
    assert f.base64 == "YWJj"
    assert f.media_type == "image/png"


@pytest.mark.asyncio
async def test_generate_result_files():
    from tests.conftest import FakeModel

    model = FakeModel(
        steps=[
            ProviderResult(
                content=[FilePart(data=b"\x89PNG", media_type="image/png")],
                finish_reason="stop",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
            )
        ]
    )
    result = await generate_text(model=model, prompt="hi")
    assert len(result.files) == 1
    assert isinstance(result.files[0], GeneratedFile)
    assert result.files[0].data == b"\x89PNG"
    # content still has the FilePart
    assert any(isinstance(p, FilePart) for p in result.content)


@pytest.mark.asyncio
async def test_stream_result_files():
    from model_message.stream import FilePartEvent

    model = ScriptedStreamModel(
        parts=[
            ResponseMetadataPart(id="r1", model_id="fake-1"),
            FilePartEvent(media_type="image/png", data=b"\x89PNG"),
            _finish(),
        ]
    )
    result = stream_text(model=model, prompt="hi")
    await result.consume_stream()
    files = await result.files
    assert len(files) == 1
    assert files[0].data == b"\x89PNG"


# ---------------------------------------------------------------------------
# 3. Provider-executed tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_executed_tool_not_run_locally():
    from model_message import tool
    from tests.conftest import FakeModel

    executed = {"count": 0}

    def run(_inp: Any) -> str:
        executed["count"] += 1
        return "local"

    # Provider returns a tool call marked provider_executed plus its result.
    model = FakeModel(
        steps=[
            ProviderResult(
                content=[
                    ToolCallPart(
                        tool_call_id="c1",
                        tool_name="search",
                        input={"q": "x"},
                        provider_executed=True,
                    ),
                    ToolResultPart(
                        tool_call_id="c1",
                        tool_name="search",
                        output={"type": "json", "value": {"hits": 3}},
                        provider_executed=True,
                    ),
                ],
                finish_reason="tool-calls",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
            )
        ]
    )
    result = await generate_text(
        model=model,
        prompt="hi",
        tools={"search": tool(description="d", input_schema={"type": "object"}, execute=run)},
    )
    # local execute never ran
    assert executed["count"] == 0
    # provider result surfaced on the step
    assert len(result.steps) == 1
    prov = [r for r in result.tool_results if r.provider_executed]
    assert len(prov) == 1
    assert prov[0].tool_call_id == "c1"
    # loop did NOT continue (only one provider call, no local results pending)
    assert len(model.calls) == 1
    # no extra tool message was generated for the provider-executed result
    tool_msgs = [m for m in result.response.messages if isinstance(m, ToolModelMessage)]
    assert tool_msgs == []


@pytest.mark.asyncio
async def test_stream_provider_executed_tool_result_event():
    model = ScriptedStreamModel(
        parts=[
            ResponseMetadataPart(id="r1", model_id="fake-1"),
            ToolResultEvent(
                tool_call_id="c1",
                tool_name="search",
                input={"q": "x"},
                output={"hits": 1},
                model_output=None,
                provider_executed=True,
            ),
            _finish(),
        ]
    )
    result = stream_text(model=model, prompt="hi")
    saw_event = False
    async for part in result.full_stream:
        if isinstance(part, ToolResultEvent):
            saw_event = True
            assert part.provider_executed is True
    assert saw_event
    tool_results = await result.tool_results
    assert any(r.provider_executed for r in tool_results)


# ---------------------------------------------------------------------------
# 4. Structured warnings
# ---------------------------------------------------------------------------


def test_call_warning_coercion():
    assert CallWarning.coerce("oops") == CallWarning(type="other", message="oops")
    cw = CallWarning(type="unsupported-setting", setting="seed")
    assert CallWarning.coerce(cw) is cw
    assert CallWarning.coerce(
        {"type": "unsupported-tool", "message": "no"}
    ) == CallWarning(type="unsupported-tool", message="no")


@pytest.mark.asyncio
async def test_engine_coerces_string_warnings():
    from tests.conftest import FakeModel

    model = FakeModel(
        steps=[
            ProviderResult(
                content=[TextPart(text="hi")],
                finish_reason="stop",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
                warnings=["temperature is not supported"],
            )
        ]
    )
    result = await generate_text(model=model, prompt="hi")
    assert len(result.warnings) == 1
    assert isinstance(result.warnings[0], CallWarning)
    assert result.warnings[0].type == "other"
    assert result.warnings[0].message == "temperature is not supported"


# ---------------------------------------------------------------------------
# 5. Usage detail breakdowns
# ---------------------------------------------------------------------------


def test_usage_detail_addition():
    a = Usage(
        input_tokens=10,
        input_token_details=InputTokenDetails(no_cache_tokens=8, cache_read_tokens=2),
        output_token_details=OutputTokenDetails(text_tokens=4, reasoning_tokens=1),
    )
    b = Usage(
        input_tokens=5,
        input_token_details=InputTokenDetails(no_cache_tokens=5),
        output_token_details=None,
    )
    total = a + b
    assert total.input_tokens == 15
    assert total.input_token_details.no_cache_tokens == 13
    assert total.input_token_details.cache_read_tokens == 2
    # b had no output details, so a's survive
    assert total.output_token_details.text_tokens == 4

    # both None -> None
    c = Usage(input_tokens=1)
    d = Usage(input_tokens=1)
    assert (c + d).input_token_details is None


# ---------------------------------------------------------------------------
# 6. Request metadata echo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_request_echo():
    from tests.conftest import FakeModel

    model = FakeModel(
        steps=[
            ProviderResult(
                content=[TextPart(text="hi")],
                finish_reason="stop",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
                request={"model": "fake-1", "messages": []},
            )
        ]
    )
    result = await generate_text(model=model, prompt="hi")
    assert result.request == {"model": "fake-1", "messages": []}
    assert result.steps[0].request == {"model": "fake-1", "messages": []}


@pytest.mark.asyncio
async def test_stream_request_echo():
    model = ScriptedStreamModel(
        parts=[
            ResponseMetadataPart(
                id="r1", model_id="fake-1", request={"model": "fake-1"}
            ),
            _finish(),
        ]
    )
    result = stream_text(model=model, prompt="hi")
    from model_message.stream import FinishStep

    finish_step_request = None
    async for part in result.full_stream:
        if isinstance(part, FinishStep):
            finish_step_request = part.request
    await result.consume_stream()
    assert finish_step_request == {"model": "fake-1"}
    assert result.steps[0].request == {"model": "fake-1"}


# ---------------------------------------------------------------------------
# 7. Raw chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_raw_chunks_plumbing_and_raw_part_forwarded():
    model = ScriptedStreamModel(
        parts=[
            ResponseMetadataPart(id="r1", model_id="fake-1"),
            RawPart(raw_value={"some": "chunk"}),
            _finish(),
        ]
    )
    result = stream_text(model=model, prompt="hi", include_raw_chunks=True)
    saw_raw = False
    async for part in result.full_stream:
        if isinstance(part, RawPart):
            saw_raw = True
            assert part.raw_value == {"some": "chunk"}
    assert saw_raw
    # plumbed through to CallOptions
    assert model.calls[0].include_raw_chunks is True


# ---------------------------------------------------------------------------
# 8. File references
# ---------------------------------------------------------------------------


def test_file_id_data_validation_and_serialization():
    fp = FilePart(data=FileIdData(id="file_123"), media_type="application/pdf")
    dumped = fp.model_dump(by_alias=True, exclude_none=True)
    assert dumped["data"] == {"type": "file-id", "id": "file_123"}

    ip = ImagePart(image=FileIdData(id={"fileId": "abc"}))
    dumped_img = ip.model_dump(by_alias=True, exclude_none=True)
    assert dumped_img["image"] == {"type": "file-id", "id": {"fileId": "abc"}}

    # round-trip from dict
    restored = FilePart.model_validate(
        {"type": "file", "data": {"type": "file-id", "id": "x"}, "mediaType": "image/png"}
    )
    assert isinstance(restored.data, FileIdData)
    assert restored.data.id == "x"

    # inline bytes still work
    inline = FilePart(data=b"raw", media_type="image/png")
    assert inline.model_dump(by_alias=True)["data"] == "cmF3"


# ---------------------------------------------------------------------------
# 9. v6 richer tool-result content items
# ---------------------------------------------------------------------------


def test_v6_content_output_items_round_trip():
    msg = model_message_adapter.validate_python(
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "c1",
                    "toolName": "t",
                    "output": {
                        "type": "content",
                        "value": [
                            {"type": "text", "text": "hi"},
                            {
                                "type": "file-data",
                                "data": "YWJj",
                                "mediaType": "application/pdf",
                                "filename": "f.pdf",
                            },
                            {"type": "file-url", "url": "https://f"},
                            {"type": "image-data", "data": "YWJj", "mediaType": "image/png"},
                            {"type": "image-url", "url": "https://i"},
                        ],
                    },
                }
            ],
        }
    )
    items = msg.content[0].output.value
    assert isinstance(items[0], TextContentItem)
    assert isinstance(items[1], FileDataContentItem)
    assert isinstance(items[2], FileUrlContentItem)
    assert isinstance(items[3], ImageDataContentItem)
    assert isinstance(items[4], ImageUrlContentItem)
    dumped = msg.model_dump(by_alias=True, exclude_none=True)
    out = dumped["content"][0]["output"]["value"]
    assert out[1] == {
        "type": "file-data",
        "data": "YWJj",
        "mediaType": "application/pdf",
        "filename": "f.pdf",
    }
    assert out[2] == {"type": "file-url", "url": "https://f"}


# ---------------------------------------------------------------------------
# 10. Tool approvals — types only
# ---------------------------------------------------------------------------


def test_tool_approval_request_in_assistant_content():
    msg = model_message_adapter.validate_python(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-approval-request",
                    "approvalId": "a1",
                    "toolCallId": "c1",
                    "isAutomatic": True,
                }
            ],
        }
    )
    assert isinstance(msg.content[0], ToolApprovalRequest)
    dumped = msg.model_dump(by_alias=True, exclude_none=True)
    assert dumped["content"][0] == {
        "type": "tool-approval-request",
        "approvalId": "a1",
        "toolCallId": "c1",
        "isAutomatic": True,
    }


def test_tool_approval_response_in_tool_content():
    msg = model_message_adapter.validate_python(
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-approval-response",
                    "approvalId": "a1",
                    "approved": False,
                    "reason": "denied",
                }
            ],
        }
    )
    assert isinstance(msg.content[0], ToolApprovalResponse)
    assert msg.content[0].approved is False


@pytest.mark.asyncio
async def test_loop_inert_with_approval_request():
    """An assistant approval-request part must not break the loop or be
    treated as a tool call."""
    from tests.conftest import FakeModel

    model = FakeModel(
        steps=[
            ProviderResult(
                content=[
                    TextPart(text="ok"),
                    ToolApprovalRequest(approval_id="a1", tool_call_id="c1"),
                ],
                finish_reason="stop",
                usage=Usage(input_tokens=1, output_tokens=1),
                response=ResponseMetadata(id="r1", model_id="fake-1"),
            )
        ]
    )
    result = await generate_text(model=model, prompt="hi")
    assert result.text == "ok"
    assert result.tool_calls == []
    # approval request is preserved (replayable) in the assistant message
    asst = [m for m in result.response.messages if isinstance(m, AssistantModelMessage)]
    assert any(
        isinstance(p, ToolApprovalRequest)
        for m in asst
        for p in m.content
    )


async def test_streaming_provider_metadata_via_finish_part():
    """Providers can attach provider_metadata to their Finish part; the engine
    copies it onto the step and FinishStep."""
    from model_message import stream_text
    from model_message.provider import CallOptions
    from model_message.stream import Finish, ResponseMetadataPart, TextDelta, TextEnd, TextStart

    from conftest import FakeModel, text_step

    class MetadataModel(FakeModel):
        async def do_stream(self, options: CallOptions):
            yield ResponseMetadataPart(id="r1", model_id="fake-1")
            yield TextStart(id="0")
            yield TextDelta(id="0", text="hi")
            yield TextEnd(id="0")
            yield Finish(
                finish_reason="stop",
                total_usage=text_step("hi").usage,
                provider_metadata={"fake": {"cost": 0.01}},
            )

    result = stream_text(model=MetadataModel(), prompt="hi")
    finish_steps = [p async for p in result.full_stream if p.type == "finish-step"]
    assert finish_steps[0].provider_metadata == {"fake": {"cost": 0.01}}
    steps = await result.all_steps
    assert steps[0].provider_metadata == {"fake": {"cost": 0.01}}
