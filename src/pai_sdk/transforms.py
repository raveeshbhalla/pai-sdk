"""Stream transforms — the AI SDK experimental_transform / smoothStream analog.

A transform is a callable taking an async iterator of TextStreamPart and
returning a (transformed) async iterator of TextStreamPart:

    Transform = Callable[[AsyncIterator[TextStreamPart]], AsyncIterator[TextStreamPart]]

stream_text(transform=...) applies one transform or a list (composed in order)
at the subscription/emit layer. Transforms observe the parts as they are
emitted to subscribers (full_stream / text_stream / partial_output_stream) and
on_chunk; the per-step content and the awaitable aggregates (final text, usage,
etc.) are computed by the drive loop from the *raw* provider parts upstream, so
they are NOT affected by transforms. (AI SDK applies transforms inside the
pipeline where aggregates are derived from transformed output; here aggregates
stay on untransformed content for engine simplicity — see stream_text docstring.)
"""

from __future__ import annotations

import asyncio
import re
from typing import (
    AsyncIterator,
    Callable,
    List,
    Optional,
    Union,
)

from .stream import (
    TextDelta,
    TextEnd,
    TextStart,
    TextStreamPart,
)

Transform = Callable[[AsyncIterator[TextStreamPart]], AsyncIterator[TextStreamPart]]


def compose_transforms(
    transforms: Union[Transform, List[Transform], None],
) -> Optional[Transform]:
    """Compose one or a list of transforms into a single transform applied in
    order (the first transform sees the raw stream, the last produces output)."""
    if transforms is None:
        return None
    if callable(transforms):
        items: List[Transform] = [transforms]
    else:
        items = list(transforms)
    if not items:
        return None

    def composed(stream: AsyncIterator[TextStreamPart]) -> AsyncIterator[TextStreamPart]:
        out = stream
        for t in items:
            out = t(out)
        return out

    return composed


ChunkingFn = Callable[[str], Optional[str]]


def _word_chunker(buffer: str) -> Optional[str]:
    """Return the next chunk (text up to and including a trailing word boundary)
    or None if no complete chunk is buffered yet."""
    match = re.search(r"\S+\s+", buffer)
    if match is None:
        return None
    return buffer[: match.end()]


def _line_chunker(buffer: str) -> Optional[str]:
    idx = buffer.find("\n")
    if idx == -1:
        return None
    return buffer[: idx + 1]


def _regex_chunker(pattern: "re.Pattern[str]") -> ChunkingFn:
    def chunker(buffer: str) -> Optional[str]:
        match = pattern.search(buffer)
        if match is None or match.end() == 0:
            return None
        return buffer[: match.end()]

    return chunker


def smooth_stream(
    *,
    delay_in_ms: Optional[float] = 10,
    chunking: Union[str, ChunkingFn] = "word",
) -> Transform:
    """Re-chunk streamed text into word/line (or custom) sized TextDelta parts,
    sleeping `delay_in_ms` between emitted chunks (AI SDK smoothStream).

    - `chunking`: "word" | "line" | a regex string | a callable
      `(buffer) -> matched-prefix or None`.
    - `delay_in_ms`: milliseconds to sleep between chunks (None = no sleep).

    Text is buffered per text-block id; non-text parts pass through immediately
    (flushing any buffered text for the relevant id first, e.g. on TextEnd).
    """
    chunker: ChunkingFn
    if callable(chunking):
        chunker = chunking
    elif chunking == "word":
        chunker = _word_chunker
    elif chunking == "line":
        chunker = _line_chunker
    elif isinstance(chunking, str):
        chunker = _regex_chunker(re.compile(chunking))
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported chunking: {chunking!r}")

    delay = None if delay_in_ms is None else delay_in_ms / 1000.0

    def transform(stream: AsyncIterator[TextStreamPart]) -> AsyncIterator[TextStreamPart]:
        async def gen() -> AsyncIterator[TextStreamPart]:
            buffers: dict[str, str] = {}
            first_emit = True

            async def emit_chunk(part: TextDelta) -> TextDelta:
                nonlocal first_emit
                if delay is not None and not first_emit:
                    await asyncio.sleep(delay)
                first_emit = False
                return part

            async for part in stream:
                if isinstance(part, TextStart):
                    buffers.setdefault(part.id, "")
                    yield part
                elif isinstance(part, TextDelta):
                    buf = buffers.get(part.id, "") + part.text
                    while True:
                        chunk = chunker(buf)
                        if not chunk:
                            break
                        buf = buf[len(chunk):]
                        yield await emit_chunk(
                            TextDelta(
                                id=part.id,
                                text=chunk,
                                provider_metadata=part.provider_metadata,
                            )
                        )
                    buffers[part.id] = buf
                elif isinstance(part, TextEnd):
                    buf = buffers.pop(part.id, "")
                    if buf:
                        yield await emit_chunk(TextDelta(id=part.id, text=buf))
                    yield part
                else:
                    yield part

        return gen()

    return transform
