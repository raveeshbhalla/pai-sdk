"""Shared helpers for provider adapters."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Optional, Union

from ..errors import APICallError
from ..messages import DataContent

_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;,]+)?(;base64)?,(?P<data>.*)$", re.DOTALL)

_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
    (b"%PDF", "application/pdf"),
]


def is_url(value: DataContent) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def split_data_content(
    value: DataContent, media_type: Optional[str] = None
) -> tuple[str, str, Optional[str]]:
    """Normalize DataContent into (kind, payload, media_type).

    kind is "url" (payload = http(s) URL) or "base64" (payload = base64 str).
    """
    if isinstance(value, bytes):
        return "base64", base64.b64encode(value).decode(), media_type or detect_media_type(value)
    if is_url(value):
        return "url", value, media_type
    match = _DATA_URL_RE.match(value)
    if match:
        return "base64", match.group("data"), media_type or match.group("media")
    return "base64", value, media_type  # assume plain base64


def to_bytes(value: DataContent) -> bytes:
    """Decode DataContent (bytes / base64 / data: URL) to raw bytes."""
    if isinstance(value, bytes):
        return value
    match = _DATA_URL_RE.match(value)
    payload = match.group("data") if match else value
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Could not decode data content as base64: {exc}") from exc


def detect_media_type(data: bytes) -> Optional[str]:
    for magic, media in _MAGIC:
        if data.startswith(magic):
            return media
    return None


def as_data_url(value: DataContent, media_type: Optional[str]) -> str:
    """Render DataContent as an http(s) URL or data: URL."""
    kind, payload, detected = split_data_content(value, media_type)
    if kind == "url":
        return payload
    return f"data:{detected or 'application/octet-stream'};base64,{payload}"


def wrap_provider_error(exc: Exception, provider: str) -> APICallError:
    """Convert a provider SDK exception into APICallError, preserving
    retryability based on HTTP status."""
    status: Optional[int] = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None) if isinstance(getattr(exc, "code", None), int) else None
    body = getattr(exc, "body", None) or getattr(exc, "response_json", None)
    retryable = status in (408, 409, 429) or (status is not None and status >= 500)
    # Connection-level errors (no status) are retryable too.
    if status is None and exc.__class__.__name__ in (
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
    ):
        retryable = True
    return APICallError(
        f"{provider} API call failed: {exc}",
        status_code=status,
        response_body=body,
        is_retryable=retryable,
        cause=exc,
    )


def merge_provider_options(
    request: dict, options: dict[str, dict], *keys: str
) -> None:
    """Merge providerOptions for the given provider key(s) into the request."""
    for key in keys:
        for name, value in (options.get(key) or {}).items():
            request.setdefault(name, value)


def file_id_value(file_id: object) -> str:
    """Extract a plain string file id from a FileIdData.id value.

    The id may be a plain string or a small mapping (e.g. {"file_id": "..."});
    for a mapping, prefer the "file_id" key, else fall back to the sole value.
    """
    fid = getattr(file_id, "id", file_id)
    if isinstance(fid, str):
        return fid
    if isinstance(fid, dict):
        if "file_id" in fid:
            return str(fid["file_id"])
        values = list(fid.values())
        if len(values) == 1:
            return str(values[0])
        raise ValueError(f"Ambiguous file id mapping: {fid!r}")
    return str(fid)


def raw_event_value(event: object) -> object:
    """Best-effort JSON-able representation of a provider SSE event for RawPart."""
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(event, dict):
        return event
    return event


def request_echo(request: dict) -> dict:
    """A JSON-able copy of a request body for ProviderResult.request, minus
    transport-only keys (headers)."""
    return {k: v for k, v in request.items() if k not in ("extra_headers",)}


def system_and_rest(messages: list) -> tuple[list[str], list]:
    """Split out system message texts from the rest of the prompt."""
    system_texts: list[str] = []
    rest = []
    for message in messages:
        if message.role == "system":
            system_texts.append(message.content)
        else:
            rest.append(message)
    return system_texts, rest


Stringable = Union[str, bytes]
