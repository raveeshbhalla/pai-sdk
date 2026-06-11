"""Google Vertex AI provider — two surfaces, like @ai-sdk/google-vertex.

- vertex(model_id)            -> Gemini on Vertex (google-genai with
                                 vertexai=True). provider "google.vertex".
- vertex.anthropic(model_id)  -> Anthropic (Claude) on Vertex, via the
                                 anthropic SDK's AsyncAnthropicVertex.
                                 provider "vertex.anthropic".

Project comes from GOOGLE_CLOUD_PROJECT and location from
GOOGLE_CLOUD_LOCATION (default "us-central1") unless passed explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..errors import MissingDependencyError
from .anthropic import AnthropicLanguageModel
from .google import GoogleLanguageModel

_DEFAULT_LOCATION = "us-central1"


@dataclass
class VertexGoogleLanguageModel(GoogleLanguageModel):
    """Gemini on Vertex. Same mapping as GoogleLanguageModel, but the genai
    client is built with vertexai=True + project/location."""

    provider: str = "google.vertex"
    project: Optional[str] = None
    location: Optional[str] = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            from google import genai
        except ImportError as exc:
            raise MissingDependencyError("google-genai", "vertex") from exc
        kwargs: dict[str, Any] = {"vertexai": True}
        project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = (
            self.location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or _DEFAULT_LOCATION
        )
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location
        self._client_cache = genai.Client(**kwargs)
        return self._client_cache


@dataclass
class VertexAnthropicLanguageModel(AnthropicLanguageModel):
    """Anthropic (Claude) on Vertex. Same wire mapping as
    AnthropicLanguageModel, but the client is anthropic.AsyncAnthropicVertex."""

    provider: str = "vertex.anthropic"
    project: Optional[str] = None
    location: Optional[str] = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic[vertex]", "vertex") from exc
        try:
            vertex_cls = anthropic.AsyncAnthropicVertex
        except AttributeError as exc:  # google-auth extra missing
            raise MissingDependencyError("anthropic[vertex]", "vertex") from exc

        project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = (
            self.location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or _DEFAULT_LOCATION
        )
        kwargs: dict[str, Any] = {"max_retries": 0}
        if project:
            kwargs["project_id"] = project
        if location:
            kwargs["region"] = location
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client_cache = vertex_cls(**kwargs)
        return self._client_cache
