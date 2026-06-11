"""Amazon Bedrock provider — Anthropic models served through AWS Bedrock.

Mirrors @ai-sdk/amazon-bedrock's Anthropic surface: the request/response
mapping is identical to the direct Anthropic Messages API (it reuses
AnthropicLanguageModel wholesale), only the SDK client differs — here the
anthropic SDK's AsyncAnthropicBedrock, which signs requests with AWS
credentials and routes to the Bedrock runtime endpoint.

Bedrock model ids carry an "anthropic." prefix (e.g.
"anthropic.claude-opus-4-8") and are passed through verbatim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..errors import MissingDependencyError
from .anthropic import AnthropicLanguageModel


@dataclass
class BedrockAnthropicLanguageModel(AnthropicLanguageModel):
    """Anthropic-on-Bedrock model. Same wire mapping as AnthropicLanguageModel,
    but the client is anthropic.AsyncAnthropicBedrock signed with AWS creds."""

    provider: str = "bedrock.anthropic"
    aws_region: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic[bedrock]", "bedrock") from exc
        try:
            bedrock_cls = anthropic.AsyncAnthropicBedrock
        except AttributeError as exc:  # boto3/awscrt extras missing
            raise MissingDependencyError("anthropic[bedrock]", "bedrock") from exc

        region = self.aws_region or os.environ.get("AWS_REGION")
        kwargs: dict[str, Any] = {"max_retries": 0}
        if region:
            kwargs["aws_region"] = region
        if self.aws_access_key is not None:
            kwargs["aws_access_key"] = self.aws_access_key
        if self.aws_secret_key is not None:
            kwargs["aws_secret_key"] = self.aws_secret_key
        if self.aws_session_token is not None:
            kwargs["aws_session_token"] = self.aws_session_token
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client_cache = bedrock_cls(**kwargs)
        return self._client_cache
