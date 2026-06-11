"""Error types, mirroring the AI SDK's error hierarchy."""

from __future__ import annotations

from typing import Any, Optional


class AISDKError(Exception):
    """Base error for this library."""


class APICallError(AISDKError):
    """A provider API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Any = None,
        is_retryable: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.is_retryable = is_retryable
        self.cause = cause


class NoSuchToolError(AISDKError):
    """The model called a tool that is not defined."""

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        super().__init__(
            f"Model tried to call unavailable tool '{tool_name}'. "
            f"Available tools: {', '.join(available_tools) or '(none)'}."
        )
        self.tool_name = tool_name
        self.available_tools = available_tools


class InvalidToolInputError(AISDKError):
    """The model produced input that failed the tool's input schema."""

    def __init__(self, tool_name: str, tool_input: Any, cause: BaseException) -> None:
        super().__init__(
            f"Invalid input for tool '{tool_name}': {cause}"
        )
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.cause = cause


class InvalidPromptError(AISDKError):
    """The prompt/messages/system combination is invalid."""


class NoSuchProviderError(AISDKError):
    """A 'provider/model' string referenced an unknown provider."""


class NoObjectGeneratedError(AISDKError):
    """The model output could not be parsed/validated into an object."""

    def __init__(
        self,
        message: str = "No object generated.",
        *,
        text: Optional[str] = None,
        cause: Optional[BaseException] = None,
        finish_reason: Optional[str] = None,
        usage: Any = None,
    ) -> None:
        super().__init__(message)
        self.text = text
        self.cause = cause
        self.finish_reason = finish_reason
        self.usage = usage


class AbortError(AISDKError):
    """The run was aborted via an abort_signal / StreamTextResult.abort()."""

    def __init__(self, message: str = "The operation was aborted.", *, reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason


class GenerationTimeoutError(AISDKError):
    """A generation timeout budget expired (total or per-step).

    `budget` is "total" or "step" indicating which deadline fired.
    """

    def __init__(self, budget: str, timeout_ms: Optional[float] = None) -> None:
        suffix = f" after {timeout_ms:.0f}ms" if timeout_ms is not None else ""
        super().__init__(f"Generation timed out ({budget} budget){suffix}.")
        self.budget = budget
        self.timeout_ms = timeout_ms


class MissingDependencyError(AISDKError):
    """An optional provider SDK is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"The '{package}' package is required for this provider. "
            f"Install it with: pip install 'pai-sdk[{extra}]'"
        )
        self.package = package
        self.extra = extra
