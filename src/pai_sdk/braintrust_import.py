"""Compatibility wrapper for Braintrust trace import helpers.

Prefer importing from `pai_sdk.integrations.braintrust`. This module remains so
older code that imports `pai_sdk.braintrust_import` continues to work.
"""

from .integrations.braintrust import (
    braintrust_message_to_model_message,
    braintrust_messages_to_model_messages,
    span_from_braintrust_row,
    trace_from_braintrust_rows,
)

__all__ = [
    "braintrust_message_to_model_message",
    "braintrust_messages_to_model_messages",
    "span_from_braintrust_row",
    "trace_from_braintrust_rows",
]
