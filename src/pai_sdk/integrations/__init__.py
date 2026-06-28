"""Optional integration helpers.

Core pai-sdk types stay provider- and vendor-neutral. Modules in this package
translate external observability/export formats into those core types without
making the external system part of the top-level SDK surface.
"""

from .otel import trace_from_otel_spans, trace_to_otel_spans

__all__ = ["trace_from_otel_spans", "trace_to_otel_spans"]
