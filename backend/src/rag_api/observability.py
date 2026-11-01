"""Focused Phoenix tracing for RAG runs, tools, and model calls."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import get_attributes_from_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

TRACER = trace.get_tracer("rag_api")


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def set_value(span: trace.Span, prefix: str, value: Any) -> None:
    if isinstance(value, str):
        span.set_attribute(f"{prefix}.value", value)
        span.set_attribute(f"{prefix}.mime_type", "text/plain")
    else:
        span.set_attribute(f"{prefix}.value", as_json(value))
        span.set_attribute(f"{prefix}.mime_type", "application/json")


@contextmanager
def observed(name: str, kind: str, request: Any) -> Iterator[trace.Span]:
    """Record a focused operation with inspectable input and explicit status."""
    with TRACER.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", kind)
        for key, value in get_attributes_from_context():
            span.set_attribute(key, value)
        if kind == "TOOL":
            span.set_attribute("tool.name", name.removeprefix("tool."))
        set_value(span, "input", request)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:500]))
            raise


def record_output(span: trace.Span, response: Any) -> None:
    set_value(span, "output", response)
