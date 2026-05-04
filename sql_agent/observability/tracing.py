"""Optional OpenTelemetry tracing (phase 8.6).

When ``settings.otel_enabled`` is true, a global tracer is configured
with whichever exporter the environment dictates (standard ``OTEL_*``
env vars — ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
etc.). When false, ``get_tracer()`` returns a no-op tracer so call sites
can always use ``with tracer.start_as_current_span(...)`` without any
conditional logic.

Dependencies (``opentelemetry-api`` + ``opentelemetry-sdk``) are in the
``observability`` extra — installed with ``pip install -e .[observability]``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from sql_agent.config import get_logger, settings


_log = get_logger("observability.tracing")


_tracer: Optional[Any] = None
_lock = threading.Lock()


def _noop_tracer():
    """Return an OpenTelemetry no-op tracer that yields no-op spans."""
    try:
        from opentelemetry import trace

        return trace.NoOpTracer()
    except Exception:
        # Final fallback if OTel isn't importable at all.
        class _NoOpSpan:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def set_attribute(self, *_a, **_kw): pass
            def set_attributes(self, *_a, **_kw): pass
            def record_exception(self, *_a, **_kw): pass

        class _NoOpTracer:
            def start_as_current_span(self, *_a, **_kw): return _NoOpSpan()
            def start_span(self, *_a, **_kw): return _NoOpSpan()

        return _NoOpTracer()


def _build_tracer():
    if not settings.otel_enabled:
        return _noop_tracer()

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError as exc:
        _log.warning(
            "OTEL_ENABLED=true but opentelemetry not installed: %s. "
            "Install with: pip install -e .[observability]",
            exc,
        )
        return _noop_tracer()

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )

    # Prefer OTLP if the endpoint env var is set; fall back to console
    # exporter so spans are still visible locally.
    try:
        import os

        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter()
            _log.info("OTel OTLP exporter configured")
        else:
            exporter = ConsoleSpanExporter()
            _log.info("OTel ConsoleSpanExporter configured (set OTEL_EXPORTER_OTLP_ENDPOINT for OTLP)")
    except ImportError:
        exporter = ConsoleSpanExporter()
        _log.info("OTLP exporter not installed; using ConsoleSpanExporter")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


def get_tracer():
    """Return the process-wide OTel tracer (or a no-op if disabled)."""
    global _tracer
    if _tracer is not None:
        return _tracer
    with _lock:
        if _tracer is None:
            _tracer = _build_tracer()
    return _tracer


def reset_tracer() -> None:
    """Test hook — drop the cached tracer so changes to settings take effect."""
    global _tracer
    with _lock:
        _tracer = None
