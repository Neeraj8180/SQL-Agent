"""Observability (phase 5): structured logging + Prometheus metrics.

Public names resolved lazily so importing this package does NOT import
``prometheus_client`` unless metrics are requested.

    * ``ContextFilter``     - logging filter that injects session/request ids
    * ``JsonFormatter``     - one-JSON-per-line formatter
    * ``apply_json_logging``- install the formatter on the root logger
    * ``get_metrics``       - process-wide Metrics singleton
    * ``reset_metrics``     - test hook
    * ``start_metrics_server``/``stop_metrics_server`` - Prometheus HTTP exposer
"""

from __future__ import annotations

__all__ = [
    "ContextFilter",
    "JsonFormatter",
    "apply_json_logging",
    "get_metrics",
    "reset_metrics",
    "start_metrics_server",
    "stop_metrics_server",
    "get_tracer",
    "reset_tracer",
]


def __getattr__(name: str):
    if name in {"ContextFilter", "JsonFormatter", "apply_json_logging"}:
        from . import structured_logging

        return getattr(structured_logging, name)
    if name in {
        "get_metrics",
        "reset_metrics",
        "start_metrics_server",
        "stop_metrics_server",
    }:
        from . import metrics

        return getattr(metrics, name)
    if name in {"get_tracer", "reset_tracer"}:
        from . import tracing

        return getattr(tracing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
