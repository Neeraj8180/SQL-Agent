"""Prometheus metrics for the SQL Agent.

Design:
    * All metric instances are owned by a single ``Metrics`` object that
      holds its own ``CollectorRegistry``. That keeps tests isolated
      (each test can build a fresh ``Metrics()`` against a fresh registry
      without colliding with the default global registry).
    * Production code accesses the process-wide instance via ``get_metrics``;
      tests call ``reset_metrics()`` to drop the cached instance.
    * The HTTP exposer is booted lazily via ``start_metrics_server`` so it
      only runs when ``METRICS_ENABLED=true``.
    * ``prometheus_client`` is imported at module-load time (cheap; pure
      Python), but nothing in this module is called on the hot path unless
      metrics are enabled.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.exposition import start_http_server

from sql_agent.config import get_logger, settings


_log = get_logger("observability.metrics")


# Latency buckets chosen for an NL→SQL service: most requests complete in
# 50 ms – 5 s, with a heavy tail for local-model cold-start.
_LATENCY_BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


class Metrics:
    """Holds all Prometheus collectors for the service."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self._registry = registry or CollectorRegistry()

        self.rpc_total = Counter(
            "sql_agent_rpc_total",
            "Total gRPC requests.",
            labelnames=("rpc", "status"),
            registry=self._registry,
        )
        self.rpc_latency_ms = Histogram(
            "sql_agent_rpc_latency_ms",
            "End-to-end gRPC latency in milliseconds.",
            labelnames=("rpc",),
            buckets=_LATENCY_BUCKETS_MS,
            registry=self._registry,
        )
        self.tool_used_total = Counter(
            "sql_agent_tool_used_total",
            "Number of times each tool was selected.",
            labelnames=("tool",),
            registry=self._registry,
        )
        self.validation_errors_total = Counter(
            "sql_agent_validation_errors_total",
            "Total validation errors encountered across all RPCs.",
            registry=self._registry,
        )
        self.llm_provider_total = Counter(
            "sql_agent_llm_provider_total",
            "LLM provider decisions emitted by the router.",
            labelnames=("provider",),
            registry=self._registry,
        )
        self.memory_writes_total = Counter(
            "sql_agent_memory_writes_total",
            "FAISS memory store writes by kind.",
            labelnames=("kind",),  # "reward" | "penalty"
            registry=self._registry,
        )
        # Phase 8.6: per-provider call latency + token counters.
        self.llm_call_latency_ms = Histogram(
            "sql_agent_llm_call_latency_ms",
            "Latency of a single LLM call (structured or raw) by provider.",
            labelnames=("provider", "model"),
            buckets=_LATENCY_BUCKETS_MS,
            registry=self._registry,
        )
        self.llm_tokens_total = Counter(
            "sql_agent_llm_tokens_total",
            "LLM tokens consumed, labeled by provider and direction.",
            labelnames=("provider", "model", "direction"),  # input | output
            registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Convenience helpers (keep call sites short).
    # ------------------------------------------------------------------

    def record_rpc(self, rpc: str, status: str, latency_ms: float) -> None:
        try:
            self.rpc_total.labels(rpc=rpc, status=status).inc()
            self.rpc_latency_ms.labels(rpc=rpc).observe(max(float(latency_ms), 0.0))
        except Exception as exc:  # pragma: no cover
            _log.debug("record_rpc failed: %s", exc)

    def record_tool(self, tool: str) -> None:
        if not tool:
            return
        try:
            self.tool_used_total.labels(tool=tool).inc()
        except Exception as exc:  # pragma: no cover
            _log.debug("record_tool failed: %s", exc)

    def record_validation_errors(self, n: int) -> None:
        if not n:
            return
        try:
            self.validation_errors_total.inc(int(n))
        except Exception as exc:  # pragma: no cover
            _log.debug("record_validation_errors failed: %s", exc)

    def record_routing(self, provider: str) -> None:
        if not provider:
            return
        try:
            self.llm_provider_total.labels(provider=provider).inc()
        except Exception as exc:  # pragma: no cover
            _log.debug("record_routing failed: %s", exc)

    def record_memory_write(self, kind: str) -> None:
        try:
            self.memory_writes_total.labels(kind=kind).inc()
        except Exception as exc:  # pragma: no cover
            _log.debug("record_memory_write failed: %s", exc)

    # Phase 8.6 helpers.
    def record_llm_call(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        try:
            self.llm_call_latency_ms.labels(
                provider=provider or "unknown",
                model=(model or "unknown")[:100],
            ).observe(max(float(latency_ms), 0.0))
            if input_tokens:
                self.llm_tokens_total.labels(
                    provider=provider or "unknown",
                    model=(model or "unknown")[:100],
                    direction="input",
                ).inc(int(input_tokens))
            if output_tokens:
                self.llm_tokens_total.labels(
                    provider=provider or "unknown",
                    model=(model or "unknown")[:100],
                    direction="output",
                ).inc(int(output_tokens))
        except Exception as exc:  # pragma: no cover
            _log.debug("record_llm_call failed: %s", exc)

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_instance: Optional[Metrics] = None
_lock = threading.Lock()


def get_metrics() -> Metrics:
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = Metrics()
        return _instance


def reset_metrics(registry: Optional[CollectorRegistry] = None) -> Metrics:
    """Drop the cached Metrics singleton and rebuild.

    Tests should call this with a fresh ``CollectorRegistry()`` to avoid
    cross-test label leakage.
    """
    global _instance
    with _lock:
        _instance = Metrics(registry=registry)
        return _instance


# ---------------------------------------------------------------------------
# HTTP exposer
# ---------------------------------------------------------------------------


# Stores (http_server, server_thread) for stop_metrics_server.
_server_handle: Optional[Tuple[Any, Any]] = None
_server_lock = threading.Lock()


def start_metrics_server(
    port: Optional[int] = None,
    addr: Optional[str] = None,
    *,
    registry: Optional[CollectorRegistry] = None,
) -> Tuple[Any, Any]:
    """Start the Prometheus exposer on the given port.

    Idempotent: if the server is already running, returns the existing
    handle without rebinding the port. Returns ``(http_server, thread)``
    where the thread is a daemon so it dies with the main process.
    """
    global _server_handle
    with _server_lock:
        if _server_handle is not None:
            _log.info("metrics server already running; skipping start")
            return _server_handle

        p = int(port if port is not None else settings.metrics_port)
        a = str(addr if addr is not None else settings.metrics_addr) or ""
        reg = registry or get_metrics().registry

        server, thread = start_http_server(p, addr=a, registry=reg)
        _server_handle = (server, thread)
        _log.info("Prometheus metrics server started on %s:%d", a or "0.0.0.0", p)
        return _server_handle


def stop_metrics_server() -> None:
    """Shut down the Prometheus exposer if it's running."""
    global _server_handle
    with _server_lock:
        if _server_handle is None:
            return
        server, _thread = _server_handle
        try:
            server.shutdown()
            try:
                server.server_close()
            except Exception:  # pragma: no cover
                pass
        except Exception as exc:  # pragma: no cover
            _log.warning("metrics server shutdown failed: %s", exc)
        _server_handle = None
        _log.info("Prometheus metrics server stopped")
