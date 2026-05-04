"""Phase 5 tests — observability (structured logs + Prometheus).

Three clusters:
    1. Structured logging: ContextFilter + JsonFormatter
    2. Metrics: counters, histogram buckets, router + servicer integration
    3. HTTP exposer: start_metrics_server serves /metrics content
"""

from __future__ import annotations

import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

import pytest
from prometheus_client import CollectorRegistry

from sql_agent.config import settings
from sql_agent.observability.metrics import (
    Metrics,
    get_metrics,
    reset_metrics,
    start_metrics_server,
    stop_metrics_server,
)
from sql_agent.observability.structured_logging import (
    ContextFilter,
    JsonFormatter,
)
from sql_agent.request_context import (
    request_id_var,
    request_scope,
    session_id_var,
    session_scope,
)


# ---------------------------------------------------------------------------
# 1. Structured logging
# ---------------------------------------------------------------------------


def _capture_logger(name: str, formatter: logging.Formatter, context_filter: bool):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(formatter)
    if context_filter:
        handler.addFilter(ContextFilter())
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, buf


def test_json_formatter_emits_single_line_json():
    logger, buf = _capture_logger("test.json.format", JsonFormatter(), True)
    logger.info("hello world")
    line = buf.getvalue().strip()
    rec = json.loads(line)
    assert rec["message"] == "hello world"
    assert rec["level"] == "INFO"
    assert rec["logger"] == "test.json.format"
    assert "timestamp" in rec
    # No contextvars set => empty strings.
    assert rec["session_id"] == ""
    assert rec["request_id"] == ""


def test_json_formatter_includes_contextvars():
    logger, buf = _capture_logger("test.json.ctx", JsonFormatter(), True)
    with session_scope("sess-42"), request_scope("req-abc") as rid:
        logger.info("inside scope")
        assert rid == "req-abc"
    rec = json.loads(buf.getvalue().strip())
    assert rec["session_id"] == "sess-42"
    assert rec["request_id"] == "req-abc"


def test_json_formatter_promotes_extras():
    logger, buf = _capture_logger("test.json.extras", JsonFormatter(), True)
    logger.info(
        "tool call",
        extra={"tool_used": "count", "latency_ms": 12.5, "row_count": 42},
    )
    rec = json.loads(buf.getvalue().strip())
    assert rec["tool_used"] == "count"
    assert rec["latency_ms"] == 12.5
    assert rec["row_count"] == 42


def test_json_formatter_includes_exception_info():
    logger, buf = _capture_logger("test.json.exc", JsonFormatter(), True)
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("caught")
    rec = json.loads(buf.getvalue().strip())
    assert "exc_info" in rec
    assert "ValueError: boom" in rec["exc_info"]


def test_context_filter_isolates_across_threads():
    """Each thread's log line carries its own contextvar values."""
    logger, buf = _capture_logger("test.json.threads", JsonFormatter(), True)

    def worker(i: int) -> None:
        with session_scope(f"s{i}"), request_scope(f"r{i}"):
            logger.info("thread log")

    with ThreadPoolExecutor(max_workers=6) as exe:
        futs = [exe.submit(worker, i) for i in range(6)]
        for f in as_completed(futs):
            f.result()

    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert len(lines) == 6
    sids = sorted(l["session_id"] for l in lines)
    rids = sorted(l["request_id"] for l in lines)
    assert sids == [f"s{i}" for i in range(6)]
    assert rids == [f"r{i}" for i in range(6)]


def test_request_scope_generates_uuid_when_none_given():
    with request_scope() as rid:
        assert rid
        assert request_id_var.get() == rid
    assert request_id_var.get() is None


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_metrics():
    """A Metrics bound to a brand-new CollectorRegistry for one test."""
    reg = CollectorRegistry()
    m = reset_metrics(registry=reg)
    yield m, reg
    reset_metrics()  # restore a fresh default


def _scrape(registry: CollectorRegistry) -> str:
    from prometheus_client import generate_latest

    return generate_latest(registry).decode("utf-8")


def test_metrics_rpc_counter_and_histogram(fresh_metrics):
    m, reg = fresh_metrics
    m.record_rpc("ExecuteSQL", "success", 35.0)
    m.record_rpc("ExecuteSQL", "success", 150.0)
    m.record_rpc("ExecuteSQL", "error", 20.0)

    text = _scrape(reg)
    # Counter
    assert 'sql_agent_rpc_total{rpc="ExecuteSQL",status="success"} 2.0' in text
    assert 'sql_agent_rpc_total{rpc="ExecuteSQL",status="error"} 1.0' in text
    # Histogram: the 150ms observation lands in the le=250 bucket.
    assert 'sql_agent_rpc_latency_ms_count{rpc="ExecuteSQL"} 3.0' in text
    assert 'le="250.0"' in text


def test_metrics_tool_and_validation_counters(fresh_metrics):
    m, reg = fresh_metrics
    m.record_tool("count")
    m.record_tool("count")
    m.record_tool("data_fetch")
    m.record_validation_errors(3)

    text = _scrape(reg)
    assert 'sql_agent_tool_used_total{tool="count"} 2.0' in text
    assert 'sql_agent_tool_used_total{tool="data_fetch"} 1.0' in text
    assert "sql_agent_validation_errors_total 3.0" in text


def test_metrics_routing_counter_from_router(fresh_metrics):
    """When routing is enabled, router.route() bumps the provider counter."""
    _m, reg = fresh_metrics
    from sql_agent.routing.router import LLMRouter
    from sql_agent.routing.weighted import WeightedRandomStrategy

    r = LLMRouter(
        strategy=WeightedRandomStrategy({"mock": 100}),
        decision_log=None,
        enable_log=False,
    )
    for _ in range(5):
        assert r.route(session_id=None) == "mock"

    text = _scrape(reg)
    assert 'sql_agent_llm_provider_total{provider="mock"} 5.0' in text


def test_metrics_record_helpers_are_silent_on_empty_values(fresh_metrics):
    m, reg = fresh_metrics
    m.record_tool("")  # should silently skip
    m.record_routing("")  # should silently skip
    m.record_validation_errors(0)  # should silently skip
    text = _scrape(reg)
    # No labels registered, so raw lines are present only as empty series
    # (the counter exists but has no label combinations written).
    assert "sql_agent_tool_used_total" in text
    assert "sql_agent_llm_provider_total" in text


# ---------------------------------------------------------------------------
# 3. HTTP exposer
# ---------------------------------------------------------------------------


def test_start_metrics_server_exposes_metrics_over_http():
    """Boot the exposer on an ephemeral port and scrape it via HTTP."""
    reg = CollectorRegistry()
    m = reset_metrics(registry=reg)
    m.record_rpc("ExecuteSQL", "success", 42.0)
    m.record_tool("count")

    server, _thread = start_metrics_server(port=0, addr="127.0.0.1", registry=reg)
    try:
        # port=0 tells the OS to pick a free port; the server exposes it.
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/metrics"
        with urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert 'sql_agent_rpc_total{rpc="ExecuteSQL",status="success"} 1.0' in body
        assert 'sql_agent_tool_used_total{tool="count"} 1.0' in body
    finally:
        stop_metrics_server()
        reset_metrics()


def test_start_metrics_server_is_idempotent():
    reg = CollectorRegistry()
    reset_metrics(registry=reg)
    h1 = start_metrics_server(port=0, addr="127.0.0.1", registry=reg)
    try:
        h2 = start_metrics_server(port=0, addr="127.0.0.1", registry=reg)
        # Second call should return the same handle, not bind a new port.
        assert h1 is h2
    finally:
        stop_metrics_server()
        reset_metrics()


# ---------------------------------------------------------------------------
# 4. End-to-end: servicer records metrics + log fields
# ---------------------------------------------------------------------------


@pytest.fixture
def observability_env(tmp_path, monkeypatch):
    """Switch providers to mock, install a fresh Metrics, and capture logs."""
    from sql_agent.llm_serving import registry as llm_reg
    from sql_agent.services.memory_manager import reset_memory_manager
    from sql_agent.tracking.registry import reset_tracker

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_reg.reset_caches()
        reset_memory_manager()
        reset_tracker()
        reg = CollectorRegistry()
        m = reset_metrics(registry=reg)
        yield m, reg
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_reg.reset_caches()
        reset_memory_manager()
        reset_tracker()
        reset_metrics()


def test_servicer_emits_rpc_and_tool_metrics_end_to_end(observability_env):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server.server import SqlAgentServicer

    class _Ctx:
        def abort(self, code, details):
            raise AssertionError(f"abort: {code} {details}")

        def set_code(self, _): pass

        def set_details(self, _): pass

    m, reg = observability_env
    servicer = SqlAgentServicer()
    for i in range(3):
        resp = servicer.ExecuteSQL(
            pb2.ExecuteSQLRequest(
                query="count orders",
                session_id=f"obs-{i}",
            ),
            _Ctx(),
        )
        assert resp.success

    text = _scrape(reg)
    assert 'sql_agent_rpc_total{rpc="ExecuteSQL",status="success"} 3.0' in text
    assert 'sql_agent_tool_used_total{tool="count"} 3.0' in text
    assert 'sql_agent_rpc_latency_ms_count{rpc="ExecuteSQL"} 3.0' in text
