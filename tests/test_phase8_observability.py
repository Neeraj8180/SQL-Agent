"""Phase 8.6 tests — per-provider metrics, log rotation, OTel spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from sql_agent.config import settings
from sql_agent.llm_serving import registry as llm_registry
from sql_agent.observability.metrics import Metrics, reset_metrics
from sql_agent.observability.rotating_jsonl import RotatingJsonlWriter


# ---------------------------------------------------------------------------
# Per-provider latency + token metrics
# ---------------------------------------------------------------------------


def _scrape(registry: CollectorRegistry) -> str:
    from prometheus_client import generate_latest

    return generate_latest(registry).decode("utf-8")


def test_record_llm_call_populates_per_provider_histogram_and_counters():
    reg = CollectorRegistry()
    m = reset_metrics(registry=reg)
    m.record_llm_call("openai", "gpt-4o-mini", 123.4, input_tokens=50, output_tokens=20)
    m.record_llm_call("hf", "Qwen2.5-1.5B", 890.0, input_tokens=120, output_tokens=60)

    text = _scrape(reg)
    assert 'sql_agent_llm_call_latency_ms_count{model="gpt-4o-mini",provider="openai"} 1.0' in text
    assert 'sql_agent_llm_call_latency_ms_count{model="Qwen2.5-1.5B",provider="hf"} 1.0' in text
    assert 'sql_agent_llm_tokens_total{direction="input",model="gpt-4o-mini",provider="openai"} 50.0' in text
    assert 'sql_agent_llm_tokens_total{direction="output",model="gpt-4o-mini",provider="openai"} 20.0' in text
    assert 'sql_agent_llm_tokens_total{direction="input",model="Qwen2.5-1.5B",provider="hf"} 120.0' in text

    reset_metrics()


def test_record_llm_call_silent_on_zero_tokens():
    reg = CollectorRegistry()
    m = reset_metrics(registry=reg)
    m.record_llm_call("openai", "gpt-4o-mini", 50.0)  # no tokens

    text = _scrape(reg)
    # Histogram IS observed (even with zero tokens).
    assert 'sql_agent_llm_call_latency_ms_count{model="gpt-4o-mini",provider="openai"} 1.0' in text
    # But the token counter for this (provider, direction) isn't emitted
    # with a 0 value — Prometheus doesn't materialize untouched label sets.
    assert 'sql_agent_llm_tokens_total{direction="input",model="gpt-4o-mini"' not in text
    reset_metrics()


# ---------------------------------------------------------------------------
# Rotating JSONL writer
# ---------------------------------------------------------------------------


def test_rotating_jsonl_rotates_when_over_max_bytes(tmp_path):
    path = tmp_path / "log.jsonl"
    w = RotatingJsonlWriter(path, max_bytes=200, backup_count=3)

    # Each line is ~60 bytes; after 5 lines we've exceeded 200 bytes and
    # at least one rotation should have happened.
    for i in range(8):
        w.append_line(json.dumps({"seq": i, "payload": "x" * 40}))

    assert path.exists()
    # Backups 1..3 exist (we wrote 8 lines × ~60 bytes = ~480 bytes > 200).
    backups = sorted(tmp_path.glob("log.jsonl.*"))
    assert 1 <= len(backups) <= 3, f"expected 1-3 backups, got {len(backups)}"

    # Concatenating current + backups should yield all 8 lines in some order.
    all_lines = []
    for p in [path] + backups:
        all_lines.extend(p.read_text(encoding="utf-8").splitlines())
    assert len(all_lines) == 8
    seqs = sorted(int(json.loads(l)["seq"]) for l in all_lines)
    assert seqs == list(range(8))


def test_rotating_jsonl_respects_backup_count(tmp_path):
    path = tmp_path / "log.jsonl"
    w = RotatingJsonlWriter(path, max_bytes=50, backup_count=2)

    for i in range(30):
        w.append_line(json.dumps({"seq": i, "payload": "x" * 40}))

    # After many rotations, we must never have more than backup_count backups.
    backups = list(tmp_path.glob("log.jsonl.*"))
    assert len(backups) <= 2


def test_rotating_jsonl_disabled_when_max_bytes_zero(tmp_path):
    path = tmp_path / "log.jsonl"
    w = RotatingJsonlWriter(path, max_bytes=0, backup_count=5)

    for i in range(20):
        w.append_line(json.dumps({"seq": i}))

    assert path.exists()
    assert not list(tmp_path.glob("log.jsonl.*"))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 20


def test_decision_log_writer_uses_rotation(tmp_path):
    """Phase 8.6: DecisionLogWriter is backed by RotatingJsonlWriter."""
    from sql_agent.routing.base import RoutingDecision
    from sql_agent.routing.decision_log import DecisionLogWriter

    path = tmp_path / "decisions.jsonl"

    orig_max = settings.log_rotation_max_bytes
    orig_bk = settings.log_rotation_backup_count
    try:
        settings.log_rotation_max_bytes = 200
        settings.log_rotation_backup_count = 2
        w = DecisionLogWriter(path=path)
        for i in range(20):
            w.append(
                RoutingDecision.now(
                    session_id=f"s{i}",
                    provider="mock",
                    strategy="weighted_random",
                )
            )
        backups = list(tmp_path.glob("decisions.jsonl.*"))
        assert len(backups) <= 2
    finally:
        settings.log_rotation_max_bytes = orig_max
        settings.log_rotation_backup_count = orig_bk


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------


def test_otel_disabled_returns_noop_tracer():
    from sql_agent.observability.tracing import get_tracer, reset_tracer

    orig = settings.otel_enabled
    try:
        settings.otel_enabled = False
        reset_tracer()
        t = get_tracer()
        # NoOpTracer / fallback _NoOpTracer both expose start_as_current_span.
        with t.start_as_current_span("noop") as span:
            span.set_attribute("k", "v")  # must not raise
    finally:
        settings.otel_enabled = orig
        reset_tracer()


def test_otel_enabled_builds_real_tracer_with_console_exporter():
    from opentelemetry import trace

    from sql_agent.observability.tracing import get_tracer, reset_tracer

    orig = settings.otel_enabled
    try:
        settings.otel_enabled = True
        reset_tracer()
        t = get_tracer()
        # Should produce a span span-object with record_exception + attributes.
        with t.start_as_current_span("phase8-test") as span:
            span.set_attribute("k", "v")
            try:
                raise RuntimeError("trace-me")
            except RuntimeError as exc:
                span.record_exception(exc)

        # Deterministically flush + shut down the span exporter so the
        # background BatchSpanProcessor doesn't race with pytest teardown
        # (which closes stdout; the exporter would otherwise log a noisy
        # "I/O on closed file" ValueError after the test has passed).
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    finally:
        settings.otel_enabled = orig
        reset_tracer()


# ---------------------------------------------------------------------------
# Servicer-level: /metrics exposes per-provider counters after an RPC
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env():
    from sql_agent.services.memory_manager import reset_memory_manager

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_registry.reset_caches()
        reset_memory_manager()
        reg = CollectorRegistry()
        m = reset_metrics(registry=reg)
        yield m, reg
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_registry.reset_caches()
        reset_memory_manager()
        reset_metrics()


def test_openai_provider_wrapper_records_latency_and_tokens(mock_env):
    """Phase 8.6: the ChatOpenAI proxy emits latency + token metrics
    transparently. We feed a synthetic AIMessage to avoid a real API call."""
    from langchain_core.messages import AIMessage

    from sql_agent.llm_serving.openai_provider import _ChatOpenAIProxy
    from sql_agent.request_context import token_usage_scope

    class _FakeOpenAI:
        # Minimal stand-in: invoke returns an AIMessage with usage_metadata.
        def invoke(self, _messages):
            return AIMessage(
                content="hi",
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            )

    proxy = _ChatOpenAIProxy(_FakeOpenAI(), "gpt-test")

    m, reg = mock_env
    with token_usage_scope():
        proxy.invoke([])
    text = _scrape(reg)
    assert 'sql_agent_llm_call_latency_ms_count{model="gpt-test",provider="openai"} 1.0' in text
    assert 'sql_agent_llm_tokens_total{direction="input",model="gpt-test",provider="openai"} 11.0' in text
    assert 'sql_agent_llm_tokens_total{direction="output",model="gpt-test",provider="openai"} 7.0' in text
