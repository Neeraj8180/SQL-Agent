"""Phase 3 tests — routing / A/B testing.

Coverage:
    1. WeightedRandomStrategy construction / validation / env-string parsing
    2. Distribution correctness (seeded RNG, deterministic count)
    3. RoutingDecision JSONL serialization
    4. DecisionLogWriter atomic append (thread-safe spot check)
    5. LLMRouter: strategy failures fall back to default provider
    6. services.llm.get_chat_model consults router iff LLM_ROUTING_ENABLED
    7. session_id_var is thread-isolated (ThreadPoolExecutor test)
    8. End-to-end: servicer writes one decision log line per RPC
"""

from __future__ import annotations

import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry
from sql_agent.request_context import session_id_var, session_scope
from sql_agent.routing.base import RoutingDecision
from sql_agent.routing.decision_log import DecisionLogWriter
from sql_agent.routing.router import LLMRouter, get_router, reset_router
from sql_agent.routing.weighted import WeightedRandomStrategy


# ---------------------------------------------------------------------------
# 1. Weighted strategy validation
# ---------------------------------------------------------------------------


def test_weighted_empty_weights_rejected():
    with pytest.raises(ValueError, match="at least one weight"):
        WeightedRandomStrategy({})


def test_weighted_negative_weight_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        WeightedRandomStrategy({"a": -1})


def test_weighted_zero_sum_rejected():
    with pytest.raises(ValueError, match="Sum of weights"):
        WeightedRandomStrategy({"a": 0, "b": 0})


def test_weighted_float_weight_rejected():
    with pytest.raises(TypeError, match="must be int"):
        WeightedRandomStrategy({"a": 1.5})  # type: ignore[dict-item]


def test_weighted_from_env_string_rejects_malformed_chunk():
    with pytest.raises(ValueError, match="provider:weight"):
        WeightedRandomStrategy.from_env_string("no-colon")


def test_weighted_from_env_string_rejects_non_integer_weight():
    with pytest.raises(ValueError, match="Invalid weight"):
        WeightedRandomStrategy.from_env_string("a:abc")


def test_weighted_from_env_string_empty_name():
    with pytest.raises(ValueError, match="Empty provider name"):
        WeightedRandomStrategy.from_env_string(":50")


def test_weighted_from_env_string_trims_whitespace():
    s = WeightedRandomStrategy.from_env_string(" a : 70 , b : 30 ")
    assert s.weights == {"a": 70, "b": 30}


def test_weighted_from_env_string_accumulates_duplicates():
    s = WeightedRandomStrategy.from_env_string("a:30, a:40, b:30")
    assert s.weights == {"a": 70, "b": 30}


# ---------------------------------------------------------------------------
# 2. Distribution (seeded RNG for reproducibility)
# ---------------------------------------------------------------------------


def test_weighted_distribution_matches_target_ratio_within_tolerance():
    s = WeightedRandomStrategy({"a": 70, "b": 30}, rng=random.Random(42))
    picks = Counter(s.choose() for _ in range(10_000))
    a_ratio = picks["a"] / 10_000
    # ±3% tolerance is generous for N=10K; the seeded RNG makes this stable.
    assert 0.67 <= a_ratio <= 0.73, f"expected ~0.7, got {a_ratio}"
    assert picks["a"] + picks["b"] == 10_000


def test_weighted_single_provider_always_returns_it():
    s = WeightedRandomStrategy({"only": 100}, rng=random.Random(0))
    for _ in range(50):
        assert s.choose() == "only"


# ---------------------------------------------------------------------------
# 3. RoutingDecision serialization
# ---------------------------------------------------------------------------


def test_routing_decision_round_trips_as_jsonl():
    d = RoutingDecision.now(
        session_id="sess-1",
        provider="openai",
        strategy="weighted_random",
        weights={"openai": 70, "hf": 30},
    )
    line = d.to_json_line()
    parsed = json.loads(line)
    assert parsed["session_id"] == "sess-1"
    assert parsed["provider"] == "openai"
    assert parsed["strategy"] == "weighted_random"
    assert parsed["weights"] == {"openai": 70, "hf": 30}
    # ts is an ISO timestamp.
    assert "T" in parsed["ts"] and parsed["ts"].endswith("+00:00")


# ---------------------------------------------------------------------------
# 4. DecisionLogWriter (JSONL + thread-safety)
# ---------------------------------------------------------------------------


def test_decision_log_writes_one_line_per_append(tmp_path):
    log_path = tmp_path / "nested" / "decisions.jsonl"
    w = DecisionLogWriter(path=log_path)
    for i in range(5):
        w.append(
            RoutingDecision.now(
                session_id=f"s{i}",
                provider="mock",
                strategy="weighted_random",
                weights={"mock": 100},
            )
        )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    sessions = [json.loads(l)["session_id"] for l in lines]
    assert sessions == ["s0", "s1", "s2", "s3", "s4"]


def test_decision_log_thread_safe_across_many_appenders(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    w = DecisionLogWriter(path=log_path)

    def worker(i: int) -> None:
        for j in range(20):
            w.append(
                RoutingDecision.now(
                    session_id=f"t{i}-{j}",
                    provider="mock",
                    strategy="weighted_random",
                )
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8 * 20
    # Every line must be independently parseable (no interleaved writes).
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# 5. LLMRouter fallbacks
# ---------------------------------------------------------------------------


class _ExplodingStrategy:
    name = "exploding"
    weights: dict = {}

    def choose(self, session_id=None):  # pragma: no cover
        raise RuntimeError("simulated strategy failure")


def test_router_falls_back_to_default_on_strategy_error(tmp_path):
    """Any strategy error must NOT propagate — caller gets settings.llm_provider."""
    log_path = tmp_path / "decisions.jsonl"
    r = LLMRouter(
        strategy=_ExplodingStrategy(),  # type: ignore[arg-type]
        decision_log=DecisionLogWriter(path=log_path),
    )
    chosen = r.route("sess-x")
    assert chosen == settings.llm_provider
    # Failure path should NOT have written a decision line.
    assert not log_path.exists() or log_path.read_text() == ""


def test_router_logs_decision_on_success(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    r = LLMRouter(
        strategy=WeightedRandomStrategy({"mock": 100}),
        decision_log=DecisionLogWriter(path=log_path),
    )
    assert r.route("sess-a") == "mock"
    assert r.route("sess-b") == "mock"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    sessions = [json.loads(l)["session_id"] for l in lines]
    assert sessions == ["sess-a", "sess-b"]


# ---------------------------------------------------------------------------
# 6. services.llm.get_chat_model wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def routing_on(tmp_path):
    """Turn routing on with a tmp decision log; restore settings on exit."""
    orig_enabled = settings.llm_routing_enabled
    orig_weights = settings.llm_routing_weights
    orig_log = settings.llm_routing_decision_log
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider

    try:
        settings.llm_routing_enabled = True
        settings.llm_routing_weights = "mock:100"
        settings.llm_routing_decision_log = str(tmp_path / "decisions.jsonl")
        # Everything must still work if the chat provider is mock.
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        registry.reset_caches()
        reset_router()
        yield tmp_path / "decisions.jsonl"
    finally:
        settings.llm_routing_enabled = orig_enabled
        settings.llm_routing_weights = orig_weights
        settings.llm_routing_decision_log = orig_log
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        registry.reset_caches()
        reset_router()


def test_routing_disabled_does_not_consult_router(monkeypatch):
    """When LLM_ROUTING_ENABLED=false, the router's route() is never called."""
    from sql_agent.services import llm as llm_mod

    calls = {"n": 0}

    def spy_router():
        calls["n"] += 1

        class _R:
            def route(self, _):
                calls["n"] += 1000  # big marker if .route ever runs
                return "openai"

        return _R()

    # Ensure the flag is off (default).
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "embedding_provider", "mock")
    monkeypatch.setattr(settings, "llm_routing_enabled", False)
    registry.reset_caches()

    # Patch get_router so any inadvertent call would show up.
    import sql_agent.routing.router as router_mod
    monkeypatch.setattr(router_mod, "get_router", spy_router)

    chat = llm_mod.get_chat_model()
    assert chat is not None
    assert calls["n"] == 0  # spy_router never invoked


def test_routing_enabled_writes_one_decision_per_call(routing_on):
    from sql_agent.services import llm as llm_mod

    log_path = routing_on
    for i in range(4):
        with session_scope(f"sess-{i}"):
            chat = llm_mod.get_chat_model()
            assert chat is not None

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    decisions = [json.loads(l) for l in lines]
    assert [d["session_id"] for d in decisions] == [
        "sess-0",
        "sess-1",
        "sess-2",
        "sess-3",
    ]
    assert all(d["provider"] == "mock" for d in decisions)
    assert all(d["strategy"] == "weighted_random" for d in decisions)


# ---------------------------------------------------------------------------
# 7. session_id_var thread isolation
# ---------------------------------------------------------------------------


def test_session_id_var_isolates_across_threads():
    """Each thread sees its own session id, regardless of others' sets."""
    observed: dict = {}

    def worker(i: int) -> None:
        with session_scope(f"thread-{i}"):
            # Give other threads a chance to interleave.
            import time

            time.sleep(0.01 * ((i % 3) + 1))
            observed[i] = session_id_var.get()

    with ThreadPoolExecutor(max_workers=8) as exe:
        futs = [exe.submit(worker, i) for i in range(8)]
        for f in as_completed(futs):
            f.result()

    for i in range(8):
        assert observed[i] == f"thread-{i}"


def test_session_scope_resets_outer_value():
    """Nested scopes restore the outer value on exit."""
    with session_scope("outer"):
        assert session_id_var.get() == "outer"
        with session_scope("inner"):
            assert session_id_var.get() == "inner"
        assert session_id_var.get() == "outer"
    assert session_id_var.get() is None


# ---------------------------------------------------------------------------
# 8. End-to-end through the gRPC servicer (in-process)
# ---------------------------------------------------------------------------


def test_servicer_writes_routing_decision_per_rpc(routing_on):
    """Each ExecuteSQL RPC should produce exactly one decision log line."""
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server.server import SqlAgentServicer

    class _Ctx:
        def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

        def set_code(self, _): pass

        def set_details(self, _): pass

    log_path: Path = routing_on
    # Reset memory manager so it picks up mock embedder dim.
    from sql_agent.services.memory_manager import reset_memory_manager
    reset_memory_manager()

    servicer = SqlAgentServicer()
    for i in range(3):
        resp = servicer.ExecuteSQL(
            pb2.ExecuteSQLRequest(
                query="How many orders are there?",
                session_id=f"rpc-{i}",
            ),
            _Ctx(),
        )
        assert resp.success is True

    decisions = [
        json.loads(l)
        for l in log_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    # The pipeline makes MULTIPLE get_chat_model() calls per run_turn (intent
    # + param_builder + potentially summarize). What we can reliably assert:
    # at least one decision per session_id, AND every decision's session_id
    # is one of the three we submitted.
    assert len(decisions) >= 3
    submitted = {"rpc-0", "rpc-1", "rpc-2"}
    assert {d["session_id"] for d in decisions} == submitted
    assert all(d["provider"] == "mock" for d in decisions)
