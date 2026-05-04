"""Phase 8.7 tests — deterministic (hash-by-id) routing, circuit breaker,
SIGHUP reload."""

from __future__ import annotations

import time
from collections import Counter

import pytest

from sql_agent.config import settings
from sql_agent.routing.circuit_breaker import CircuitBreakingStrategy
from sql_agent.routing.deterministic import HashByIDStrategy
from sql_agent.routing.router import (
    LLMRouter,
    reload_router,
    reset_router,
)
from sql_agent.routing.weighted import WeightedRandomStrategy


# ---------------------------------------------------------------------------
# HashByIDStrategy
# ---------------------------------------------------------------------------


def test_hash_by_id_is_deterministic():
    s = HashByIDStrategy({"a": 50, "b": 50})
    # Same session id -> same provider every time.
    for _ in range(100):
        assert s.choose("sess-A") == s.choose("sess-A")


def test_hash_by_id_distribution_is_approximately_uniform_across_session_ids():
    """Over many distinct session ids, a 50/50 split should land
    ~50/50. We use a generous tolerance (±10%) because there are only 1000
    buckets in this test."""
    s = HashByIDStrategy({"a": 50, "b": 50})
    picks = Counter(s.choose(f"sess-{i}") for i in range(1000))
    assert 400 <= picks["a"] <= 600
    assert 400 <= picks["b"] <= 600


def test_hash_by_id_respects_weights():
    """70/30 split with 1000 session ids should give ~700/300."""
    s = HashByIDStrategy({"prod": 70, "canary": 30})
    picks = Counter(s.choose(f"sess-{i}") for i in range(1000))
    assert 600 <= picks["prod"] <= 800
    assert 200 <= picks["canary"] <= 400


def test_hash_by_id_rejects_bad_weights():
    with pytest.raises(ValueError):
        HashByIDStrategy({})
    with pytest.raises(ValueError):
        HashByIDStrategy({"a": 0, "b": 0})
    with pytest.raises(ValueError):
        HashByIDStrategy({"a": -1})


def test_hash_by_id_from_env_string():
    s = HashByIDStrategy.from_env_string("openai:70, hf:30")
    assert s.weights == {"openai": 70, "hf": 30}


# ---------------------------------------------------------------------------
# CircuitBreakingStrategy
# ---------------------------------------------------------------------------


def test_circuit_opens_after_threshold_failures():
    inner = WeightedRandomStrategy({"bad": 100})
    cb = CircuitBreakingStrategy(inner, failure_threshold=3, cooldown_seconds=60)
    for _ in range(3):
        cb.report_failure("bad")
    assert cb.is_open("bad")
    assert cb.state_of("bad") == "open"


def test_circuit_success_resets_counter():
    inner = WeightedRandomStrategy({"good": 100})
    cb = CircuitBreakingStrategy(inner, failure_threshold=3, cooldown_seconds=60)
    cb.report_failure("good")
    cb.report_failure("good")
    cb.report_success("good")
    # Third failure should NOT open (counter was reset).
    cb.report_failure("good")
    assert cb.state_of("good") == "closed"


def test_circuit_half_open_after_cooldown():
    inner = WeightedRandomStrategy({"flaky": 100})
    cb = CircuitBreakingStrategy(
        inner, failure_threshold=2, cooldown_seconds=0.05
    )
    cb.report_failure("flaky")
    cb.report_failure("flaky")
    assert cb.state_of("flaky") == "open"
    time.sleep(0.1)
    assert cb.state_of("flaky") == "half_open"


def test_circuit_routes_around_open_providers():
    """When one provider's circuit is open, choose() should pick the other."""
    # 50/50 split; when "bad" opens, all traffic goes to "good".
    inner = WeightedRandomStrategy({"bad": 50, "good": 50})
    cb = CircuitBreakingStrategy(inner, failure_threshold=1, cooldown_seconds=60)
    cb.report_failure("bad")
    assert cb.state_of("bad") == "open"

    picks = Counter(cb.choose(None) for _ in range(200))
    assert picks["good"] == 200
    assert picks["bad"] == 0


def test_circuit_fails_closed_when_all_providers_open():
    """If every provider is tripped, choose() returns SOMETHING (not spin)."""
    inner = WeightedRandomStrategy({"a": 50, "b": 50})
    cb = CircuitBreakingStrategy(inner, failure_threshold=1, cooldown_seconds=60)
    cb.report_failure("a")
    cb.report_failure("b")
    result = cb.choose(None)
    assert result in ("a", "b")


# ---------------------------------------------------------------------------
# Router feedback path
# ---------------------------------------------------------------------------


def test_router_report_success_and_failure_are_silent_on_non_cb_strategy():
    """No circuit breaker => report_success/failure are no-ops that don't raise."""
    r = LLMRouter(
        strategy=WeightedRandomStrategy({"mock": 100}),
        decision_log=None,
        enable_log=False,
    )
    r.report_success("mock")  # no-op
    r.report_failure("mock")  # no-op
    # No exception => pass.


def test_router_builds_hash_strategy_when_settings_say_so():
    orig_s = settings.llm_routing_strategy
    orig_w = settings.llm_routing_weights
    try:
        settings.llm_routing_strategy = "hash_by_id"
        settings.llm_routing_weights = "a:50, b:50"
        reset_router()
        from sql_agent.routing.router import get_router

        r = get_router()
        assert isinstance(r.strategy, HashByIDStrategy)
    finally:
        settings.llm_routing_strategy = orig_s
        settings.llm_routing_weights = orig_w
        reset_router()


def test_router_wraps_with_circuit_breaker_when_enabled():
    orig_cb = settings.llm_routing_circuit_breaker
    orig_w = settings.llm_routing_weights
    try:
        settings.llm_routing_circuit_breaker = True
        settings.llm_routing_weights = "mock:100"
        reset_router()
        from sql_agent.routing.router import get_router

        r = get_router()
        assert isinstance(r.strategy, CircuitBreakingStrategy)
    finally:
        settings.llm_routing_circuit_breaker = orig_cb
        settings.llm_routing_weights = orig_w
        reset_router()


# ---------------------------------------------------------------------------
# SIGHUP / reload
# ---------------------------------------------------------------------------


def test_reload_router_drops_cached_instance():
    from sql_agent.routing.router import get_router

    orig_cb = settings.llm_routing_circuit_breaker
    orig_w = settings.llm_routing_weights
    try:
        settings.llm_routing_circuit_breaker = False
        settings.llm_routing_weights = "a:100"
        reset_router()
        r1 = get_router()
        assert not isinstance(r1.strategy, CircuitBreakingStrategy)

        # Change settings, reload, observe new strategy type.
        settings.llm_routing_circuit_breaker = True
        reload_router()
        r2 = get_router()
        assert r1 is not r2
        assert isinstance(r2.strategy, CircuitBreakingStrategy)
    finally:
        settings.llm_routing_circuit_breaker = orig_cb
        settings.llm_routing_weights = orig_w
        reset_router()


def test_install_sighup_handler_returns_false_on_windows():
    """On Windows (no SIGHUP), the installer should gracefully return False."""
    import sys

    from sql_agent.routing.router import install_sighup_reload_handler

    installed = install_sighup_reload_handler()
    if sys.platform == "win32":
        assert installed is False
    else:
        # POSIX: may be True (if we're on the main thread) or False
        # (signal.signal raises from non-main thread, e.g. pytest-xdist).
        assert installed in (True, False)
