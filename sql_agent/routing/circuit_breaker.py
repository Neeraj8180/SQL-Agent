"""Circuit-breaker wrapper for routing strategies — phase 8.7.

Tracks consecutive failures per provider and temporarily opens a circuit
(removes the provider from selection) when the failure threshold is hit.
The circuit transitions:

    CLOSED  --[failures >= threshold]-->  OPEN
    OPEN    --[cooldown elapsed]--------> HALF_OPEN
    HALF_OPEN --[success]------------->  CLOSED
    HALF_OPEN --[failure]-------------->  OPEN (cooldown resets)

Pluggable: wraps any underlying ``RoutingStrategy`` (weighted, hash-by-id,
future bandit strategies…). Reports failures via ``report_failure`` and
successes via ``report_success`` — typically called by the ``LLMRouter``
after the actual LLM call completes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _ProviderState:
    failure_count: int = 0
    opened_at: float = 0.0  # 0.0 => closed
    state: str = "closed"   # "closed" | "open" | "half_open"


class CircuitBreakingStrategy:
    """Wrap a strategy with per-provider failure tracking."""

    name: str = "circuit_breaker"

    def __init__(
        self,
        inner,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        self._inner = inner
        self._threshold = int(failure_threshold)
        self._cooldown = float(cooldown_seconds)
        self._states: Dict[str, _ProviderState] = {}
        self._lock = threading.Lock()

    @property
    def weights(self):
        # Exposed so the router's decision log still records meaningful
        # weight metadata.
        return getattr(self._inner, "weights", {})

    def choose(self, session_id: Optional[str] = None) -> str:
        with self._lock:
            self._refresh_half_open_locked()
            blocked = {
                p for p, s in self._states.items() if s.state == "open"
            }

        # Fast path: if nothing is blocked, just delegate.
        if not blocked:
            return self._inner.choose(session_id)

        # Slow path: try up to N times; if the inner strategy keeps picking
        # a blocked provider, eventually just return it (the circuit may be
        # open but we'd rather serve-with-error than spin forever).
        #
        # In practice, with weighted/hash strategies this converges in ~1
        # retry for typical 2-3 provider setups.
        for _ in range(10):
            provider = self._inner.choose(session_id)
            if provider not in blocked:
                return provider
        return provider  # surrender

    def report_failure(self, provider: str) -> None:
        """Record a failed LLM call for ``provider``. Opens the circuit
        when the threshold is hit."""
        with self._lock:
            st = self._states.setdefault(provider, _ProviderState())
            if st.state == "open":
                return  # already open; cooldown governs re-entry
            st.failure_count += 1
            if st.failure_count >= self._threshold:
                st.state = "open"
                st.opened_at = time.monotonic()

    def report_success(self, provider: str) -> None:
        """Record a successful LLM call — resets counter and closes circuit."""
        with self._lock:
            st = self._states.setdefault(provider, _ProviderState())
            st.failure_count = 0
            st.state = "closed"
            st.opened_at = 0.0

    def is_open(self, provider: str) -> bool:
        """Introspection for tests / admin tools."""
        with self._lock:
            self._refresh_half_open_locked()
            st = self._states.get(provider)
            return bool(st and st.state == "open")

    def state_of(self, provider: str) -> str:
        with self._lock:
            self._refresh_half_open_locked()
            st = self._states.get(provider)
            return st.state if st else "closed"

    def _refresh_half_open_locked(self) -> None:
        now = time.monotonic()
        for st in self._states.values():
            if st.state == "open" and (now - st.opened_at) >= self._cooldown:
                st.state = "half_open"
                st.failure_count = 0
