"""The main router — combines a strategy with a decision log.

Process-wide singleton via ``get_router()``; ``reset_router()`` is a test
hook that drops the cached instance so the next call re-reads settings.

Phase 8.7 additions:
    * ``LLM_ROUTING_STRATEGY=hash_by_id`` for session-sticky routing.
    * ``LLM_ROUTING_CIRCUIT_BREAKER=true`` wraps the chosen strategy in a
      per-provider circuit breaker.
    * SIGHUP on POSIX reloads the router (re-reads settings). On Windows
      we expose ``reload_router()`` which the admin can wire to any
      trigger they like.
"""

from __future__ import annotations

import signal
import threading
from typing import Optional

from sql_agent.config import get_logger, settings

from .base import RoutingDecision, RoutingStrategy
from .circuit_breaker import CircuitBreakingStrategy
from .decision_log import DecisionLogWriter
from .deterministic import HashByIDStrategy
from .weighted import WeightedRandomStrategy


_log = get_logger("routing.router")


class LLMRouter:
    """Chooses a provider per request and (optionally) logs the decision.

    Routing failures are never fatal — any exception in ``choose`` or
    ``append`` is caught, logged at WARNING, and the caller receives the
    fallback default provider name (taken from settings.llm_provider).
    """

    def __init__(
        self,
        strategy: Optional[RoutingStrategy] = None,
        decision_log: Optional[DecisionLogWriter] = None,
        *,
        enable_log: bool = True,
    ) -> None:
        self._strategy: RoutingStrategy = strategy or self._default_strategy()
        if enable_log and decision_log is None:
            try:
                self._decision_log: Optional[DecisionLogWriter] = DecisionLogWriter()
            except Exception as exc:  # pragma: no cover — filesystem edge-case
                _log.warning("DecisionLogWriter init failed: %s", exc)
                self._decision_log = None
        else:
            self._decision_log = decision_log

    @staticmethod
    def _default_strategy() -> RoutingStrategy:
        strategy_name = (settings.llm_routing_strategy or "weighted").strip().lower()
        weights_spec = settings.llm_routing_weights
        try:
            if strategy_name == "hash_by_id":
                base: RoutingStrategy = HashByIDStrategy.from_env_string(weights_spec)
            elif strategy_name in ("weighted", "weighted_random", ""):
                base = WeightedRandomStrategy.from_env_string(weights_spec)
            else:
                _log.warning(
                    "Unknown LLM_ROUTING_STRATEGY=%r; falling back to weighted.",
                    strategy_name,
                )
                base = WeightedRandomStrategy.from_env_string(weights_spec)
        except Exception as exc:
            _log.warning(
                "Failed to build routing strategy (%s); degrading to single-provider.",
                exc,
            )
            base = WeightedRandomStrategy({settings.llm_provider: 1})

        if settings.llm_routing_circuit_breaker:
            base = CircuitBreakingStrategy(
                base,
                failure_threshold=settings.llm_routing_breaker_threshold,
                cooldown_seconds=settings.llm_routing_breaker_cooldown_seconds,
            )
            _log.info(
                "CircuitBreakingStrategy wrapping %s (threshold=%d cooldown=%.1fs)",
                getattr(base._inner, "name", "?"),  # type: ignore[attr-defined]
                settings.llm_routing_breaker_threshold,
                settings.llm_routing_breaker_cooldown_seconds,
            )
        return base

    # ------------------------------------------------------------------

    def route(self, session_id: Optional[str]) -> str:
        """Return the provider name to use for this request.

        Falls back to ``settings.llm_provider`` on any internal error so
        that routing never blocks the pipeline.
        """
        try:
            provider = self._strategy.choose(session_id)
        except Exception as exc:
            _log.warning(
                "Routing strategy failed (%s); falling back to %s",
                exc,
                settings.llm_provider,
            )
            return settings.llm_provider

        if self._decision_log is not None:
            try:
                self._decision_log.append(
                    RoutingDecision.now(
                        session_id=session_id,
                        provider=provider,
                        strategy=getattr(self._strategy, "name", "custom"),
                        weights=dict(getattr(self._strategy, "weights", {})),
                    )
                )
            except Exception as exc:
                _log.warning("Decision log append failed: %s", exc)

        # Phase 5 metrics: best-effort, never fails routing.
        try:
            from sql_agent.observability.metrics import get_metrics

            get_metrics().record_routing(provider)
        except Exception as exc:  # pragma: no cover — defensive
            _log.debug("metrics record_routing failed: %s", exc)

        return provider

    @property
    def strategy(self) -> RoutingStrategy:
        return self._strategy

    @property
    def decision_log(self) -> Optional[DecisionLogWriter]:
        return self._decision_log

    # Phase 8.7 — call these from the LLM execution path to feed the
    # circuit breaker (if one is wrapping the strategy).
    def report_failure(self, provider: str) -> None:
        strat = self._strategy
        reporter = getattr(strat, "report_failure", None)
        if callable(reporter):
            try:
                reporter(provider)
            except Exception as exc:  # pragma: no cover — defensive
                _log.debug("report_failure(%s) raised: %s", provider, exc)

    def report_success(self, provider: str) -> None:
        strat = self._strategy
        reporter = getattr(strat, "report_success", None)
        if callable(reporter):
            try:
                reporter(provider)
            except Exception as exc:  # pragma: no cover
                _log.debug("report_success(%s) raised: %s", provider, exc)


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------


_instance: Optional[LLMRouter] = None
_instance_lock = threading.Lock()


def get_router() -> LLMRouter:
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = LLMRouter()
        return _instance


def reset_router() -> None:
    """Drop the cached router. Tests call this after mutating settings."""
    global _instance
    with _instance_lock:
        _instance = None


def reload_router() -> None:
    """Drop the cached router so the next ``get_router()`` re-reads settings.

    Wire to SIGHUP via ``install_sighup_reload_handler()`` for a live
    config reload. Safe to call at any time.
    """
    _log.info("reload_router(): dropping cached router instance")
    reset_router()


def install_sighup_reload_handler() -> bool:
    """Install a SIGHUP handler that reloads the router.

    POSIX-only (Windows has no SIGHUP). Returns True if the handler was
    installed, False otherwise. Safe to call multiple times.
    """
    if not hasattr(signal, "SIGHUP"):
        _log.info(
            "SIGHUP not available on this platform; router reload must "
            "be triggered programmatically via reload_router()."
        )
        return False
    try:
        signal.signal(signal.SIGHUP, lambda *_: reload_router())
        _log.info("SIGHUP handler installed — kill -HUP <pid> reloads routing.")
        return True
    except (ValueError, OSError) as exc:  # not main thread / permission
        _log.warning("Could not install SIGHUP handler: %s", exc)
        return False
