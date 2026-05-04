"""LLM routing / A/B testing (phase 3).

Public surface (resolved lazily so that importing this package is free for
OpenAI-only, routing-disabled deployments):

    * ``RoutingStrategy``    - protocol
    * ``RoutingDecision``    - dataclass + JSONL serializer
    * ``WeightedRandomStrategy`` - probability-weighted provider selection
    * ``DecisionLogWriter``  - append-only JSONL, thread-safe
    * ``LLMRouter``          - combines the above
    * ``get_router``         - process-wide singleton accessor
    * ``reset_router``       - test hook
"""

from __future__ import annotations

__all__ = [
    "RoutingStrategy",
    "RoutingDecision",
    "WeightedRandomStrategy",
    "HashByIDStrategy",
    "CircuitBreakingStrategy",
    "DecisionLogWriter",
    "LLMRouter",
    "get_router",
    "reset_router",
    "reload_router",
    "install_sighup_reload_handler",
]


def __getattr__(name: str):
    if name in {"RoutingStrategy", "RoutingDecision"}:
        from . import base

        return getattr(base, name)
    if name == "WeightedRandomStrategy":
        from .weighted import WeightedRandomStrategy

        return WeightedRandomStrategy
    if name == "HashByIDStrategy":
        from .deterministic import HashByIDStrategy

        return HashByIDStrategy
    if name == "CircuitBreakingStrategy":
        from .circuit_breaker import CircuitBreakingStrategy

        return CircuitBreakingStrategy
    if name == "DecisionLogWriter":
        from .decision_log import DecisionLogWriter

        return DecisionLogWriter
    if name in {
        "LLMRouter",
        "get_router",
        "reset_router",
        "reload_router",
        "install_sighup_reload_handler",
    }:
        from . import router

        return getattr(router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
