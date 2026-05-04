"""Routing protocols and shared types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class RoutingStrategy(Protocol):
    """Selects a provider name per request.

    Implementations SHOULD be thread-safe — ``choose`` may be called
    concurrently from gRPC handler threads.
    """

    name: str

    def choose(self, session_id: Optional[str] = None) -> str: ...


@dataclass
class RoutingDecision:
    """One row of the decision log."""

    ts: str
    session_id: Optional[str]
    provider: str
    strategy: str
    weights: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        *,
        session_id: Optional[str],
        provider: str,
        strategy: str,
        weights: Optional[Dict[str, int]] = None,
    ) -> "RoutingDecision":
        return cls(
            ts=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            provider=provider,
            strategy=strategy,
            weights=dict(weights or {}),
        )

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "session_id": self.session_id,
                "provider": self.provider,
                "strategy": self.strategy,
                "weights": self.weights,
            },
            ensure_ascii=False,
        )
