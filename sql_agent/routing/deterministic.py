"""Deterministic (session-sticky) routing strategy — phase 8.7.

Hashes ``session_id`` into the weighted buckets: the same session always
routes to the same provider. Useful for multi-turn chats where a consistent
LLM improves UX (user sees consistent "personality" / style / latency).

Sessions without an id fall back to random selection.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, Optional


class HashByIDStrategy:
    name: str = "hash_by_id"

    def __init__(
        self,
        weights: Dict[str, int],
        *,
        fallback_rng: Optional[random.Random] = None,
    ) -> None:
        if not weights:
            raise ValueError("HashByIDStrategy requires at least one weight")
        for k, v in weights.items():
            if not isinstance(v, int):
                raise TypeError(f"Weight for {k!r} must be int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"Weight for {k!r} must be >= 0, got {v}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"Sum of weights must be > 0, got {total}")

        self.weights: Dict[str, int] = dict(weights)
        # Deterministic iteration order.
        self._providers = list(self.weights.keys())
        self._values = [self.weights[p] for p in self._providers]
        self._cumulative = []
        running = 0
        for v in self._values:
            running += v
            self._cumulative.append(running)
        self._total = running
        self._rng = fallback_rng or random.Random()

    def choose(self, session_id: Optional[str] = None) -> str:
        if session_id:
            # Stable uniform distribution over [0, total) derived from
            # sha256(session_id). Bucketed via the cumulative weights.
            digest = hashlib.sha256(session_id.encode("utf-8")).digest()
            n = int.from_bytes(digest[:8], "big") % self._total
        else:
            n = self._rng.randint(0, self._total - 1)
        for provider, upper in zip(self._providers, self._cumulative):
            if n < upper:
                return provider
        return self._providers[-1]  # pragma: no cover — numerical safety

    @classmethod
    def from_env_string(
        cls, spec: str, *, fallback_rng: Optional[random.Random] = None
    ) -> "HashByIDStrategy":
        from .weighted import WeightedRandomStrategy  # reuse the parser

        parsed = WeightedRandomStrategy.from_env_string(spec)
        return cls(parsed.weights, fallback_rng=fallback_rng)
