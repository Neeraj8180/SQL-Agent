"""Probability-weighted random strategy.

Example:
    >>> s = WeightedRandomStrategy.from_env_string("openai:70, hf:30")
    >>> s.choose()
    'openai'  # ~70% of the time
"""

from __future__ import annotations

import random
from typing import Dict, Optional


class WeightedRandomStrategy:
    name: str = "weighted_random"

    def __init__(
        self,
        weights: Dict[str, int],
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not weights:
            raise ValueError("WeightedRandomStrategy requires at least one weight")
        for k, v in weights.items():
            if not isinstance(v, int):
                raise TypeError(f"Weight for {k!r} must be int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"Weight for {k!r} must be >= 0, got {v}")
        if sum(weights.values()) <= 0:
            raise ValueError(
                f"Sum of weights must be > 0, got {sum(weights.values())}"
            )

        # Deterministic key order makes choose() reproducible under a seeded RNG.
        self.weights: Dict[str, int] = dict(weights)
        self._providers = list(self.weights.keys())
        self._values = [self.weights[p] for p in self._providers]
        self._rng = rng or random.Random()

    def choose(self, session_id: Optional[str] = None) -> str:
        # session_id is not used by weighted-random; accepted for protocol
        # uniformity (other strategies may use it).
        return self._rng.choices(self._providers, weights=self._values, k=1)[0]

    # ------------------------------------------------------------------
    # Env-string parser.
    # Format: "provider1:weight1, provider2:weight2, ..."
    # Duplicate provider names accumulate their weights.
    # ------------------------------------------------------------------

    @classmethod
    def from_env_string(
        cls, spec: str, *, rng: Optional[random.Random] = None
    ) -> "WeightedRandomStrategy":
        weights: Dict[str, int] = {}
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(
                    f"Invalid routing spec chunk {chunk!r}; expected 'provider:weight'."
                )
            name, weight_str = chunk.split(":", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Empty provider name in chunk {chunk!r}.")
            try:
                weight = int(weight_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid weight for {name!r}: {weight_str!r}"
                ) from exc
            weights[name] = weights.get(name, 0) + weight
        return cls(weights, rng=rng)
