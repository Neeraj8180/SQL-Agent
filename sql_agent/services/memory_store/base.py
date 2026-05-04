"""Memory-store Protocol shared by all backends.

The four public operations are symmetric across reward / penalty indices
so ``MemoryManager`` can treat both kinds uniformly.

Every backend returns payloads as plain ``dict``s so the calling layer
(memory_agent.format_rules_for_prompt) needs no backend-specific code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Contract for all reward/penalty memory backends."""

    #: Human-readable backend id: "faiss" | "qdrant" | ...
    name: str
    #: Embedding dimension the store is configured for.
    dimension: int

    # ---- writes ---------------------------------------------------------

    def add_reward(self, vector: List[float], payload: Dict[str, Any]) -> None: ...

    def add_penalty(self, vector: List[float], payload: Dict[str, Any]) -> None: ...

    # ---- reads ----------------------------------------------------------

    def search_rewards(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]: ...

    def search_penalties(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]: ...

    # ---- introspection (for metrics / admin) ----------------------------

    @property
    def reward_size(self) -> int: ...

    @property
    def penalty_size(self) -> int: ...
