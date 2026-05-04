"""Dual-layer (reward + penalty) memory with pluggable backends.

Phase 9: delegates to ``memory_store`` backends (FAISS or Qdrant) instead
of holding FAISS directly. Public API is unchanged — agents call
``record_reward`` / ``record_penalty`` / ``recall`` exactly as before.

Backend selection via the ``MEMORY_STORE_BACKEND`` setting:
    * ``faiss`` (default)   — pod-local, zero-deps, single-replica-safe.
    * ``qdrant``            — shared vector DB; multi-replica-safe.
    * ``auto``              — qdrant if ``QDRANT_URL`` set, else faiss.

The ``READ_ONLY_MEMORY`` flag from phase 7 is still respected: it now
means "this pod will not WRITE to the shared store", which is useful
for read-heavy HPA replicas even with Qdrant (reduces write contention).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_agent.config import get_logger, settings
from sql_agent.services.llm import embed_text


_log = get_logger("memory_manager")


# Dimension assumed for legacy (pre-2.4) FAISS files. Used only during
# one-time migration in FaissBackend.
_LEGACY_DIM = 1536


def _probe_embedding_dim() -> int:
    """Ask the registry for the current embedding dim.

    Falls back to ``_LEGACY_DIM`` if the registry cannot construct a
    provider (e.g. missing API key for the OpenAI default).
    """
    try:
        from sql_agent.llm_serving.registry import get_embedding_provider

        return int(get_embedding_provider().dimension)
    except Exception as exc:
        _log.debug(
            "Could not probe embedding dim (%s); defaulting to %d",
            exc, _LEGACY_DIM,
        )
        return _LEGACY_DIM


class MemoryManager:
    """Coordinator for reward + penalty memory stores."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        dim = _probe_embedding_dim()
        self._dim = dim
        self._root_base = Path(root_dir or settings.faiss_dir)

        # Phase 9: delegate to the backend factory. Backend choice is
        # data-driven (MEMORY_STORE_BACKEND), so the same MemoryManager
        # code handles both FAISS and Qdrant transparently.
        from sql_agent.services.memory_store import build_memory_store

        self._store = build_memory_store(dimension=dim, root_dir=self._root_base)

        _log.info(
            "MemoryManager ready: backend=%s dim=%d reward_size=%d penalty_size=%d",
            self._store.name,
            self._dim,
            self._store.reward_size,
            self._store.penalty_size,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> str:
        return self._store.name

    @property
    def root(self) -> Path:
        return self._root_base

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record_reward(
        self,
        query: str,
        *,
        parameters: Dict[str, Any],
        tool_used: str,
        reasoning: Optional[str] = None,
    ) -> None:
        if settings.read_only_memory:
            return
        try:
            vec = embed_text(query)
        except Exception as exc:
            _log.warning("Skipping reward store (embed failed): %s", exc)
            return
        payload = {
            "query": query,
            "parameters": parameters,
            "tool_used": tool_used,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._store.add_reward(vec, payload)
        except ValueError as exc:
            _log.warning("Skipping reward store (dim mismatch): %s", exc)
        except Exception as exc:  # pragma: no cover — backend-specific
            _log.warning("Skipping reward store (backend error): %s", exc)

    def record_penalty(
        self,
        query: str,
        *,
        reason: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if settings.read_only_memory:
            return
        try:
            vec = embed_text(query)
        except Exception as exc:
            _log.warning("Skipping penalty store (embed failed): %s", exc)
            return
        payload = {
            "query": query,
            "reason": reason,
            "parameters": parameters or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._store.add_penalty(vec, payload)
        except ValueError as exc:
            _log.warning("Skipping penalty store (dim mismatch): %s", exc)
        except Exception as exc:  # pragma: no cover
            _log.warning("Skipping penalty store (backend error): %s", exc)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def recall(
        self, query: str, *, k_reward: int = 3, k_penalty: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            vec = embed_text(query)
        except Exception as exc:
            _log.warning("Skipping recall (embed failed): %s", exc)
            return []
        rewards = self._store.search_rewards(vec, k=k_reward, min_score=0.3)
        penalties = self._store.search_penalties(vec, k=k_penalty, min_score=0.3)
        return rewards + penalties

    @property
    def reward_size(self) -> int:
        return self._store.reward_size

    @property
    def penalty_size(self) -> int:
        return self._store.penalty_size


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_instance: Optional[MemoryManager] = None
_instance_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = MemoryManager()
        return _instance


def reset_memory_manager() -> None:
    """Drop the cached MemoryManager singleton. Test hook."""
    global _instance
    with _instance_lock:
        _instance = None
