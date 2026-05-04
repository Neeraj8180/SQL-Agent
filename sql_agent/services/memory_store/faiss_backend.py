"""FAISS file-backed memory store — the default backend.

Characteristics:
    * Pod-local: each process has its own copy.
    * Zero-dependency: FAISS is already in the core deps.
    * Best for single-replica dev / single-node production / edge deployments.

For multi-replica K8s deployments, use ``QdrantBackend`` instead so all pods
share one view of memory. FAISS multi-replica causes divergence (each pod
writes to its own filesystem) — phase 7 mitigated this with
``READ_ONLY_MEMORY=true``; phase 9 ``QdrantBackend`` eliminates it.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from sql_agent.config import get_logger, settings


_log = get_logger("memory_store.faiss")


def _migrate_legacy_indices(root_base: Path, legacy_dim: int = 1536) -> None:
    """Move pre-2.4 loose ``{name}.{index,jsonl}`` into ``dim{N}/``.

    Idempotent: if the target path already exists we leave both files and
    log a warning so the operator can reconcile.
    """
    target_dir = root_base / f"dim{legacy_dim}"
    for name in ("reward", "penalty"):
        for ext in (".index", ".jsonl"):
            legacy = root_base / f"{name}{ext}"
            if not legacy.exists():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{name}{ext}"
            if target.exists():
                _log.warning(
                    "Legacy %s and %s both exist; skipping migration.",
                    legacy,
                    target,
                )
                continue
            try:
                legacy.rename(target)
                _log.info("Migrated legacy FAISS file: %s -> %s", legacy, target)
            except OSError as exc:  # pragma: no cover — OS-specific
                _log.warning("Failed to migrate %s: %s", legacy, exc)


class _FaissIndex:
    """Single FAISS index + JSONL sidecar, thread-safe."""

    def __init__(self, root: Path, name: str, dim: int) -> None:
        self._root = root
        self._name = name
        self._dim = dim
        self._index_path = root / f"{name}.index"
        self._meta_path = root / f"{name}.jsonl"
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
        self._meta: List[Dict[str, Any]] = self._load_meta()

    def _load_index(self) -> faiss.IndexFlatIP:
        if self._index_path.exists():
            try:
                idx = faiss.read_index(str(self._index_path))
                if getattr(idx, "d", self._dim) != self._dim:
                    _log.warning(
                        "FAISS index %s has dim %d but expected %d; rebuilding.",
                        self._index_path, idx.d, self._dim,
                    )
                    return faiss.IndexFlatIP(self._dim)
                return idx
            except Exception as exc:
                _log.warning(
                    "Failed to load %s: %s — rebuilding.", self._index_path, exc
                )
        return faiss.IndexFlatIP(self._dim)

    def _load_meta(self) -> List[Dict[str, Any]]:
        if not self._meta_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            for line in self._meta_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as exc:
            _log.warning("Failed to load meta %s: %s", self._meta_path, exc)
        return records

    def _persist(self) -> None:
        tmp_idx = self._index_path.with_suffix(".index.tmp")
        faiss.write_index(self._index, str(tmp_idx))
        os.replace(tmp_idx, self._index_path)

        tmp_meta = self._meta_path.with_suffix(".jsonl.tmp")
        with tmp_meta.open("w", encoding="utf-8") as f:
            for rec in self._meta:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_meta, self._meta_path)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add(self, vector: List[float], payload: Dict[str, Any]) -> None:
        with self._lock:
            v = np.asarray(vector, dtype="float32").reshape(1, -1)
            if v.shape[1] != self._dim:
                raise ValueError(
                    f"FAISS store '{self._name}' expected dim {self._dim} "
                    f"but got vector of dim {v.shape[1]}"
                )
            v = self._normalize(v)
            self._index.add(v)
            self._meta.append(payload)
            self._persist()

    def search(
        self, vector: List[float], k: int = 3, min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if self._index.ntotal == 0 or k <= 0:
                return []
            v = np.asarray(vector, dtype="float32").reshape(1, -1)
            if v.shape[1] != self._dim:
                _log.warning(
                    "Search vector dim %d != store dim %d; returning [].",
                    v.shape[1], self._dim,
                )
                return []
            v = self._normalize(v)
            k = min(k, self._index.ntotal)
            scores, idxs = self._index.search(v, k)
            out: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
                if idx < 0 or idx >= len(self._meta):
                    continue
                if score < min_score:
                    continue
                rec = dict(self._meta[idx])
                rec["score"] = float(score)
                out.append(rec)
            return out

    @property
    def size(self) -> int:
        return self._index.ntotal


class FaissBackend:
    """Filesystem-local memory backend. Conforms to ``MemoryStore`` protocol."""

    name: str = "faiss"

    def __init__(
        self, *, root_dir: Optional[Path] = None, dimension: int
    ) -> None:
        self.dimension = int(dimension)
        root_base = Path(root_dir or settings.faiss_dir)
        root_base.mkdir(parents=True, exist_ok=True)

        # One-time migration of pre-2.4 loose files (idempotent).
        _migrate_legacy_indices(root_base)

        self._root = root_base / f"dim{self.dimension}"
        self._root.mkdir(parents=True, exist_ok=True)
        self._reward = _FaissIndex(self._root, "reward", self.dimension)
        self._penalty = _FaissIndex(self._root, "penalty", self.dimension)
        _log.info(
            "FaissBackend ready: root=%s dim=%d reward_size=%d penalty_size=%d",
            self._root, self.dimension,
            self._reward.size, self._penalty.size,
        )

    # ---- writes ---------------------------------------------------------

    def add_reward(self, vector: List[float], payload: Dict[str, Any]) -> None:
        self._reward.add(vector, {**payload, "kind": "reward"})

    def add_penalty(self, vector: List[float], payload: Dict[str, Any]) -> None:
        self._penalty.add(vector, {**payload, "kind": "penalty"})

    # ---- reads ----------------------------------------------------------

    def search_rewards(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]:
        return self._reward.search(vector, k=k, min_score=min_score)

    def search_penalties(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]:
        return self._penalty.search(vector, k=k, min_score=min_score)

    # ---- introspection --------------------------------------------------

    @property
    def reward_size(self) -> int:
        return self._reward.size

    @property
    def penalty_size(self) -> int:
        return self._penalty.size
