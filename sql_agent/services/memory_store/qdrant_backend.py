"""Qdrant vector-DB backend — shared across replicas.

Phase 9 production fix for the multi-replica FAISS divergence problem.
All replicas of the sql-agent Deployment talk to the same Qdrant
instance, so a reward recorded by pod-A is immediately visible to pod-B.

Two operational modes:
    * Local / embedded: ``QDRANT_URL`` empty → opens ``<faiss_dir>/qdrant``
      as a persistent on-disk store. Good for single-node prod or dev
      parity with real Qdrant.
    * Remote: ``QDRANT_URL=http://qdrant:6333`` → talks to a separate
      Qdrant server (docker-compose service or K8s StatefulSet).

Collections:
    * ``<prefix>reward``  — stores successful query → params patterns.
    * ``<prefix>penalty`` — stores failure patterns to avoid.
    Prefix defaults to ``sql_agent_``; override via ``QDRANT_COLLECTION_PREFIX``
    to share one Qdrant cluster across multiple sql-agent tenants.

Thread-safety: the official ``qdrant-client`` is thread-safe for
read/write operations; we do not add our own lock.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_agent.config import get_logger, settings


_log = get_logger("memory_store.qdrant")


class QdrantBackend:
    """Qdrant-backed memory store. Conforms to ``MemoryStore`` protocol."""

    name: str = "qdrant"

    def __init__(
        self,
        *,
        dimension: int,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[Path] = None,
        collection_prefix: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "QdrantBackend requires qdrant-client. Install via: "
                "pip install -e .[vector-db]"
            ) from exc

        self.dimension = int(dimension)
        self._prefix = (
            collection_prefix
            if collection_prefix is not None
            else settings.qdrant_collection_prefix
        )
        self._reward_name = f"{self._prefix}reward"
        self._penalty_name = f"{self._prefix}penalty"

        # Pick local-path or remote based on explicit arg or settings.
        resolved_url = url if url is not None else settings.qdrant_url
        resolved_api_key = api_key if api_key is not None else settings.qdrant_api_key

        if resolved_url.strip():
            self._client = QdrantClient(
                url=resolved_url,
                api_key=resolved_api_key or None,
                timeout=10.0,
            )
            self._mode = f"remote({resolved_url})"
        else:
            # Embedded / file-backed. Stored alongside FAISS artifacts so
            # one rm -rf wipes both.
            qdrant_dir = Path(path or (settings.faiss_dir / "qdrant"))
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(qdrant_dir))
            self._mode = f"embedded({qdrant_dir})"

        # Ensure collections exist with the right dim. Idempotent.
        self._ensure_collection(
            self._reward_name, dimension=self.dimension,
            distance=Distance.COSINE, vec_params=VectorParams,
        )
        self._ensure_collection(
            self._penalty_name, dimension=self.dimension,
            distance=Distance.COSINE, vec_params=VectorParams,
        )

        self._lock = threading.Lock()  # only for counter caching; writes are thread-safe

        _log.info(
            "QdrantBackend ready: mode=%s dim=%d collections=%s,%s",
            self._mode, self.dimension,
            self._reward_name, self._penalty_name,
        )

    # ---- helpers --------------------------------------------------------

    def _ensure_collection(
        self, name: str, *, dimension: int, distance, vec_params
    ) -> None:
        try:
            info = self._client.get_collection(name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dimension:
                _log.warning(
                    "Qdrant collection %s has dim %d but expected %d; "
                    "using existing collection (will reject mismatched writes).",
                    name, existing_dim, dimension,
                )
        except Exception:
            # Doesn't exist — create it.
            self._client.create_collection(
                collection_name=name,
                vectors_config=vec_params(size=dimension, distance=distance),
            )
            _log.info("Created Qdrant collection: %s (dim=%d)", name, dimension)

    def _upsert(self, collection: str, vector: List[float], payload: Dict[str, Any]) -> None:
        from qdrant_client.http.models import PointStruct

        if len(vector) != self.dimension:
            raise ValueError(
                f"Qdrant '{collection}' expected dim {self.dimension} "
                f"but got vector of dim {len(vector)}"
            )
        # Qdrant uses UUIDs or ints for point IDs; a UUID4 is simplest and
        # collision-free across replicas.
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=list(vector),
            payload=dict(payload),
        )
        self._client.upsert(collection_name=collection, points=[point])

    def _search(
        self,
        collection: str,
        vector: List[float],
        k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        if k <= 0 or len(vector) != self.dimension:
            return []
        try:
            res = self._client.query_points(
                collection_name=collection,
                query=list(vector),
                limit=int(k),
                with_payload=True,
            )
        except Exception as exc:
            _log.warning("Qdrant search on %s failed: %s", collection, exc)
            return []
        hits = res.points if hasattr(res, "points") else res
        out: List[Dict[str, Any]] = []
        for hit in hits:
            score = float(hit.score)
            if score < min_score:
                continue
            payload = dict(hit.payload or {})
            payload["score"] = score
            out.append(payload)
        return out

    def _count(self, collection: str) -> int:
        try:
            result = self._client.count(
                collection_name=collection, exact=False
            )
            return int(result.count)
        except Exception as exc:  # pragma: no cover — defensive
            _log.debug("count(%s) failed: %s", collection, exc)
            return 0

    # ---- writes ---------------------------------------------------------

    def add_reward(self, vector: List[float], payload: Dict[str, Any]) -> None:
        self._upsert(self._reward_name, vector, {**payload, "kind": "reward"})

    def add_penalty(self, vector: List[float], payload: Dict[str, Any]) -> None:
        self._upsert(self._penalty_name, vector, {**payload, "kind": "penalty"})

    # ---- reads ----------------------------------------------------------

    def search_rewards(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]:
        return self._search(self._reward_name, vector, k, min_score)

    def search_penalties(
        self, vector: List[float], k: int, min_score: float
    ) -> List[Dict[str, Any]]:
        return self._search(self._penalty_name, vector, k, min_score)

    # ---- introspection --------------------------------------------------

    @property
    def reward_size(self) -> int:
        return self._count(self._reward_name)

    @property
    def penalty_size(self) -> int:
        return self._count(self._penalty_name)
