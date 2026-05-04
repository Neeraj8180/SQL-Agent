"""Memory-store factory — pick FAISS or Qdrant based on settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sql_agent.config import get_logger, settings


_log = get_logger("memory_store.factory")


def build_memory_store(
    *, dimension: int, root_dir: Optional[Path] = None
):
    """Return the configured ``MemoryStore`` backend.

    Resolution rules:
        1. ``MEMORY_STORE_BACKEND=qdrant`` (explicit) → QdrantBackend.
        2. ``MEMORY_STORE_BACKEND=faiss``  (default)  → FaissBackend.
        3. ``MEMORY_STORE_BACKEND=auto``   → Qdrant if ``QDRANT_URL`` is
           set (multi-replica deployments), FAISS otherwise.
    """
    backend = (settings.memory_store_backend or "faiss").strip().lower()

    if backend == "auto":
        backend = "qdrant" if settings.qdrant_url.strip() else "faiss"

    if backend == "qdrant":
        from .qdrant_backend import QdrantBackend

        return QdrantBackend(dimension=dimension)

    if backend == "faiss":
        from .faiss_backend import FaissBackend

        return FaissBackend(dimension=dimension, root_dir=root_dir)

    _log.warning(
        "Unknown MEMORY_STORE_BACKEND=%r; falling back to FAISS.", backend
    )
    from .faiss_backend import FaissBackend

    return FaissBackend(dimension=dimension, root_dir=root_dir)
