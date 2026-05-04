"""Pluggable memory backends for the reward/penalty vector store.

Public surface:
    * ``MemoryStore`` — runtime-checkable Protocol every backend satisfies.
    * ``build_memory_store`` — factory that reads settings and returns the
      right backend (FAISS for single-replica, Qdrant for multi-replica).

Two implementations ship today:
    * ``FaissBackend`` — filesystem-local, zero extra infrastructure.
    * ``QdrantBackend`` — shared vector DB; eliminates the multi-replica
      divergence problem that READ_ONLY_MEMORY only *mitigates*.
"""

from __future__ import annotations

__all__ = [
    "MemoryStore",
    "build_memory_store",
    "FaissBackend",
    "QdrantBackend",
]


def __getattr__(name: str):
    if name == "MemoryStore":
        from .base import MemoryStore

        return MemoryStore
    if name == "FaissBackend":
        from .faiss_backend import FaissBackend

        return FaissBackend
    if name == "QdrantBackend":
        from .qdrant_backend import QdrantBackend

        return QdrantBackend
    if name == "build_memory_store":
        from .factory import build_memory_store

        return build_memory_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
