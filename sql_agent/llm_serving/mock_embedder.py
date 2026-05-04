"""Deterministic mock embedder — no network, no torch.

Returns a stable vector derived from SHA-256 of the input so tests can
verify "same text -> same vector" without relying on model-specific
numerical behavior.

Default dimension is 64 — small enough to keep test FAISS files tiny,
large enough that collision probability is negligible. Phase 2.4 added
dim-namespaced FAISS (``dim{N}/`` subdirectory), so a non-1536 default is
now safe and does not interfere with OpenAI-backed indices written at a
different dim.
"""

from __future__ import annotations

import hashlib
from typing import List

from sql_agent.config import get_logger


_log = get_logger("llm_serving.mock_embed")


_DEFAULT_DIM = 64


def _text_to_vector(text: str, dim: int) -> List[float]:
    """Deterministic pseudo-embedding.

    Uses SHA-256 of the text as a seed source and expands it to `dim` floats
    in [-1, 1]. Collision-unlikely for any realistic test corpus.
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    # Cycle through bytes to fill `dim` slots.
    out: List[float] = []
    for i in range(dim):
        b = digest[i % len(digest)]
        out.append((b / 127.5) - 1.0)
    return out


class MockEmbedder:
    name: str = "mock"
    model_id: str = "mock"

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("MockEmbedder dim must be > 0")
        self.dimension = dim

    def embed(self, text: str) -> List[float]:
        return _text_to_vector(text or "", self.dimension)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]
