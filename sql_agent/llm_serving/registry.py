"""Provider factory + process-wide singleton cache.

Reads ``settings.llm_provider`` / ``settings.embedding_provider`` to build the
right implementations. Callers may also request a specific provider by name,
which is useful for tests and A/B routing (phase 3).

Thread-safety: the cache is lock-protected; providers themselves are
expected to be thread-safe (both OpenAI's ChatOpenAI and transformers
pipelines are used under GIL-bound async/sync code paths).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from sql_agent.config import get_logger, settings

from .base import EmbeddingProvider, LLMProvider, ProviderUnavailableError


_log = get_logger("llm_serving.registry")


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


_llm_cache: Dict[str, LLMProvider] = {}
_embed_cache: Dict[str, EmbeddingProvider] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _build_llm(name: str) -> LLMProvider:
    n = name.strip().lower()
    if n == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if n == "mock":
        from .mock_provider import MockProvider

        return MockProvider()
    if n == "hf":
        # Imported lazily so torch / transformers are not required unless
        # the user opts in. Phase 2.3 will add hf_provider.py.
        try:
            from .hf_provider import HuggingFaceProvider  # type: ignore
        except ImportError as exc:
            raise ProviderUnavailableError(
                f"HuggingFace provider not available yet (phase 2.3). "
                f"Underlying error: {exc}"
            ) from exc
        return HuggingFaceProvider()
    raise ProviderUnavailableError(
        f"Unknown LLM_PROVIDER={name!r}. Expected one of: openai | hf | mock."
    )


def _build_embedder(name: str) -> EmbeddingProvider:
    n = name.strip().lower()
    if n == "auto":
        # Follow LLM_PROVIDER by default, but OpenAI embedding if the chat
        # provider is 'mock' (so tests using mock chat still get stable dim).
        chat = settings.llm_provider.lower()
        if chat == "mock":
            n = "mock"
        else:
            n = chat

    if n == "openai":
        from .openai_embedder import OpenAIEmbedder  # pragma: no cover

        return OpenAIEmbedder()
    if n == "mock":
        from .mock_embedder import MockEmbedder  # pragma: no cover

        return MockEmbedder()
    if n == "hf":
        try:
            from .hf_embedder import HuggingFaceEmbedder  # type: ignore
        except ImportError as exc:
            raise ProviderUnavailableError(
                f"HuggingFace embedder not available yet (phase 2.4). "
                f"Underlying error: {exc}"
            ) from exc
        return HuggingFaceEmbedder()
    raise ProviderUnavailableError(
        f"Unknown embedding provider {name!r}. Expected one of: "
        f"auto | openai | hf | mock."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    """Return a cached LLMProvider for the given name (default: settings)."""
    key = (name or settings.llm_provider).strip().lower()
    with _lock:
        if key in _llm_cache:
            return _llm_cache[key]
        prov = _build_llm(key)
        _llm_cache[key] = prov
        _log.info(
            "Built LLMProvider name=%s model_id=%s device=%s",
            prov.name,
            prov.model_id,
            prov.device,
        )
        return prov


def get_embedding_provider(name: Optional[str] = None) -> EmbeddingProvider:
    """Return a cached EmbeddingProvider for the given name (default: settings)."""
    key = (name or settings.embedding_provider).strip().lower()
    with _lock:
        if key in _embed_cache:
            return _embed_cache[key]
        prov = _build_embedder(key)
        _embed_cache[key] = prov
        _log.info(
            "Built EmbeddingProvider name=%s model_id=%s dim=%d",
            prov.name,
            prov.model_id,
            prov.dimension,
        )
        return prov


def reset_caches() -> None:
    """Drop cached providers (useful in tests after monkey-patching settings)."""
    with _lock:
        _llm_cache.clear()
        _embed_cache.clear()
