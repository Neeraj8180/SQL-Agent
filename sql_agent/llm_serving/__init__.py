"""LLM serving abstraction (phase 2).

Goals:
    * Decouple agent code from any specific LLM vendor.
    * Support OpenAI (default), HuggingFace (local GPU/CPU), and a deterministic
      Mock provider for tests and demos.
    * Expose the minimum duck-typed surface the existing agents already rely on
      (``.with_structured_output(Cls).invoke(msgs)`` and ``.invoke(msgs)``),
      so no agent code needs to change.

Public API is resolved lazily so that importing this package never forces a
torch or transformers import on pure-OpenAI deployments.
"""

from __future__ import annotations

__all__ = [
    "ChatModel",
    "StructuredInvoker",
    "LLMProvider",
    "EmbeddingProvider",
    "get_llm_provider",
    "get_embedding_provider",
    "detect_device",
    "log_execution_mode",
]


def __getattr__(name: str):
    if name in {"ChatModel", "StructuredInvoker", "LLMProvider", "EmbeddingProvider"}:
        from . import base

        return getattr(base, name)
    if name in {"detect_device", "log_execution_mode"}:
        from . import hardware

        return getattr(hardware, name)
    if name in {"get_llm_provider", "get_embedding_provider"}:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
