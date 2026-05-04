"""Protocols defining the LLM serving contract.

These protocols are intentionally minimal — they describe *only* what the
existing agent code already uses. Any conforming provider can be plugged in
without touching the agents.

Key invariants:
    - ``ChatModel.with_structured_output(cls).invoke(messages)`` returns an
      instance of ``cls`` (a pydantic BaseModel subclass).
    - ``ChatModel.invoke(messages)`` returns an object with a ``.content``
      string attribute (used by ``summarize_node``).
    - Providers are responsible for adapting their native API to this shape.
"""

from __future__ import annotations

from typing import Any, List, Protocol, Type, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class HasContent(Protocol):
    content: Any


@runtime_checkable
class StructuredInvoker(Protocol):
    """An invocable returning a pydantic model instance."""

    def invoke(self, messages: List[Any]) -> BaseModel: ...


@runtime_checkable
class ChatModel(Protocol):
    """Minimal chat-model surface used across the agent graph.

    Mirrors the fragment of ``langchain_openai.ChatOpenAI`` that the agents
    actually call. A provider may return any object conforming to this
    protocol (duck-typed; ``@runtime_checkable`` permits ``isinstance``).
    """

    def with_structured_output(self, model_cls: Type[BaseModel]) -> StructuredInvoker: ...

    def invoke(self, messages: List[Any]) -> HasContent: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Factory for chat-capable models, one per (provider, device, model)."""

    name: str                 # "openai" | "hf" | "mock"
    model_id: str             # human-readable identifier for logs / MLflow
    device: str               # "cuda" | "mps" | "cpu" | "remote"

    def chat_model(self, temperature: float = 0.0) -> ChatModel: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Vendor-neutral embedding contract."""

    name: str
    model_id: str
    dimension: int            # enables dim-namespaced FAISS indices

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class ProviderUnavailableError(RuntimeError):
    """Raised when a requested provider cannot be constructed (missing deps,
    network, credentials, etc.). Caller may fall back to a simpler provider."""
