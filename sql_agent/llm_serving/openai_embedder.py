"""OpenAI embedding provider — preserves the existing 1536-dim behavior.

Mirrors the logic in the original ``sql_agent/services/llm.py::embed_text`` so
callers that were using OpenAI embeddings see no behavior change.
"""

from __future__ import annotations

from typing import List, Optional

from sql_agent.config import get_logger, settings

from .base import ProviderUnavailableError


_log = get_logger("llm_serving.openai_embed")


# Dimensions for supported OpenAI embedding models (static lookup; avoids a
# blocking round-trip just to discover dim).
_OPENAI_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    name: str = "openai"

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or settings.openai_embedding_model
        self.dimension = _OPENAI_EMBED_DIMS.get(self.model_id, 1536)
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenAIEmbedder requires OPENAI_API_KEY to be set."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailableError(
                f"openai package not installed: {exc}"
            ) from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> List[float]:
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model_id, input=text)
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model_id, input=texts)
        return [list(d.embedding) for d in resp.data]
