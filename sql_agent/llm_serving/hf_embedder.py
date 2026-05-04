"""HuggingFace embedding provider via sentence-transformers.

Default model: ``sentence-transformers/all-MiniLM-L6-v2`` — 384-dim, CPU-fast,
widely used. Produces embeddings compatible with the FAISS IndexFlatIP inner-
product search after per-vector L2 normalization (done in ``_FaissStore``).

sentence-transformers is imported lazily so OpenAI-only installs are never
forced to pull it in.
"""

from __future__ import annotations

from typing import List, Optional

from sql_agent.config import get_logger, settings

from .base import ProviderUnavailableError
from .hardware import detect_device, log_execution_mode


_log = get_logger("llm_serving.hf_embed")


class HuggingFaceEmbedder:
    name: str = "hf"

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderUnavailableError(
                "HuggingFaceEmbedder requires sentence-transformers. "
                "Install with: pip install -r requirements-llm-local.txt "
                f"(underlying error: {exc})"
            ) from exc

        self.model_id = model_id or settings.hf_embedding_model
        self.device = device or detect_device()
        cache = cache_dir or (settings.hf_cache_dir or None)

        log_execution_mode(self.device, f"embed:{self.model_id}")
        load_kwargs = {}
        if cache:
            load_kwargs["cache_folder"] = cache
        self._model = SentenceTransformer(
            self.model_id, device=self.device, **load_kwargs
        )
        # The API was renamed in sentence-transformers 5.x; keep a fallback
        # for older versions.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self.dimension = int((get_dim() if get_dim else 0) or 0)
        if self.dimension <= 0:  # pragma: no cover — defensive
            raise ProviderUnavailableError(
                f"Could not determine embedding dimension for {self.model_id}"
            )
        _log.info(
            "HuggingFaceEmbedder ready: model_id=%s dim=%d device=%s",
            self.model_id,
            self.dimension,
            self.device,
        )

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode([text or ""], show_progress_bar=False)
        return [float(x) for x in vec[0]]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(list(texts), show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]
