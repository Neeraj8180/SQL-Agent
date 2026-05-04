"""Public LLM service API — thin delegation layer over the provider registry.

Historical note (phase 2): this module previously contained direct OpenAI
bindings. It now delegates to ``sql_agent.llm_serving.registry``, which picks
an implementation based on the ``LLM_PROVIDER`` / ``EMBEDDING_PROVIDER``
settings.

Why keep this module at all?
    * Agents and services import ``get_chat_model`` / ``embed_text`` / 
      ``embed_texts`` by name from this module. Changing those import sites
      would be a sweeping edit — the exact thing phase 2 is designed to
      avoid. Instead, we keep this module as the stable public surface.
    * With ``LLM_PROVIDER=openai`` (the default), the call path is
      byte-for-byte identical to pre-phase-2 behavior: ``OpenAIProvider``
      wraps ``langchain_openai.ChatOpenAI`` with the same constructor args
      this module used to pass directly.
"""

from __future__ import annotations

from typing import Any, List, Optional

from sql_agent.config import get_logger, settings
from sql_agent.llm_serving import registry


__all__ = ["get_chat_model", "embed_text", "embed_texts"]


_log = get_logger("services.llm")


def _route_provider_name() -> Optional[str]:
    """Ask the router to pick a provider, if routing is enabled.

    Returns ``None`` when routing is disabled OR when the router itself
    errors — in either case the caller should fall back to the default
    provider selected by ``settings.llm_provider``.
    """
    if not settings.llm_routing_enabled:
        return None
    try:
        from sql_agent.request_context import session_id_var
        from sql_agent.routing.router import get_router

        return get_router().route(session_id_var.get())
    except Exception as exc:
        _log.warning("Routing failed; using default provider: %s", exc)
        return None


def get_chat_model(temperature: float = 0.0) -> Any:
    """Return a chat model from the configured LLM provider.

    When ``LLM_ROUTING_ENABLED=true``, the provider name is chosen by the
    router (phase 3). Otherwise the default (``settings.llm_provider``) is
    used, which preserves post-phase-2 behavior byte-for-byte.

    Phase 8.7: when provider construction fails, report the failure to the
    router so the circuit breaker (if enabled) can open that provider.
    """
    provider_name = _route_provider_name()
    try:
        chat = registry.get_llm_provider(provider_name).chat_model(temperature)
    except Exception as exc:
        # Only report to the breaker if routing was actually used; there's
        # no value reporting failure of the "default" provider on a single-
        # provider deployment.
        if provider_name and settings.llm_routing_enabled:
            try:
                from sql_agent.routing.router import get_router

                get_router().report_failure(provider_name)
                _log.warning(
                    "get_chat_model: reported failure for provider %s",
                    provider_name,
                )
            except Exception:  # pragma: no cover — defensive
                pass
        raise
    else:
        if provider_name and settings.llm_routing_enabled:
            try:
                from sql_agent.routing.router import get_router

                get_router().report_success(provider_name)
            except Exception:  # pragma: no cover
                pass
    return chat


def embed_text(text: str) -> List[float]:
    """Return a single embedding vector from the configured embedder."""
    return registry.get_embedding_provider().embed(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a batch of embedding vectors from the configured embedder."""
    return registry.get_embedding_provider().embed_batch(texts)
