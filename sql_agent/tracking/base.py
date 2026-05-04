"""Tracking protocols + a shared state-to-summary helper.

The helper is factored out because every concrete tracker wants the same
normalized view of an ``AgentState`` dict (same param/metric/tag shapes),
and duplicating that logic would quickly drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TurnHandle(Protocol):
    """Opaque per-turn handle returned by ``start`` and passed to ``finish``.

    Implementations typically carry start-time, a run id, and the session id.
    The caller MUST NOT inspect these fields — treat the handle as opaque.
    """

    start_time_ns: int


@runtime_checkable
class TurnTracker(Protocol):
    """Lifecycle contract for one invocation of ``run_turn``."""

    name: str

    def start(self, *, session_id: str, user_query: str) -> TurnHandle: ...

    def finish(self, handle: TurnHandle, final_state: Dict[str, Any]) -> None: ...

    def finish_error(
        self, handle: TurnHandle, exc: BaseException
    ) -> None: ...


# ---------------------------------------------------------------------------
# Shared state summarizer
# ---------------------------------------------------------------------------


def _classify_error(state: Dict[str, Any]) -> str:
    """Bucket the state's error message into a coarse tag value.

    Coarse buckets keep the MLflow "tags" sidebar readable. The full error
    string is still captured as an artifact by each tracker.
    """
    err = state.get("error") or ""
    if not err:
        return ""
    err_lc = err.lower()
    if "parameter schema" in err_lc or "validation" in err_lc:
        return "validation_error"
    if "schema discovery" in err_lc:
        return "schema_error"
    if "preview" in err_lc:
        return "preview_error"
    if "data fetch" in err_lc or "fetch failed" in err_lc:
        return "fetch_error"
    if "intent extraction" in err_lc:
        return "intent_error"
    if "parameter building" in err_lc:
        return "param_builder_error"
    if "analysis" in err_lc:
        return "analysis_error"
    return "other_error"


def summarize_state(
    state: Dict[str, Any],
    *,
    user_query: str,
    session_id: str,
    query_max_chars: int = 500,
) -> Dict[str, Any]:
    """Produce a JSON-safe dict of params/metrics/tags/artifacts for a turn.

    Structure:
        {
          "params":     {str: str},     # small, categorical / textual
          "metrics":    {str: float},   # numeric, plot-able
          "tags":       {str: str},
          "artifacts":  {str: Any},     # logged as JSON objects
        }
    """
    from sql_agent.config import settings  # local import: avoid cycles at module import

    # Best-effort provider introspection via registry (no construction if
    # the singleton already exists; we just report what we *would* use).
    llm_name = llm_model = llm_device = ""
    emb_name = emb_model = ""
    emb_dim = 0
    try:
        from sql_agent.llm_serving import registry as _reg

        # Peek into the cache without forcing construction.
        with _reg._lock:  # type: ignore[attr-defined]
            llm_entry = _reg._llm_cache.get(settings.llm_provider)  # type: ignore[attr-defined]
            emb_key = (
                settings.embedding_provider
                if settings.embedding_provider != "auto"
                else settings.llm_provider
            )
            emb_entry = _reg._embed_cache.get(emb_key)  # type: ignore[attr-defined]
        if llm_entry is not None:
            llm_name = getattr(llm_entry, "name", "") or ""
            llm_model = getattr(llm_entry, "model_id", "") or ""
            llm_device = getattr(llm_entry, "device", "") or ""
        else:
            llm_name = settings.llm_provider
        if emb_entry is not None:
            emb_name = getattr(emb_entry, "name", "") or ""
            emb_model = getattr(emb_entry, "model_id", "") or ""
            emb_dim = int(getattr(emb_entry, "dimension", 0) or 0)
        else:
            emb_name = (
                settings.embedding_provider
                if settings.embedding_provider != "auto"
                else settings.llm_provider
            )
    except Exception:  # pragma: no cover — defensive, never blocks tracking
        pass

    rows: List[Dict[str, Any]] = (
        state.get("data_cleaned") or state.get("data") or []
    )
    insights_text = str(state.get("insights") or "")
    param_dict = state.get("parameters") or {}
    validation_errors = state.get("validation_errors") or []

    params: Dict[str, str] = {
        "user_query": (user_query or "")[:query_max_chars],
        "session_id": session_id,
        "llm_provider": llm_name,
        "llm_model_id": llm_model,
        "llm_device": llm_device,
        "embedding_provider": emb_name,
        "embedding_model_id": emb_model,
        "embedding_dim": str(emb_dim),
        "routing_enabled": str(bool(settings.llm_routing_enabled)).lower(),
        "routing_weights": (
            settings.llm_routing_weights if settings.llm_routing_enabled else ""
        ),
    }

    # Phase 8.3: aggregate per-call LLM token usage if the orchestrator set
    # up a token_usage_scope. Safe no-op if not set.
    input_tokens = output_tokens = 0
    try:
        from sql_agent.request_context import token_usage_var

        usages = token_usage_var.get() or []
        input_tokens = sum(int(u.get("input_tokens", 0)) for u in usages)
        output_tokens = sum(int(u.get("output_tokens", 0)) for u in usages)
    except Exception:  # pragma: no cover — defensive
        pass

    metrics: Dict[str, float] = {
        "row_count": float(len(rows)),
        "retry_count": float(state.get("retry_count") or 0),
        "validation_error_count": float(len(validation_errors)),
        "insights_word_count": float(len(insights_text.split()) if insights_text else 0),
        "success": 1.0 if state.get("success") and not state.get("error") else 0.0,
        "invalid_sql_rate_proxy": 1.0 if validation_errors else 0.0,
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(input_tokens + output_tokens),
    }

    tool_used = str(state.get("tool_used") or "")
    error_tag = _classify_error(state)
    tags: Dict[str, str] = {
        "tool_used": tool_used,
        "error_type": error_tag,
    }

    artifacts: Dict[str, Any] = {
        "parameters.json": param_dict,
    }
    if rows:
        artifacts["data_head.json"] = rows[:20]
    if state.get("error"):
        artifacts["error.txt"] = str(state.get("error"))

    return {
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "artifacts": artifacts,
    }
