"""LangGraph StateGraph wiring all agent nodes together."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState, ChatMessage, empty_state

from .analysis_agent import analysis_node
from .data_agent import clean_node, fetch_node, preview_node
from .datetime_agent import datetime_node
from .intent_agent import intent_node
from .memory_agent import (
    memory_recall_node,
    penalty_node,
    reward_node,
    summarize_node,
)
from .param_builder_agent import param_builder_node
from .schema_agent import schema_node
from .tool_selection_agent import tool_selection_node
from .validation_agent import MAX_RETRIES, validation_node, validation_router
from .visualization_agent import viz_node


_log = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def _halt_if_error(state: AgentState, next_node: str) -> str:
    """If the state already carries an error, jump to penalty + end."""
    if state.get("error"):
        return "penalty"
    return next_node


def _after_schema(state: AgentState) -> str:
    return _halt_if_error(state, "memory_recall")


def _after_preview(state: AgentState) -> str:
    return _halt_if_error(state, "fetch")


def _after_fetch(state: AgentState) -> str:
    return _halt_if_error(state, "clean")


def _after_validation(state: AgentState) -> str:
    verdict = validation_router(state)
    if verdict == "ok":
        return "tool_select"
    if verdict == "retry":
        return "param_builder"
    # abort: record penalty, stop.
    return "penalty"


def _after_param_builder(state: AgentState) -> str:
    return _halt_if_error(state, "validate")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("schema", schema_node)
    g.add_node("memory_recall", memory_recall_node)
    g.add_node("intent", intent_node)
    g.add_node("datetime", datetime_node)
    g.add_node("param_builder", param_builder_node)
    g.add_node("validate", validation_node)
    g.add_node("tool_select", tool_selection_node)
    g.add_node("preview", preview_node)
    g.add_node("fetch", fetch_node)
    g.add_node("clean", clean_node)
    g.add_node("analysis", analysis_node)
    g.add_node("viz", viz_node)
    g.add_node("reward", reward_node)
    g.add_node("penalty", penalty_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("schema")

    g.add_conditional_edges(
        "schema", _after_schema, {"memory_recall": "memory_recall", "penalty": "penalty"}
    )
    g.add_edge("memory_recall", "intent")
    g.add_edge("intent", "datetime")
    g.add_edge("datetime", "param_builder")
    g.add_conditional_edges(
        "param_builder",
        _after_param_builder,
        {"validate": "validate", "penalty": "penalty"},
    )
    g.add_conditional_edges(
        "validate",
        _after_validation,
        {
            "tool_select": "tool_select",
            "param_builder": "param_builder",  # retry
            "penalty": "penalty",
        },
    )
    g.add_edge("tool_select", "preview")
    g.add_conditional_edges(
        "preview", _after_preview, {"fetch": "fetch", "penalty": "penalty"}
    )
    g.add_conditional_edges(
        "fetch", _after_fetch, {"clean": "clean", "penalty": "penalty"}
    )
    g.add_edge("clean", "analysis")
    g.add_edge("analysis", "viz")
    g.add_edge("viz", "reward")
    g.add_edge("reward", "summarize")
    g.add_edge("penalty", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Plan-only subgraph (phase 8.5 — closes a phase-1 deferred item).
#
# Purpose: the gRPC ``GenerateSQL`` RPC only wants the planned query
# (tool + FetchParams) — it does NOT need to hit the database, clean, run
# analysis, or update memory. Previously `GenerateSQL` ran the whole
# pipeline and stripped the response; this subgraph short-circuits at
# ``validate`` so the DB roundtrip + analysis + FAISS write are skipped,
# cutting end-to-end latency by ~half in the common case.
#
# Graph: schema -> memory_recall -> intent -> datetime -> param_builder
#                                                  |
#                                              validate -> END
# On error, we still route to `penalty` -> `summarize` -> END so the
# tracker sees the failure via the same hooks as the full pipeline.
# ---------------------------------------------------------------------------


def _plan_after_validation(state: AgentState) -> str:
    verdict = validation_router(state)
    if verdict == "ok":
        # In the plan-only subgraph, a successful validation routes to
        # tool_select (so the final plan tells us which execution tool
        # would run), then straight to END via the "plan_done" shortcut.
        return "tool_select"
    if verdict == "retry":
        return "param_builder"
    return "penalty"


@lru_cache(maxsize=1)
def build_plan_graph():
    """Plan-only subgraph. Mirrors build_graph() up to tool_select, then
    exits. Cached per-process like build_graph()."""
    g = StateGraph(AgentState)

    g.add_node("schema", schema_node)
    g.add_node("memory_recall", memory_recall_node)
    g.add_node("intent", intent_node)
    g.add_node("datetime", datetime_node)
    g.add_node("param_builder", param_builder_node)
    g.add_node("validate", validation_node)
    g.add_node("tool_select", tool_selection_node)
    g.add_node("penalty", penalty_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("schema")

    g.add_conditional_edges(
        "schema", _after_schema, {"memory_recall": "memory_recall", "penalty": "penalty"}
    )
    g.add_edge("memory_recall", "intent")
    g.add_edge("intent", "datetime")
    g.add_edge("datetime", "param_builder")
    g.add_conditional_edges(
        "param_builder",
        _after_param_builder,
        {"validate": "validate", "penalty": "penalty"},
    )
    g.add_conditional_edges(
        "validate",
        _plan_after_validation,
        {
            "tool_select": "tool_select",
            "param_builder": "param_builder",
            "penalty": "penalty",
        },
    )
    g.add_edge("tool_select", END)
    g.add_edge("penalty", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Public entrypoint used by UI + CLI
# ---------------------------------------------------------------------------


def run_turn(
    user_query: str,
    *,
    session_id: str,
    prior_messages: Optional[List[ChatMessage]] = None,
    memory_summary: Optional[str] = None,
) -> AgentState:
    """Run one turn of the multi-agent graph.

    Parameters
    ----------
    user_query : str
        The latest user message.
    session_id : str
        UUID identifying the chat session (for history + memory).
    prior_messages : list of ChatMessage, optional
        Full prior turn history (the user message about to be processed should
        NOT be included — we add it here).
    memory_summary : str, optional
        Existing conversation summary (older than 6 turns).
    """
    # Phase 4: wrap the whole turn in a tracking span.
    # Phase 8.3: also bind a token-usage accumulator so providers can
    # report per-call input/output token counts that the tracker picks up.
    # Phase 8.6: additionally open an OpenTelemetry span (no-op when OTEL
    # is disabled).
    from sql_agent.observability.tracing import get_tracer
    from sql_agent.request_context import token_usage_scope
    from sql_agent.tracking import get_tracker

    tracker = get_tracker()
    otel_tracer = get_tracer()

    graph = build_graph()

    state: AgentState = empty_state(user_query=user_query, session_id=session_id)
    state["messages"] = list(prior_messages or []) + [
        ChatMessage(role="user", content=user_query)
    ]
    state["memory_summary"] = memory_summary

    with otel_tracer.start_as_current_span("sql_agent.run_turn") as span:
        span.set_attribute("sql_agent.session_id", session_id or "")
        span.set_attribute("sql_agent.query_len", len(user_query or ""))
        with token_usage_scope():
            handle = tracker.start(session_id=session_id, user_query=user_query)
            try:
                final: Dict[str, Any] = graph.invoke(state)
            except BaseException as exc:
                span.record_exception(exc)
                try:
                    tracker.finish_error(handle, exc)
                except Exception as track_exc:  # pragma: no cover
                    _log.warning("tracker.finish_error itself failed: %s", track_exc)
                raise

            try:
                tracker.finish(handle, dict(final))
            except Exception as track_exc:  # pragma: no cover
                _log.warning("tracker.finish failed: %s", track_exc)

            span.set_attribute("sql_agent.tool_used", final.get("tool_used") or "")
            span.set_attribute(
                "sql_agent.success",
                bool(final.get("success") and not final.get("error")),
            )
    return final  # type: ignore[return-value]


def plan_turn(
    user_query: str,
    *,
    session_id: str,
    prior_messages: Optional[List[ChatMessage]] = None,
    memory_summary: Optional[str] = None,
) -> AgentState:
    """Run only the planning portion of the pipeline.

    Produces the same ``tool_used`` + ``parameters`` + ``param_reasoning``
    as ``run_turn`` would, but skips the DB roundtrip, data cleaning,
    analysis, visualization, and FAISS memory writes. Used by the gRPC
    ``GenerateSQL`` RPC (phase 8.5).

    Failure modes:
        * Schema discovery errors, intent errors, param-builder errors,
          validation errors after MAX_RETRIES retries are all routed to
          ``penalty`` → ``summarize`` → END, then returned with ``error``
          set. Identical to ``run_turn``'s behavior up to that point.
    """
    from sql_agent.request_context import token_usage_scope
    from sql_agent.tracking import get_tracker

    tracker = get_tracker()
    graph = build_plan_graph()

    state: AgentState = empty_state(user_query=user_query, session_id=session_id)
    state["messages"] = list(prior_messages or []) + [
        ChatMessage(role="user", content=user_query)
    ]
    state["memory_summary"] = memory_summary

    with token_usage_scope():
        handle = tracker.start(
            session_id=session_id, user_query=f"[plan-only] {user_query}"
        )
        try:
            final: Dict[str, Any] = graph.invoke(state)
        except BaseException as exc:
            try:
                tracker.finish_error(handle, exc)
            except Exception as track_exc:  # pragma: no cover
                _log.warning("tracker.finish_error itself failed: %s", track_exc)
            raise

        try:
            tracker.finish(handle, dict(final))
        except Exception as track_exc:  # pragma: no cover
            _log.warning("tracker.finish failed: %s", track_exc)
    return final  # type: ignore[return-value]
