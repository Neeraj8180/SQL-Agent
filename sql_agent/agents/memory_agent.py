"""Memory nodes: recall (pre-plan), reward / penalty (post-execute),
and summarize (compress chat history).

This implements the RL-style dual memory:
  * penalty rules  : "avoid this approach — it failed previously".
  * reward rules   : "this approach worked on a similar query — prefer it".
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState
from sql_agent.services.llm import get_chat_model
from sql_agent.services.memory_manager import get_memory_manager


_log = get_logger("agent.memory")
_SUMMARY_TRIGGER = 6


def memory_recall_node(state: AgentState) -> AgentState:
    """Retrieve top-k reward/penalty rules similar to the current query."""
    mm = get_memory_manager()
    rules = mm.recall(state["user_query"], k_reward=3, k_penalty=3)
    _log.info("recalled %d memory rule(s)", len(rules))
    return {"memory_rules": rules}


def reward_node(state: AgentState) -> AgentState:
    """Store a successful (query → params → tool) triple in the reward index."""
    if state.get("error"):
        return {}
    params = state.get("parameters") or {}
    tool = state.get("tool_used") or "data_fetch"
    reasoning = state.get("param_reasoning") or ""
    try:
        get_memory_manager().record_reward(
            state["user_query"],
            parameters=params,
            tool_used=tool,
            reasoning=reasoning,
        )
        _log.info("recorded reward for query: %s", state["user_query"][:60])
    except Exception as exc:
        _log.warning("reward write failed: %s", exc)
    return {"success": True}


def penalty_node(state: AgentState) -> AgentState:
    """Store a failure reason keyed by the query in the penalty index."""
    reason = state.get("error") or "; ".join(state.get("validation_errors") or [])
    if not reason:
        reason = "unknown failure"
    params = state.get("parameters") or {}
    try:
        get_memory_manager().record_penalty(
            state["user_query"], reason=reason, parameters=params
        )
        _log.info("recorded penalty for query: %s (%s)", state["user_query"][:60], reason[:80])
    except Exception as exc:
        _log.warning("penalty write failed: %s", exc)
    return {}


def summarize_node(state: AgentState) -> AgentState:
    """Compress old turns (beyond the last 6) into `memory_summary`."""
    messages = state.get("messages") or []
    if len(messages) <= _SUMMARY_TRIGGER:
        return {}

    old = messages[:-_SUMMARY_TRIGGER]
    prior_summary = state.get("memory_summary") or ""

    text_blocks: List[str] = []
    for m in old:
        role = m.get("role", "user")
        content = m.get("content", "")
        text_blocks.append(f"[{role}] {content}")
    history_text = "\n".join(text_blocks)

    try:
        llm = get_chat_model(temperature=0.0)
        resp = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You compress conversations into a concise running "
                        "summary. Preserve: user intents, tables/columns "
                        "referenced, successful tool paths, filters, and time "
                        "windows. Output a single paragraph, <=150 words."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Existing summary:\n{prior_summary}\n\n"
                        f"Older turns to fold in:\n{history_text}"
                    )
                ),
            ]
        )
        summary = str(resp.content).strip()
    except Exception as exc:
        _log.warning("summarize failed: %s", exc)
        return {}

    return {"memory_summary": summary}


def format_rules_for_prompt(rules: List[Dict[str, Any]]) -> str:
    """Render recalled memory rules into a prompt-ready block."""
    if not rules:
        return "(no prior rules)"
    lines: List[str] = []
    rewards = [r for r in rules if r.get("kind") == "reward"]
    penalties = [r for r in rules if r.get("kind") == "penalty"]
    if rewards:
        lines.append("PROVEN SUCCESSFUL PATTERNS (prefer these when similar):")
        for r in rewards:
            lines.append(
                f"  - query: {r.get('query')!r}\n"
                f"    tool: {r.get('tool_used')}\n"
                f"    parameters: {r.get('parameters')}\n"
                f"    reasoning: {r.get('reasoning') or '(n/a)'}\n"
                f"    similarity: {r.get('score', 0):.2f}"
            )
    if penalties:
        lines.append("\nKNOWN FAILURES (avoid repeating):")
        for p in penalties:
            lines.append(
                f"  - query: {p.get('query')!r}\n"
                f"    reason: {p.get('reason')}\n"
                f"    bad parameters: {p.get('parameters')}\n"
                f"    similarity: {p.get('score', 0):.2f}"
            )
    return "\n".join(lines)
