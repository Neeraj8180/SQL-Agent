"""datetime_node — deterministic time resolution via DateTimeHandlingTool."""

from __future__ import annotations

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import DateTimeHandlingTool, DateTimeInput, ToolExecutionError


_log = get_logger("agent.datetime")
_tool = DateTimeHandlingTool()


def datetime_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or {}
    tr = intent.get("time_range") or {}
    expr = tr.get("expression") if isinstance(tr, dict) else None
    gran = tr.get("granularity") if isinstance(tr, dict) else None
    if not expr and not gran:
        return {"datetime_resolved": None}

    try:
        out = _tool.run(DateTimeInput(expression=expr, granularity=gran))
    except ToolExecutionError as exc:
        _log.warning("datetime parse failed: %s", exc)
        return {"datetime_resolved": None}

    resolved = out.model_dump()
    _log.info("datetime resolved: %s", resolved)
    return {"datetime_resolved": resolved}
