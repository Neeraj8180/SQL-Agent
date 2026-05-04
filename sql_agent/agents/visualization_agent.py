"""viz_node — chart generation via VisualizationTool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import (
    ToolExecutionError,
    VisualizationInput,
    VisualizationTool,
)


_log = get_logger("agent.viz")
_tool = VisualizationTool()


def viz_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or {}
    if not intent.get("visualize", True):
        return {"visualization": None}

    rows = state.get("data_cleaned") or state.get("data") or []
    if not rows:
        return {"visualization": None}

    params = state.get("parameters") or {}
    x, y, group, kind = _axes(params, rows)

    try:
        out = _tool.run(
            VisualizationInput(
                rows=rows,
                chart_kind=kind,
                x=x,
                y=y,
                group=group,
                title=None,
            )
        )
    except ToolExecutionError as exc:
        _log.warning("viz failed: %s", exc)
        return {"visualization": None}
    return {"visualization": out.image_base64}


def _axes(params: Dict[str, Any], rows):
    columns = list(rows[0].keys()) if rows else []
    aggs = params.get("aggregations") or []
    tg = params.get("time_grouping")
    gb = params.get("group_by") or []

    y: Optional[str] = None
    if aggs:
        y = aggs[0].get("alias")

    x: Optional[str] = None
    if tg and isinstance(tg, dict) and tg.get("alias") in columns:
        x = tg["alias"]
    elif gb:
        cand = gb[0]
        cand = cand.split(".")[-1] if "." in cand else cand
        if cand in columns:
            x = cand

    group: Optional[str] = None
    if tg and gb:
        # If both bucket AND dimension present, treat dimension as series group.
        for g in gb:
            gc = g.split(".")[-1] if "." in g else g
            if gc in columns and gc != x:
                group = gc
                break

    kind: str = "auto"
    if tg:
        kind = "line"
    elif x and y and x in columns and y in columns:
        kind = "bar"
    return x, y, group, kind  # type: ignore[return-value]
