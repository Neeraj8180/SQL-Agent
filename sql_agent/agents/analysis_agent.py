"""analysis_node — statistical analysis on cleaned data."""

from __future__ import annotations

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import (
    StatisticalAnalysisInput,
    StatisticalAnalysisTool,
    ToolExecutionError,
)


_log = get_logger("agent.analysis")
_tool = StatisticalAnalysisTool()


def analysis_node(state: AgentState) -> AgentState:
    rows = state.get("data_cleaned") or state.get("data") or []
    if not rows:
        return {
            "analysis": {"descriptive": {}, "correlation": {}, "grouped": {}, "insights": ["No data."]},
            "insights": "No data matched the query.",
        }

    params = state.get("parameters") or {}
    numeric_cols = [a.get("alias") for a in (params.get("aggregations") or []) if a.get("alias")]

    # Pick a grouping column (first dimension / group_by).
    group_by = None
    if params.get("group_by"):
        g = params["group_by"][0]
        group_by = g.split(".")[-1] if "." in g else g

    try:
        out = _tool.run(
            StatisticalAnalysisInput(
                rows=rows,
                numeric_columns=[c for c in numeric_cols if c],
                group_by=group_by,
            )
        )
    except ToolExecutionError as exc:
        return {"error": f"Analysis failed: {exc}"}

    insights_text = "\n".join(f"- {i}" for i in out.insights) if out.insights else ""
    _log.info("analysis: %d insights", len(out.insights))
    return {"analysis": out.model_dump(), "insights": insights_text}
