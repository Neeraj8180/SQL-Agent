"""tool_selection_node — deterministic (non-LLM) choice of execution tool.

Maps intent.output_type + parameters to the best single tool:
  count         → CountTool
  list_unique   → ListingTool
  others        → DataFetchTool (with preview-first flow handled elsewhere)
"""

from __future__ import annotations

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState


_log = get_logger("agent.tool_select")


def tool_selection_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or {}
    output_type = (intent.get("output_type") or "table").lower()

    if output_type == "count":
        tool = "count"
    elif output_type == "list_unique":
        tool = "listing"
    else:
        tool = "data_fetch"

    _log.info(
        "tool selected",
        extra={"tool": tool, "output_type": output_type},
    )
    return {"tool_used": tool}
