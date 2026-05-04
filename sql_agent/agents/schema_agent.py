"""schema_node — foundation: reflect DB schema, halt graph on failure."""

from __future__ import annotations

from sql_agent.config import get_logger
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import SchemaDiscoveryInput, SchemaDiscoveryTool, ToolExecutionError


_log = get_logger("agent.schema")
_tool = SchemaDiscoveryTool()


def schema_node(state: AgentState) -> AgentState:
    try:
        out = _tool.run(SchemaDiscoveryInput(force_refresh=False))
        _log.info(
            "schema discovered",
            extra={
                "table_count": len(out.db_schema.tables),
                "tables": list(out.db_schema.tables.keys()),
            },
        )
        return {"schema": out.db_schema.model_dump()}
    except ToolExecutionError as exc:
        return {"error": f"Schema discovery failed: {exc}", "success": False}
