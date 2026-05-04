"""validation_node — delegates to QueryValidationTool and tracks retry count."""

from __future__ import annotations

from sql_agent.config import get_logger
from sql_agent.models import FetchParams, SchemaInfo
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import (
    QueryValidationInput,
    QueryValidationTool,
    ToolExecutionError,
)


_log = get_logger("agent.validation")
_tool = QueryValidationTool()

MAX_RETRIES = 2


def validation_node(state: AgentState) -> AgentState:
    params_dict = state.get("parameters") or {}
    schema_dict = state.get("schema") or {}
    if not params_dict:
        return {"validation_errors": ["No parameters produced."], "retry_count": state.get("retry_count", 0) + 1}

    try:
        params = FetchParams(**params_dict)
        schema = SchemaInfo(**schema_dict)
    except Exception as exc:
        return {
            "validation_errors": [f"Parameter schema invalid: {exc}"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        out = _tool.run(QueryValidationInput(params=params, db_schema=schema))
    except ToolExecutionError as exc:
        return {
            "validation_errors": [str(exc)],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    if out.is_valid:
        _log.info("params valid (%d warning(s))", len(out.warnings))
        return {"validation_errors": []}

    _log.warning("validation errors: %s", out.errors)
    return {
        "validation_errors": out.errors,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def validation_router(state: AgentState) -> str:
    """Edge selector: 'ok' | 'retry' | 'abort'."""
    if not state.get("validation_errors"):
        return "ok"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "abort"
    return "retry"
