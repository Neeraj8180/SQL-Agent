"""param_builder_node — convert Intent → FetchParams via structured LLM output.

This is the ONLY place the LLM produces machine-actionable query parameters.
It never writes SQL. The output is a ``FetchParams`` pydantic model that the
DataFetchTool lowers into SQLAlchemy expressions.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sql_agent.config import get_logger
from sql_agent.models import FetchParams
from sql_agent.models.graph_state import AgentState
from sql_agent.services.llm import get_chat_model

from .memory_agent import format_rules_for_prompt


_log = get_logger("agent.param_builder")


class ParamPlan(BaseModel):
    """LLM output: FetchParams plus a short reasoning blurb."""

    reasoning: str = Field(
        ...,
        description="Brief (≤60 words) explanation of why these params were chosen.",
    )
    params: FetchParams


_SYS = """You convert a user's analytical intent into STRUCTURED tool
parameters for the DataFetchTool. You NEVER write SQL.

Rules:
- Use ONLY table/column names that exist in the provided schema.
- Always set a reasonable limit (default 100, max 10000).
- For time series, prefer: aggregations + group_by (dimension) + time_grouping
  on the relevant datetime column.
- For count questions: use a single aggregation with func='count' and column='*'.
- Qualify columns as 'table.column' whenever multiple tables are involved.
- If memory rules suggest a proven approach, adapt those parameters.
- If memory rules warn about a failure pattern, actively avoid it.

You must respond with the `ParamPlan` schema: a short reasoning string plus a
`params` object matching FetchParams.
"""


def param_builder_node(state: AgentState) -> AgentState:
    schema = state.get("schema") or {}
    intent = state.get("intent") or {}
    dt = state.get("datetime_resolved") or {}
    summary = state.get("memory_summary") or ""
    memory_rules = state.get("memory_rules") or []
    validation_errors = state.get("validation_errors") or []

    schema_block = _schema_block(schema)
    memory_block = format_rules_for_prompt(memory_rules)

    retry_hint = ""
    if validation_errors:
        retry_hint = (
            "\n\nPREVIOUS ATTEMPT FAILED validation with these errors — "
            "FIX THEM:\n- " + "\n- ".join(validation_errors)
        )

    dt_block = ""
    if dt and dt.get("start"):
        dt_block = (
            f"\nRESOLVED DATE RANGE: start={dt.get('start')}, end={dt.get('end')}, "
            f"granularity={dt.get('granularity')}.\n"
            "Apply this as BETWEEN filter on the relevant date column and, if "
            "a time series is requested, set time_grouping accordingly."
        )

    llm = get_chat_model(temperature=0.0).with_structured_output(ParamPlan)

    messages = [
        SystemMessage(content=_SYS),
        SystemMessage(content=f"DATABASE SCHEMA:\n{schema_block}"),
        SystemMessage(content=f"MEMORY RULES:\n{memory_block}"),
    ]
    if summary:
        messages.append(SystemMessage(content=f"CONVERSATION SUMMARY:\n{summary}"))
    messages.append(
        HumanMessage(
            content=(
                f"USER QUERY: {state['user_query']}\n\n"
                f"EXTRACTED INTENT:\n{intent}"
                f"{dt_block}"
                f"{retry_hint}"
            )
        )
    )

    try:
        plan: ParamPlan = llm.invoke(messages)
    except Exception as exc:
        return {"error": f"Parameter building failed: {exc}"}

    params_dict = plan.params.model_dump(mode="json")
    _log.info("params built", extra={"parameters": params_dict})
    return {
        "parameters": params_dict,
        "param_reasoning": plan.reasoning,
        # Clear stale validation errors once a new plan is produced.
        "validation_errors": [],
    }


def _schema_block(schema: Dict[str, Any]) -> str:
    tables = schema.get("tables") or {}
    lines = []
    for tname, tdef in tables.items():
        cols = tdef.get("columns") or {}
        col_parts = []
        for cname, meta in cols.items():
            tag = meta.get("type", "other")
            if meta.get("primary_key"):
                tag += ",PK"
            if meta.get("foreign_key"):
                tag += f",FK→{meta['foreign_key']}"
            col_parts.append(f"{cname}[{tag}]")
        lines.append(f"- {tname}: " + ", ".join(col_parts))
    return "\n".join(lines) or "(no tables)"
