"""intent_node — extracts structured Intent from natural language."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from sql_agent.config import get_logger
from sql_agent.models import Intent
from sql_agent.models.graph_state import AgentState
from sql_agent.services.llm import get_chat_model


_log = get_logger("agent.intent")

_SYS = """You extract structured intent from a user's analytical question.
You DO NOT generate SQL. You only identify what the user wants:

- metrics: quantities to compute (e.g. revenue, count, average price)
- dimensions: grouping attributes (e.g. country, category, month)
- filters: simple {field, operator, value} triples (operator in eq|neq|gt|gte|lt|lte|in|like)
- time_range: a natural language time expression if present (e.g. "last 6 months")
- output_type: one of count | list_unique | table | time_series | aggregate
- visualize: true if the user wants a chart, else false
- notes: short hint for the downstream parameter-builder

Use ONLY the provided schema's tables/columns when hinting. Be concise.
"""


def intent_node(state: AgentState) -> AgentState:
    schema = state.get("schema") or {}
    summary = state.get("memory_summary") or ""

    schema_hint = _schema_hint(schema)

    llm = get_chat_model(temperature=0.0).with_structured_output(Intent)

    context = [
        SystemMessage(content=_SYS),
        SystemMessage(content=f"DATABASE SCHEMA:\n{schema_hint}"),
    ]
    if summary:
        context.append(
            SystemMessage(content=f"CONVERSATION SUMMARY:\n{summary}")
        )
    context.append(HumanMessage(content=state["user_query"]))

    try:
        intent = llm.invoke(context)
    except Exception as exc:
        return {"error": f"Intent extraction failed: {exc}"}

    intent_dict = intent.model_dump(mode="json")
    _log.info("intent extracted", extra={"intent": intent_dict})
    return {"intent": intent_dict}


def _schema_hint(schema: Dict[str, Any]) -> str:
    tables = schema.get("tables") or {}
    lines = []
    for tname, tdef in tables.items():
        cols = tdef.get("columns") or {}
        col_list = ", ".join(
            f"{c} ({meta.get('type')})" for c, meta in cols.items()
        )
        lines.append(f"- {tname}({col_list})")
    return "\n".join(lines) or "(no tables)"
