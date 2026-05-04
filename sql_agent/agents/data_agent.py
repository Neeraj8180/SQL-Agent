"""Data nodes: preview, fetch, clean."""

from __future__ import annotations

from typing import Any, Dict, List

from sql_agent.config import get_logger
from sql_agent.models import FetchParams, SchemaInfo
from sql_agent.models.graph_state import AgentState
from sql_agent.tools import (
    CountInput,
    CountTool,
    DataCleaningInput,
    DataCleaningTool,
    DataFetchInput,
    DataFetchTool,
    DataPreviewInput,
    DataPreviewTool,
    ListingInput,
    ListingTool,
    ToolExecutionError,
)


_log = get_logger("agent.data")

_preview = DataPreviewTool()
_fetch = DataFetchTool()
_count = CountTool()
_listing = ListingTool()
_clean = DataCleaningTool()


def preview_node(state: AgentState) -> AgentState:
    """Always run a tiny preview to catch obvious issues before the full fetch."""
    if state.get("tool_used") != "data_fetch":
        return {}
    try:
        params = FetchParams(**(state.get("parameters") or {}))
        schema = SchemaInfo(**(state.get("schema") or {}))
        out = _preview.run(
            DataPreviewInput(params=params, db_schema=schema, preview_rows=5)
        )
    except ToolExecutionError as exc:
        return {"error": f"Preview failed: {exc}"}
    except Exception as exc:
        return {"error": f"Preview failed unexpectedly: {exc}"}
    return {"data_preview": out.rows}


def fetch_node(state: AgentState) -> AgentState:
    tool = state.get("tool_used") or "data_fetch"
    try:
        params_dict = state.get("parameters") or {}
        schema = SchemaInfo(**(state.get("schema") or {}))

        if tool == "count":
            params = FetchParams(**params_dict)
            if not params.table_names:
                return {"error": "count requires a table."}
            # Reuse filter list from FetchParams (ignore other fields).
            out = _count.run(
                CountInput(
                    table=params.table_names[0],
                    db_schema=schema,
                    filters=params.filters,
                )
            )
            rows: List[Dict[str, Any]] = [{"count": out.count}]
            return {"data": rows}

        if tool == "listing":
            params = FetchParams(**params_dict)
            if not params.columns:
                return {"error": "listing requires a column."}
            col_ref = params.columns[0]
            if "." in col_ref:
                t, c = col_ref.split(".", 1)
            else:
                t = params.table_names[0] if params.table_names else ""
                c = col_ref
            out = _listing.run(
                ListingInput(
                    table=t,
                    column=c,
                    db_schema=schema,
                    filters=params.filters,
                    limit=params.limit,
                )
            )
            rows = [{c: v} for v in out.values]
            return {"data": rows}

        # Default: data_fetch
        params = FetchParams(**params_dict)
        out = _fetch.run(DataFetchInput(params=params, db_schema=schema))
        return {"data": out.rows}

    except ToolExecutionError as exc:
        return {"error": f"Data fetch failed: {exc}"}
    except Exception as exc:
        return {"error": f"Data fetch failed unexpectedly: {exc}"}


def clean_node(state: AgentState) -> AgentState:
    data = state.get("data") or []
    if not data:
        return {"data_cleaned": []}

    schema = state.get("schema") or {}
    params = state.get("parameters") or {}
    numeric_cols = _infer_numeric_columns(data, schema, params)
    datetime_cols = _infer_datetime_columns(data, schema, params)

    try:
        out = _clean.run(
            DataCleaningInput(
                rows=data,
                drop_duplicates=True,
                null_strategy="keep",
                numeric_columns=numeric_cols,
                datetime_columns=datetime_cols,
            )
        )
    except ToolExecutionError as exc:
        return {"error": f"Data cleaning failed: {exc}"}

    return {"data_cleaned": out.rows}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _infer_numeric_columns(
    rows: List[Dict[str, Any]], schema: Dict[str, Any], params: Dict[str, Any]
) -> List[str]:
    if not rows:
        return []
    sample = rows[0]
    numeric: List[str] = []
    # Aggregation aliases are numeric.
    for agg in params.get("aggregations") or []:
        alias = agg.get("alias")
        if alias and alias in sample:
            numeric.append(alias)
    # Schema-typed integer/float columns.
    tables = schema.get("tables") or {}
    for col_name in sample:
        for _, tdef in tables.items():
            meta = (tdef.get("columns") or {}).get(col_name)
            if meta and meta.get("type") in ("integer", "float"):
                numeric.append(col_name)
                break
    # Dedup preserving order.
    seen = set()
    out = []
    for c in numeric:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _infer_datetime_columns(
    rows: List[Dict[str, Any]], schema: Dict[str, Any], params: Dict[str, Any]
) -> List[str]:
    if not rows:
        return []
    sample = rows[0]
    tables = schema.get("tables") or {}
    out: List[str] = []
    for col_name in sample:
        for _, tdef in tables.items():
            meta = (tdef.get("columns") or {}).get(col_name)
            if meta and meta.get("type") == "datetime":
                out.append(col_name)
                break
    return out
