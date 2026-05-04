"""DataFetchTool — executes FetchParams via SQLAlchemy (NO raw SQL)."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Engine,
    MetaData,
    Table,
    and_,
    asc,
    desc,
    func,
    select,
)
from sqlalchemy.sql import ColumnElement

from sql_agent.config import settings
from sql_agent.models import (
    AggFunc,
    AggregationSpec,
    FetchParams,
    FilterCondition,
    FilterOp,
    JoinSpec,
    SchemaInfo,
    TimeBucket,
    TimeGrouping,
)
from sql_agent.services.db import get_engine

from .base import BaseTool, ToolExecutionError
from .table_relationship import TableRelationshipTool


class DataFetchInput(BaseModel):
    params: FetchParams
    db_schema: SchemaInfo
    # For CountTool / ListingTool etc. to enforce different defaults.
    row_cap: Optional[int] = None


class DataFetchOutput(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0


class DataFetchTool(BaseTool[DataFetchInput, DataFetchOutput]):
    """Parameter-driven, dialect-agnostic data fetcher.

    The LLM never constructs SQL — it produces a ``FetchParams`` and this tool
    lowers it into a SQLAlchemy ``Select`` that binds all values safely.
    """

    name = "data_fetch"
    description = "Fetch rows from one or more tables given FetchParams."
    input_schema = DataFetchInput
    output_schema = DataFetchOutput

    def __init__(self, engine: Optional[Engine] = None) -> None:
        super().__init__()
        self._engine = engine or get_engine()

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    def _execute(self, payload: DataFetchInput) -> DataFetchOutput:
        params = payload.params
        schema = payload.db_schema

        self._assert_tables(params, schema)
        tables = self._reflect_tables(params.table_names)
        joined = self._build_from(tables, params, schema)

        select_cols = self._build_select_columns(params, tables)
        stmt = select(*select_cols).select_from(joined)

        where = self._build_filters(params.filters, tables, schema)
        if where is not None:
            stmt = stmt.where(where)

        stmt = self._apply_group_by(stmt, params, tables)
        stmt = self._apply_order_by(stmt, params, tables, select_cols)

        limit = payload.row_cap or params.limit or settings.data_fetch_default_limit
        limit = min(limit, settings.data_fetch_max_limit)
        stmt = stmt.limit(limit)

        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            column_names = list(result.keys())
            rows = [dict(zip(column_names, r)) for r in result.fetchall()]

        rows = [self._serialize_row(r) for r in rows]
        return DataFetchOutput(
            rows=rows, columns=column_names, row_count=len(rows)
        )

    # ------------------------------------------------------------------
    # Validation + reflection
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_tables(params: FetchParams, schema: SchemaInfo) -> None:
        for t in params.table_names:
            if not schema.has_table(t):
                raise ToolExecutionError(f"Unknown table: '{t}'.")

    def _reflect_tables(self, table_names: List[str]) -> Dict[str, Table]:
        md = MetaData()
        out: Dict[str, Table] = {}
        for name in table_names:
            out[name] = Table(name, md, autoload_with=self._engine)
        return out

    # ------------------------------------------------------------------
    # Column resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_column(
        ref: str,
        tables: Dict[str, Table],
        *,
        default_table: Optional[str] = None,
    ) -> Column:
        if "." in ref:
            t, c = ref.split(".", 1)
            if t not in tables:
                raise ToolExecutionError(f"Unknown table in '{ref}'.")
            if c not in tables[t].c:
                raise ToolExecutionError(f"Unknown column '{ref}'.")
            return tables[t].c[c]

        # Bare name: search across tables, require uniqueness.
        if default_table and default_table in tables and ref in tables[default_table].c:
            return tables[default_table].c[ref]
        matches = [t for t in tables.values() if ref in t.c]
        if len(matches) == 0:
            raise ToolExecutionError(f"Unknown column '{ref}'.")
        if len(matches) > 1:
            raise ToolExecutionError(
                f"Ambiguous column '{ref}' — qualify as '<table>.{ref}'."
            )
        return matches[0].c[ref]

    # ------------------------------------------------------------------
    # FROM / JOIN construction
    # ------------------------------------------------------------------

    def _build_from(
        self,
        tables: Dict[str, Table],
        params: FetchParams,
        schema: SchemaInfo,
    ):
        if len(tables) == 1:
            return next(iter(tables.values()))

        # Prefer explicit joins when provided; else auto-resolve from FKs.
        if params.joins:
            from_expr = tables[params.table_names[0]]
            joined_names = {params.table_names[0]}
            # Use each explicit join in declared order.
            for j in params.joins:
                from_expr = self._apply_explicit_join(
                    from_expr, j, tables, joined_names
                )
            # If any table still unjoined, try to auto-resolve.
            remaining = [t for t in tables if t not in joined_names]
            if remaining:
                from_expr = self._apply_auto_joins(
                    from_expr, tables, joined_names, remaining, schema
                )
            return from_expr

        from_expr = tables[params.table_names[0]]
        joined_names = {params.table_names[0]}
        remaining = params.table_names[1:]
        return self._apply_auto_joins(
            from_expr, tables, joined_names, remaining, schema
        )

    @staticmethod
    def _apply_explicit_join(
        from_expr,
        j: JoinSpec,
        tables: Dict[str, Table],
        joined_names: set,
    ):
        if j.left_table not in tables or j.right_table not in tables:
            raise ToolExecutionError(
                f"Join references unknown table: {j.left_table} / {j.right_table}"
            )
        left = tables[j.left_table]
        right = tables[j.right_table]
        if j.left_column not in left.c or j.right_column not in right.c:
            raise ToolExecutionError(
                "Join references unknown column: "
                f"{j.left_table}.{j.left_column} ↔ {j.right_table}.{j.right_column}"
            )
        on_clause = left.c[j.left_column] == right.c[j.right_column]
        # Join the side that's not yet in the FROM.
        if j.left_table in joined_names and j.right_table not in joined_names:
            target = right
            joined_names.add(j.right_table)
        elif j.right_table in joined_names and j.left_table not in joined_names:
            target = left
            joined_names.add(j.left_table)
        else:
            # Both already joined → skip (idempotent). Neither joined yet:
            # join right onto left.
            if j.left_table not in joined_names:
                from_expr = from_expr.join(left, on_clause)
                joined_names.add(j.left_table)
            if j.right_table not in joined_names:
                from_expr = from_expr.join(right, on_clause)
                joined_names.add(j.right_table)
            return from_expr

        is_outer = j.how in ("left", "right")
        return from_expr.join(target, on_clause, isouter=is_outer)

    @staticmethod
    def _apply_auto_joins(
        from_expr,
        tables: Dict[str, Table],
        joined_names: set,
        remaining: List[str],
        schema: SchemaInfo,
    ):
        rel_tool = TableRelationshipTool()
        while remaining:
            target = remaining.pop(0)
            if target in joined_names:
                continue
            edges_out = rel_tool.run(
                {
                    "schema": schema,
                    "tables": list(joined_names) + [target],
                }
            ).edges
            # Find an edge connecting something in joined_names to target.
            connecting = next(
                (
                    e
                    for e in edges_out
                    if (
                        (e.left_table in joined_names and e.right_table == target)
                        or (e.right_table in joined_names and e.left_table == target)
                    )
                ),
                None,
            )
            if connecting is None:
                raise ToolExecutionError(
                    f"No FK edge to auto-join '{target}' onto {sorted(joined_names)}."
                )
            left = tables[connecting.left_table]
            right = tables[connecting.right_table]
            on_clause = (
                left.c[connecting.left_column]
                == right.c[connecting.right_column]
            )
            if connecting.left_table in joined_names:
                from_expr = from_expr.join(tables[connecting.right_table], on_clause)
            else:
                from_expr = from_expr.join(tables[connecting.left_table], on_clause)
            joined_names.update({connecting.left_table, connecting.right_table})
        return from_expr

    # ------------------------------------------------------------------
    # SELECT clause (columns, aggregations, time bucket)
    # ------------------------------------------------------------------

    def _build_select_columns(
        self, params: FetchParams, tables: Dict[str, Table]
    ) -> List[ColumnElement]:
        cols: List[ColumnElement] = []

        if params.time_grouping:
            cols.append(self._time_bucket_expr(params.time_grouping, tables))

        for c in params.columns:
            cols.append(self._resolve_column(c, tables))

        for agg in params.aggregations:
            cols.append(self._agg_expr(agg, tables))

        if not cols:
            # Default: all columns from the first table.
            first = next(iter(tables.values()))
            cols = list(first.c)
        return cols

    def _agg_expr(
        self, agg: AggregationSpec, tables: Dict[str, Table]
    ) -> ColumnElement:
        if agg.column == "*" and agg.func == AggFunc.COUNT:
            expr = func.count()
        else:
            col_ref = f"{agg.table}.{agg.column}" if agg.table else agg.column
            col = self._resolve_column(col_ref, tables)
            fn = {
                AggFunc.COUNT: func.count,
                AggFunc.SUM: func.sum,
                AggFunc.AVG: func.avg,
                AggFunc.MIN: func.min,
                AggFunc.MAX: func.max,
            }[agg.func]
            expr = fn(col)
        return expr.label(agg.alias)

    def _time_bucket_expr(
        self, tg: TimeGrouping, tables: Dict[str, Table]
    ) -> ColumnElement:
        col_ref = f"{tg.table}.{tg.column}" if tg.table else tg.column
        col = self._resolve_column(col_ref, tables)
        dialect = self._engine.dialect.name
        return self._bucket_for_dialect(col, tg.bucket, dialect).label(tg.alias)

    @staticmethod
    def _bucket_for_dialect(
        col: Column, bucket: TimeBucket, dialect: str
    ) -> ColumnElement:
        if dialect == "sqlite":
            fmt = {
                TimeBucket.DAILY: "%Y-%m-%d",
                TimeBucket.WEEKLY: "%Y-%W",
                TimeBucket.MONTHLY: "%Y-%m",
                TimeBucket.YEARLY: "%Y",
            }[bucket]
            return func.strftime(fmt, col)
        # Postgres / generic ANSI.
        trunc_unit = {
            TimeBucket.DAILY: "day",
            TimeBucket.WEEKLY: "week",
            TimeBucket.MONTHLY: "month",
            TimeBucket.YEARLY: "year",
        }[bucket]
        return func.date_trunc(trunc_unit, col)

    # ------------------------------------------------------------------
    # WHERE
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        filters: List[FilterCondition],
        tables: Dict[str, Table],
        schema: SchemaInfo,
    ) -> Optional[ColumnElement]:
        if not filters:
            return None
        clauses: List[ColumnElement] = []
        for f in filters:
            col_ref = f"{f.table}.{f.column}" if f.table else f.column
            col = self._resolve_column(col_ref, tables)
            clauses.append(self._filter_expr(col, f))
        return and_(*clauses)

    @staticmethod
    def _filter_expr(col: Column, f: FilterCondition) -> ColumnElement:
        v = f.value
        op = f.op
        if op == FilterOp.EQ:
            return col == v
        if op == FilterOp.NEQ:
            return col != v
        if op == FilterOp.GT:
            return col > v
        if op == FilterOp.GTE:
            return col >= v
        if op == FilterOp.LT:
            return col < v
        if op == FilterOp.LTE:
            return col <= v
        if op == FilterOp.IN:
            if not isinstance(v, (list, tuple)) or not v:
                raise ToolExecutionError(
                    f"IN filter on '{col.name}' requires a non-empty list."
                )
            return col.in_(list(v))
        if op == FilterOp.NOT_IN:
            if not isinstance(v, (list, tuple)) or not v:
                raise ToolExecutionError(
                    f"NOT_IN filter on '{col.name}' requires a non-empty list."
                )
            return col.notin_(list(v))
        if op == FilterOp.LIKE:
            return col.like(str(v))
        if op == FilterOp.BETWEEN:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ToolExecutionError(
                    f"BETWEEN filter on '{col.name}' requires [low, high]."
                )
            return col.between(v[0], v[1])
        if op == FilterOp.IS_NULL:
            return col.is_(None)
        if op == FilterOp.IS_NOT_NULL:
            return col.isnot(None)
        raise ToolExecutionError(f"Unsupported filter op: {op}")

    # ------------------------------------------------------------------
    # GROUP BY / ORDER BY
    # ------------------------------------------------------------------

    def _apply_group_by(self, stmt, params: FetchParams, tables):
        group_cols: List[ColumnElement] = []
        if params.time_grouping:
            tg = params.time_grouping
            group_cols.append(
                self._time_bucket_expr(tg, tables)
            )
        for g in params.group_by:
            group_cols.append(self._resolve_column(g, tables))
        if group_cols:
            stmt = stmt.group_by(*group_cols)
        return stmt

    def _apply_order_by(self, stmt, params: FetchParams, tables, select_cols):
        if not params.order_by:
            return stmt
        exprs = []
        labeled_cols = {getattr(c, "name", None): c for c in select_cols}
        for o in params.order_by:
            # Allow ordering by a select-list alias (e.g., aggregation alias).
            if o.column in labeled_cols and labeled_cols[o.column] is not None:
                expr = labeled_cols[o.column]
            else:
                col_ref = f"{o.table}.{o.column}" if o.table else o.column
                expr = self._resolve_column(col_ref, tables)
            exprs.append(desc(expr) if o.direction == "desc" else asc(expr))
        return stmt.order_by(*exprs)

    # ------------------------------------------------------------------
    # Result serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (_dt.date, _dt.datetime)):
                out[k] = v.isoformat()
            elif isinstance(v, _dt.time):
                out[k] = v.isoformat()
            else:
                out[k] = v
        return out
