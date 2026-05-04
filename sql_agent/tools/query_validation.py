"""QueryValidationTool — guard-rails on FetchParams before execution."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from sql_agent.config import settings
from sql_agent.models import (
    AggFunc,
    ColumnType,
    FetchParams,
    FilterCondition,
    FilterOp,
    SchemaInfo,
)

from .base import BaseTool


class QueryValidationInput(BaseModel):
    params: FetchParams
    db_schema: SchemaInfo


class QueryValidationOutput(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


_NUMERIC = {ColumnType.INTEGER, ColumnType.FLOAT}
_STRINGY = {ColumnType.STRING}
_DATEY = {ColumnType.DATETIME}


class QueryValidationTool(BaseTool[QueryValidationInput, QueryValidationOutput]):
    name = "query_validation"
    description = "Check FetchParams against schema: tables, columns, types, limits."
    input_schema = QueryValidationInput
    output_schema = QueryValidationOutput

    def _execute(self, payload: QueryValidationInput) -> QueryValidationOutput:
        errors: List[str] = []
        warnings: List[str] = []
        p = payload.params
        s = payload.db_schema

        # Tables exist.
        for t in p.table_names:
            if not s.has_table(t):
                errors.append(f"Unknown table: '{t}'.")
        if errors:
            return QueryValidationOutput(is_valid=False, errors=errors)

        # Columns exist.
        for c in p.columns:
            self._check_column_ref(c, p.table_names, s, errors)

        # Aggregations.
        for agg in p.aggregations:
            if agg.column == "*":
                if agg.func != AggFunc.COUNT:
                    errors.append(f"'{agg.func}' cannot use '*'.")
                continue
            col_ref = f"{agg.table}.{agg.column}" if agg.table else agg.column
            ct = self._resolve_type(col_ref, p.table_names, s, errors)
            if ct is not None and agg.func in {AggFunc.SUM, AggFunc.AVG} and ct not in _NUMERIC:
                errors.append(
                    f"Cannot apply {agg.func.value.upper()} to non-numeric "
                    f"column '{col_ref}' (type={ct.value})."
                )

        # Group-by refs.
        for g in p.group_by:
            self._check_column_ref(g, p.table_names, s, errors)

        # Time grouping.
        if p.time_grouping:
            col_ref = (
                f"{p.time_grouping.table}.{p.time_grouping.column}"
                if p.time_grouping.table
                else p.time_grouping.column
            )
            ct = self._resolve_type(col_ref, p.table_names, s, errors)
            if ct is not None and ct not in _DATEY:
                errors.append(
                    f"time_grouping requires a datetime column; "
                    f"'{col_ref}' is {ct.value}."
                )

        # Filters.
        for f in p.filters:
            self._check_filter(f, p.table_names, s, errors)

        # Order by.
        for o in p.order_by:
            col_ref = f"{o.table}.{o.column}" if o.table else o.column
            # Allow ordering by an aggregation alias.
            if any(a.alias == o.column for a in p.aggregations):
                continue
            self._check_column_ref(col_ref, p.table_names, s, errors)

        # Limits.
        if p.limit <= 0:
            errors.append("limit must be > 0.")
        if p.limit > settings.data_fetch_max_limit:
            errors.append(
                f"limit {p.limit} exceeds max ({settings.data_fetch_max_limit})."
            )

        # Full scan warning: no filters, no aggregation, big limit.
        if not p.filters and not p.aggregations and p.limit >= 1000:
            warnings.append(
                "Potential full scan — add at least one filter or reduce limit."
            )

        return QueryValidationOutput(
            is_valid=not errors, errors=errors, warnings=warnings
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _check_column_ref(
        ref: str, tables: List[str], schema: SchemaInfo, errors: List[str]
    ) -> None:
        if "." in ref:
            t, c = ref.split(".", 1)
            if t not in tables:
                errors.append(f"Column '{ref}' references table not in query.")
                return
            if not schema.has_column(t, c):
                errors.append(f"Unknown column '{ref}'.")
                return
        else:
            present = [t for t in tables if schema.has_column(t, ref)]
            if not present:
                errors.append(f"Unknown column '{ref}'.")
            elif len(present) > 1:
                errors.append(
                    f"Ambiguous column '{ref}' across {present} — qualify as '<table>.{ref}'."
                )

    @staticmethod
    def _resolve_type(
        ref: str, tables: List[str], schema: SchemaInfo, errors: List[str]
    ):
        if "." in ref:
            t, c = ref.split(".", 1)
            if t in tables and schema.has_column(t, c):
                return schema.column_type(t, c)
            errors.append(f"Unknown column '{ref}'.")
            return None
        for t in tables:
            if schema.has_column(t, ref):
                return schema.column_type(t, ref)
        errors.append(f"Unknown column '{ref}'.")
        return None

    def _check_filter(
        self,
        f: FilterCondition,
        tables: List[str],
        schema: SchemaInfo,
        errors: List[str],
    ) -> None:
        col_ref = f"{f.table}.{f.column}" if f.table else f.column
        ct = self._resolve_type(col_ref, tables, schema, errors)
        if ct is None:
            return

        if f.op in (FilterOp.IS_NULL, FilterOp.IS_NOT_NULL):
            return

        if f.value is None:
            errors.append(f"Filter on '{col_ref}' with op '{f.op}' requires a value.")
            return

        if f.op == FilterOp.BETWEEN:
            if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                errors.append(f"BETWEEN on '{col_ref}' needs [low, high].")
                return
        if f.op in (FilterOp.IN, FilterOp.NOT_IN):
            if not isinstance(f.value, (list, tuple)) or len(f.value) == 0:
                errors.append(f"{f.op.value.upper()} on '{col_ref}' needs non-empty list.")
                return

        # Coarse type checks: numeric ops on non-numeric columns.
        if f.op in (FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE):
            if ct not in _NUMERIC and ct not in _DATEY:
                errors.append(
                    f"Cannot use '{f.op.value}' on non-numeric/non-date column "
                    f"'{col_ref}' (type={ct.value})."
                )
        if f.op == FilterOp.LIKE and ct not in _STRINGY:
            errors.append(
                f"LIKE requires a string column; '{col_ref}' is {ct.value}."
            )
