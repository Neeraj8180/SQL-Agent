"""Shared Pydantic schemas used by tools and agents.

These are the *only* structures the LLM is allowed to produce for DB
interaction — raw SQL is never exposed to the model.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


class ColumnType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    OTHER = "other"


class ColumnSchema(BaseModel):
    type: ColumnType
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None  # "table.column"


class TableSchema(BaseModel):
    columns: Dict[str, ColumnSchema]


class SchemaInfo(BaseModel):
    tables: Dict[str, TableSchema]

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return self.has_table(table) and column in self.tables[table].columns

    def column_type(self, table: str, column: str) -> Optional[ColumnType]:
        if not self.has_column(table, column):
            return None
        return self.tables[table].columns[column].type


# ---------------------------------------------------------------------------
# Query parameters (what the LLM emits via function calling)
# ---------------------------------------------------------------------------


class FilterOp(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class FilterCondition(BaseModel):
    """A single filter clause. Table is optional when unambiguous."""

    table: Optional[str] = Field(
        default=None,
        description="Table the column belongs to (required if multiple tables involved).",
    )
    column: str
    op: FilterOp
    value: Optional[Any] = None  # None for IS_NULL / IS_NOT_NULL

    @field_validator("column")
    @classmethod
    def _no_sql_chars(cls, v: str) -> str:
        if any(ch in v for ch in (";", "--", "/*", "*/", "`")):
            raise ValueError(f"Illegal characters in column name: {v!r}")
        return v


class JoinSpec(BaseModel):
    """One explicit join hop. Usually the system resolves joins automatically
    from the FK graph; this is an escape hatch."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    how: Literal["inner", "left", "right"] = "inner"


class AggFunc(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class AggregationSpec(BaseModel):
    """An aggregation column: `SUM(orders.revenue) AS total_revenue`."""

    func: AggFunc
    table: Optional[str] = None
    column: str  # "*" allowed only for COUNT
    alias: str


class OrderBySpec(BaseModel):
    table: Optional[str] = None
    column: str
    direction: Literal["asc", "desc"] = "asc"


class TimeBucket(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class TimeGrouping(BaseModel):
    """Optional time-bucketing instruction for a datetime column.

    The DataFetchTool converts this into a dialect-appropriate truncation
    (e.g. strftime for SQLite, date_trunc for Postgres)."""

    table: Optional[str] = None
    column: str
    bucket: TimeBucket
    alias: str = "time_bucket"


class FetchParams(BaseModel):
    """Structured input for DataFetchTool — the LLM produces this, never SQL."""

    table_names: List[str] = Field(..., min_length=1)
    columns: List[str] = Field(
        default_factory=list,
        description="Plain column selections in 'table.column' or 'column' form.",
    )
    filters: List[FilterCondition] = Field(default_factory=list)
    joins: List[JoinSpec] = Field(default_factory=list)
    aggregations: List[AggregationSpec] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    time_grouping: Optional[TimeGrouping] = None
    order_by: List[OrderBySpec] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=10000)

    @field_validator("table_names", "columns", "group_by")
    @classmethod
    def _no_sql_chars_list(cls, v: List[str]) -> List[str]:
        for item in v:
            if any(ch in item for ch in (";", "--", "/*", "*/", "`")):
                raise ValueError(f"Illegal characters in identifier: {item!r}")
        return v
