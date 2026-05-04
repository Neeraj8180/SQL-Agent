"""ListingTool — SELECT DISTINCT for a column (filter-aware)."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, Table, select

from sql_agent.config import settings
from sql_agent.models import FilterCondition, SchemaInfo
from sql_agent.services.db import get_engine

from .base import BaseTool, ToolExecutionError
from .data_fetch import DataFetchTool


class ListingInput(BaseModel):
    table: str
    column: str
    db_schema: SchemaInfo
    filters: List[FilterCondition] = Field(default_factory=list)
    limit: int = Field(default=500, ge=1, le=10000)


class ListingOutput(BaseModel):
    values: List[Any]


class ListingTool(BaseTool[ListingInput, ListingOutput]):
    name = "listing"
    description = "Return distinct values of a column."
    input_schema = ListingInput
    output_schema = ListingOutput

    def __init__(self, engine: Optional[Engine] = None) -> None:
        super().__init__()
        self._engine = engine or get_engine()

    def _execute(self, payload: ListingInput) -> ListingOutput:
        if not payload.db_schema.has_column(payload.table, payload.column):
            raise ToolExecutionError(
                f"Column '{payload.table}.{payload.column}' not found."
            )
        md = MetaData()
        tbl = Table(payload.table, md, autoload_with=self._engine)
        tables = {payload.table: tbl}
        col = tbl.c[payload.column]

        stmt = select(col).distinct()
        where = DataFetchTool._build_filters(
            DataFetchTool(engine=self._engine),
            payload.filters,
            tables,
            payload.db_schema,
        )
        if where is not None:
            stmt = stmt.where(where)
        stmt = stmt.limit(min(payload.limit, settings.data_fetch_max_limit))

        with self._engine.connect() as conn:
            rows = [r[0] for r in conn.execute(stmt).fetchall()]
        return ListingOutput(values=rows)
