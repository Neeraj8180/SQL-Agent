"""CountTool — efficient COUNT(*) with optional filters, built on SQLAlchemy."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, Table, func, select

from sql_agent.models import FilterCondition, SchemaInfo
from sql_agent.services.db import get_engine

from .base import BaseTool, ToolExecutionError
from .data_fetch import DataFetchTool


class CountInput(BaseModel):
    table: str
    db_schema: SchemaInfo
    filters: List[FilterCondition] = Field(default_factory=list)


class CountOutput(BaseModel):
    count: int


class CountTool(BaseTool[CountInput, CountOutput]):
    name = "count"
    description = "Return COUNT(*) for a table with optional filters."
    input_schema = CountInput
    output_schema = CountOutput

    def __init__(self, engine: Optional[Engine] = None) -> None:
        super().__init__()
        self._engine = engine or get_engine()

    def _execute(self, payload: CountInput) -> CountOutput:
        if not payload.db_schema.has_table(payload.table):
            raise ToolExecutionError(f"Unknown table '{payload.table}'.")
        md = MetaData()
        tbl = Table(payload.table, md, autoload_with=self._engine)
        tables = {payload.table: tbl}

        stmt = select(func.count()).select_from(tbl)
        where = DataFetchTool._build_filters(
            DataFetchTool(engine=self._engine),
            payload.filters,
            tables,
            payload.db_schema,
        )
        if where is not None:
            stmt = stmt.where(where)

        with self._engine.connect() as conn:
            total = conn.execute(stmt).scalar_one()
        return CountOutput(count=int(total or 0))
