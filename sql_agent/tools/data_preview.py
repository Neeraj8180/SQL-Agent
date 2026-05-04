"""DataPreviewTool — small, safe preview before a full fetch."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Engine

from sql_agent.models import FetchParams, SchemaInfo
from sql_agent.services.db import get_engine

from .base import BaseTool
from .data_fetch import DataFetchInput, DataFetchOutput, DataFetchTool


class DataPreviewInput(BaseModel):
    params: FetchParams
    db_schema: SchemaInfo
    preview_rows: int = Field(default=5, ge=1, le=50)


class DataPreviewOutput(DataFetchOutput):
    pass


class DataPreviewTool(BaseTool[DataPreviewInput, DataPreviewOutput]):
    name = "data_preview"
    description = "Return a small preview of rows to validate a query quickly."
    input_schema = DataPreviewInput
    output_schema = DataPreviewOutput

    def __init__(self, engine: Optional[Engine] = None) -> None:
        super().__init__()
        self._fetch = DataFetchTool(engine=engine or get_engine())

    def _execute(self, payload: DataPreviewInput) -> DataPreviewOutput:
        out = self._fetch.run(
            DataFetchInput(
                params=payload.params,
                db_schema=payload.db_schema,
                row_cap=payload.preview_rows,
            )
        )
        return DataPreviewOutput(
            rows=out.rows, columns=out.columns, row_count=out.row_count
        )
