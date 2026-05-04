"""DataCleaningTool — type casting, null handling, deduplication via pandas."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .base import BaseTool, ToolExecutionError


class DataCleaningInput(BaseModel):
    rows: List[Dict[str, Any]]
    drop_duplicates: bool = True
    null_strategy: Literal["drop", "fill_zero", "fill_mean", "keep"] = "keep"
    numeric_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)


class DataCleaningOutput(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[str]
    dropped_rows: int = 0
    notes: List[str] = Field(default_factory=list)


class DataCleaningTool(BaseTool[DataCleaningInput, DataCleaningOutput]):
    name = "data_cleaning"
    description = "Normalize a result set before analysis (nulls/dups/types)."
    input_schema = DataCleaningInput
    output_schema = DataCleaningOutput

    def _execute(self, payload: DataCleaningInput) -> DataCleaningOutput:
        if not payload.rows:
            return DataCleaningOutput(rows=[], columns=[], notes=["empty input"])

        df = pd.DataFrame(payload.rows)
        initial = len(df)
        notes: List[str] = []

        for col in payload.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in payload.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if payload.null_strategy == "drop":
            df = df.dropna()
        elif payload.null_strategy == "fill_zero":
            df = df.fillna(0)
        elif payload.null_strategy == "fill_mean":
            for col in df.select_dtypes(include="number").columns:
                df[col] = df[col].fillna(df[col].mean())
            df = df.fillna(0)

        if payload.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            if len(df) < before:
                notes.append(f"dropped {before - len(df)} duplicate row(s)")

        dropped = initial - len(df)

        # Back to JSON-safe dicts.
        records: List[Dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            records.append(
                {
                    k: (
                        v.isoformat()
                        if hasattr(v, "isoformat")
                        else (None if pd.isna(v) else v)
                    )
                    for k, v in rec.items()
                }
            )

        return DataCleaningOutput(
            rows=records,
            columns=list(df.columns),
            dropped_rows=dropped,
            notes=notes,
        )
