"""Intent extraction schema — what the intent_agent LLM produces."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OutputType(str, Enum):
    COUNT = "count"
    LIST_UNIQUE = "list_unique"
    TABLE = "table"
    TIME_SERIES = "time_series"
    AGGREGATE = "aggregate"


class IntentTimeRange(BaseModel):
    """Raw natural language time expression. DateTimeHandlingTool resolves it."""

    expression: Optional[str] = Field(
        default=None,
        description="Free text like 'last 6 months', 'January 2024', etc.",
    )
    granularity: Optional[str] = Field(
        default=None,
        description="daily | weekly | monthly | yearly — if user implied bucketing.",
    )


class IntentFilter(BaseModel):
    field: str
    operator: str = "eq"
    value: str


class Intent(BaseModel):
    """LLM-extracted user intent."""

    metrics: List[str] = Field(
        default_factory=list,
        description="Quantities the user wants (e.g. 'revenue', 'order_count').",
    )
    dimensions: List[str] = Field(
        default_factory=list,
        description="Grouping attributes (e.g. 'country', 'category').",
    )
    filters: List[IntentFilter] = Field(default_factory=list)
    time_range: Optional[IntentTimeRange] = None
    output_type: OutputType = OutputType.TABLE
    visualize: bool = Field(
        default=True,
        description="Whether the user wants a chart.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-form hint for the parameter-builder agent.",
    )
