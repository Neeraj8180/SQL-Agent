"""StatisticalAnalysisTool — basic descriptives, correlation, grouped stats."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .base import BaseTool


class StatisticalAnalysisInput(BaseModel):
    rows: List[Dict[str, Any]]
    numeric_columns: List[str] = Field(default_factory=list)
    group_by: Optional[str] = None


class StatisticalAnalysisOutput(BaseModel):
    descriptive: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    correlation: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    grouped: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)


class StatisticalAnalysisTool(
    BaseTool[StatisticalAnalysisInput, StatisticalAnalysisOutput]
):
    name = "statistical_analysis"
    description = "Compute descriptive stats, correlations, and grouped aggs."
    input_schema = StatisticalAnalysisInput
    output_schema = StatisticalAnalysisOutput

    def _execute(
        self, payload: StatisticalAnalysisInput
    ) -> StatisticalAnalysisOutput:
        if not payload.rows:
            return StatisticalAnalysisOutput(insights=["No data to analyze."])

        df = pd.DataFrame(payload.rows)
        numeric = payload.numeric_columns or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric = [c for c in numeric if c in df.columns]
        for c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        descriptive: Dict[str, Dict[str, float]] = {}
        for col in numeric:
            s = df[col].dropna()
            if s.empty:
                continue
            descriptive[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "count": int(s.count()),
                "sum": float(s.sum()),
            }

        correlation: Dict[str, Dict[str, float]] = {}
        if len(numeric) >= 2:
            corr_df = df[numeric].corr(numeric_only=True)
            correlation = {
                col: {
                    other: (
                        float(corr_df.loc[col, other])
                        if not pd.isna(corr_df.loc[col, other])
                        else 0.0
                    )
                    for other in corr_df.columns
                }
                for col in corr_df.index
            }

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        if payload.group_by and payload.group_by in df.columns and numeric:
            g = df.groupby(payload.group_by)[numeric].agg(["mean", "sum", "count"])
            g.columns = ["__".join(c) for c in g.columns]
            grouped[payload.group_by] = g.reset_index().to_dict(orient="records")

        insights = self._build_insights(descriptive, correlation, df, payload)
        return StatisticalAnalysisOutput(
            descriptive=descriptive,
            correlation=correlation,
            grouped=grouped,
            insights=insights,
        )

    @staticmethod
    def _build_insights(
        descriptive: Dict[str, Dict[str, float]],
        correlation: Dict[str, Dict[str, float]],
        df: pd.DataFrame,
        payload: StatisticalAnalysisInput,
    ) -> List[str]:
        out: List[str] = [f"Analyzed {len(df)} row(s)."]
        for col, stats in descriptive.items():
            out.append(
                f"'{col}': mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
                f"std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]."
            )
        # Surface strongest correlation pair.
        best: Optional[tuple] = None
        for a, row in correlation.items():
            for b, v in row.items():
                if a >= b:
                    continue
                if best is None or abs(v) > abs(best[2]):
                    best = (a, b, v)
        if best and abs(best[2]) > 0.5:
            out.append(
                f"Strong correlation between '{best[0]}' and '{best[1]}': "
                f"r={best[2]:.2f}."
            )
        return out
