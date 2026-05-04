"""VisualizationTool — matplotlib charts returned as base64 PNG strings."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Literal, Optional

import matplotlib

matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .base import BaseTool, ToolExecutionError  # noqa: E402


ChartKind = Literal["line", "bar", "histogram", "scatter", "auto"]


class VisualizationInput(BaseModel):
    rows: List[Dict[str, Any]]
    chart_kind: ChartKind = "auto"
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = Field(
        default=None,
        description="Optional categorical column for multi-series line/bar.",
    )
    title: Optional[str] = None


class VisualizationOutput(BaseModel):
    image_base64: Optional[str] = None
    chart_kind: str = "none"
    note: Optional[str] = None


class VisualizationTool(BaseTool[VisualizationInput, VisualizationOutput]):
    name = "visualization"
    description = "Render a chart from tabular rows into a base64 PNG."
    input_schema = VisualizationInput
    output_schema = VisualizationOutput

    def _execute(self, payload: VisualizationInput) -> VisualizationOutput:
        if not payload.rows:
            return VisualizationOutput(note="No data to plot.")
        df = pd.DataFrame(payload.rows)
        kind, x, y = self._decide(payload, df)
        if kind is None or x is None or y is None:
            return VisualizationOutput(
                note="Could not auto-detect chart axes (needs at least one "
                "numeric column)."
            )

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
        try:
            self._render(ax, df, kind, x, y, payload.group)
        except Exception as exc:
            plt.close(fig)
            raise ToolExecutionError(f"Chart rendering failed: {exc}") from exc

        ax.set_title(payload.title or self._auto_title(kind, x, y, payload.group))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if kind in ("line", "bar"):
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha("right")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return VisualizationOutput(
            image_base64=base64.b64encode(buf.read()).decode("ascii"),
            chart_kind=kind,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _decide(payload: VisualizationInput, df: pd.DataFrame):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric = [c for c in df.columns if c not in numeric_cols]

        x = payload.x
        y = payload.y
        kind: Optional[ChartKind] = (
            None if payload.chart_kind == "auto" else payload.chart_kind
        )

        if kind == "histogram":
            y = y or (numeric_cols[0] if numeric_cols else None)
            x = x or y
            return kind, x, y

        if not numeric_cols:
            return None, None, None

        if not y:
            y = numeric_cols[-1]
        if not x:
            non_y = [c for c in df.columns if c != y]
            x = non_y[0] if non_y else y

        if kind is None:
            if x and ("date" in x.lower() or "time" in x.lower() or "month" in x.lower() or "year" in x.lower()):
                kind = "line"
            elif x in non_numeric:
                kind = "bar"
            elif x in numeric_cols and y in numeric_cols and x != y:
                kind = "scatter"
            else:
                kind = "line"
        return kind, x, y

    @staticmethod
    def _render(ax, df: pd.DataFrame, kind: str, x: str, y: str, group: Optional[str]):
        if kind == "histogram":
            ax.hist(pd.to_numeric(df[y], errors="coerce").dropna(), bins=20)
            ax.set_ylabel("frequency")
            return
        if kind == "scatter":
            ax.scatter(df[x], pd.to_numeric(df[y], errors="coerce"))
            return
        if group and group in df.columns:
            for name, sub in df.groupby(group):
                sub = sub.sort_values(x)
                if kind == "line":
                    ax.plot(sub[x], pd.to_numeric(sub[y], errors="coerce"), marker="o", label=str(name))
                else:
                    ax.bar(sub[x].astype(str) + f"/{name}", pd.to_numeric(sub[y], errors="coerce"), label=str(name))
            ax.legend()
            return
        df_sorted = df.sort_values(x) if x in df.columns else df
        if kind == "line":
            ax.plot(
                df_sorted[x].astype(str),
                pd.to_numeric(df_sorted[y], errors="coerce"),
                marker="o",
            )
        else:  # bar
            ax.bar(df_sorted[x].astype(str), pd.to_numeric(df_sorted[y], errors="coerce"))

    @staticmethod
    def _auto_title(kind: str, x: str, y: str, group: Optional[str]) -> str:
        base = f"{y} by {x}" if kind != "histogram" else f"Distribution of {y}"
        if group:
            base += f" ({group})"
        return base
