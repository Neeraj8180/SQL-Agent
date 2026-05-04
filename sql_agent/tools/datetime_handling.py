"""DateTimeHandlingTool — deterministic relative/absolute date resolution.

Keeps the LLM *out* of date arithmetic. Input is a natural-language expression
(from the intent), output is a concrete ``{start, end, granularity}`` struct
suitable for building filters + time grouping.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Optional

from dateutil import parser as _date_parser
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

from .base import BaseTool, ToolExecutionError


class DateTimeInput(BaseModel):
    expression: Optional[str] = Field(
        default=None,
        description="e.g. 'last 7 days', 'last 6 months', '2024-Q1', 'January 2024'.",
    )
    granularity: Optional[str] = Field(
        default=None,
        description="Optional user-provided bucket: daily | weekly | monthly | yearly.",
    )
    # Pin "now" for deterministic tests.
    now: Optional[datetime] = None


class DateTimeOutput(BaseModel):
    start: Optional[str] = None  # ISO date
    end: Optional[str] = None  # ISO date (inclusive)
    granularity: Optional[str] = None  # daily | weekly | monthly | yearly
    human_description: str = ""


_RELATIVE_RE = re.compile(
    r"""
    ^\s*
    (last|past|previous)\s+
    (\d+)\s+
    (day|days|week|weeks|month|months|year|years|quarter|quarters)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SIMPLE_LAST_RE = re.compile(
    r"^\s*(last|past|previous)\s+(day|week|month|year|quarter)\s*$",
    re.IGNORECASE,
)


class DateTimeHandlingTool(BaseTool[DateTimeInput, DateTimeOutput]):
    name = "datetime_handling"
    description = "Resolve NL time expressions into ISO [start, end] ranges."
    input_schema = DateTimeInput
    output_schema = DateTimeOutput

    def _execute(self, payload: DateTimeInput) -> DateTimeOutput:
        if not payload.expression:
            return DateTimeOutput(granularity=self._normalize_granularity(payload.granularity))

        expr = payload.expression.strip().lower()
        now = (payload.now or datetime.utcnow()).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today = now.date()

        # Pattern 1: "last N <unit>"
        m = _RELATIVE_RE.match(expr)
        if m:
            n = int(m.group(2))
            unit = m.group(3).rstrip("s")
            return self._resolve_relative(n, unit, today, payload.granularity)

        # Pattern 2: "last <unit>"
        m = _SIMPLE_LAST_RE.match(expr)
        if m:
            unit = m.group(2)
            return self._resolve_relative(1, unit, today, payload.granularity)

        # Pattern 3: "today" / "yesterday"
        if expr in ("today",):
            return self._range(today, today, "daily", "today")
        if expr in ("yesterday",):
            d = today - timedelta(days=1)
            return self._range(d, d, "daily", "yesterday")

        # Pattern 4: "this month" / "this year" / "this week"
        if expr in ("this month", "current month"):
            start = today.replace(day=1)
            end = today
            return self._range(start, end, "daily", "this month")
        if expr in ("this year", "current year"):
            start = today.replace(month=1, day=1)
            end = today
            return self._range(start, end, "monthly", "this year")
        if expr in ("this week", "current week"):
            start = today - timedelta(days=today.weekday())
            end = today
            return self._range(start, end, "daily", "this week")

        # Pattern 5: ISO or free-form absolute — defer to dateutil.
        try:
            parsed = _date_parser.parse(payload.expression, fuzzy=True)
            d = parsed.date()
            return self._range(d, d, "daily", payload.expression)
        except (ValueError, OverflowError) as exc:
            raise ToolExecutionError(
                f"Could not parse time expression: '{payload.expression}'. "
                "Try 'last 6 months', 'yesterday', or an ISO date."
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_relative(
        self,
        n: int,
        unit: str,
        today: date,
        user_granularity: Optional[str],
    ) -> DateTimeOutput:
        end = today
        if unit == "day":
            start = today - timedelta(days=n - 1)
            granularity = "daily"
        elif unit == "week":
            start = today - timedelta(weeks=n) + timedelta(days=1)
            granularity = "daily"
        elif unit == "month":
            start = (today.replace(day=1) - relativedelta(months=n - 1))
            granularity = "monthly"
        elif unit == "quarter":
            start = (today.replace(day=1) - relativedelta(months=3 * n - 1))
            granularity = "monthly"
        elif unit == "year":
            start = today.replace(month=1, day=1) - relativedelta(years=n - 1)
            granularity = "yearly"
        else:
            raise ToolExecutionError(f"Unknown relative unit: {unit}")

        if user_granularity:
            granularity = self._normalize_granularity(user_granularity) or granularity

        return self._range(start, end, granularity, f"last {n} {unit}s".strip())

    @staticmethod
    def _normalize_granularity(g: Optional[str]) -> Optional[str]:
        if not g:
            return None
        g = g.strip().lower()
        mapping = {
            "day": "daily",
            "days": "daily",
            "daily": "daily",
            "week": "weekly",
            "weekly": "weekly",
            "month": "monthly",
            "monthly": "monthly",
            "year": "yearly",
            "yearly": "yearly",
            "annual": "yearly",
            "annually": "yearly",
        }
        return mapping.get(g)

    @staticmethod
    def _range(
        start: date, end: date, granularity: str, human: str
    ) -> DateTimeOutput:
        return DateTimeOutput(
            start=start.isoformat(),
            end=end.isoformat(),
            granularity=granularity,
            human_description=human,
        )
