"""Structured JSON logging with contextvar-backed request/session ids.

Installation is opt-in — ``apply_json_logging()`` swaps the root logger's
formatter to the JSON one without altering the existing handler / level.
Tests can build the formatter directly without installing globally.

Design:
    * ``ContextFilter`` adds ``session_id`` and ``request_id`` attributes to
      every ``LogRecord``. Attaching a Filter to the root logger (as opposed
      to a LoggerAdapter) is the contextvars-friendly idiom: Filters run on
      every record regardless of which child logger created it.
    * ``JsonFormatter`` serializes to one JSON object per line. Extra
      ``record.*`` attributes whose names are not already part of the base
      record schema are included as top-level fields, so ``logger.info(
      "...", extra={"tool_used": "count"})`` works out-of-the-box.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sql_agent.request_context import request_id_var, session_id_var


# ---------------------------------------------------------------------------
# Filter: inject context vars into every record
# ---------------------------------------------------------------------------


class ContextFilter(logging.Filter):
    """Attaches ``session_id`` and ``request_id`` to every record.

    Safe to add multiple times (each call just re-reads the contextvars);
    idempotent by construction.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get() or ""
        record.request_id = request_id_var.get() or ""
        return True


# ---------------------------------------------------------------------------
# Formatter: one JSON object per line
# ---------------------------------------------------------------------------


_BASE_ATTRS = frozenset(
    # Attributes Python's logging module sets on every LogRecord. Anything
    # not in this set is considered a user-supplied extra and is included
    # in the JSON output.
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Renders a log record as a single-line JSON object.

    Reserved keys: ``timestamp``, ``level``, ``logger``, ``message``,
    ``session_id``, ``request_id``, ``exc_info`` (only if present). All
    other user-supplied ``extra`` fields are included as top-level keys.
    """

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        super().__init__()
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", "") or "",
            "request_id": getattr(record, "request_id", "") or "",
        }
        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)

        # Promote user-supplied extras.
        for k, v in record.__dict__.items():
            if k in _BASE_ATTRS or k in obj:
                continue
            try:
                json.dumps(v, default=str)
                obj[k] = v
            except Exception:
                obj[k] = str(v)

        return json.dumps(obj, ensure_ascii=self._ensure_ascii, default=str)


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def apply_json_logging(root: logging.Logger | None = None) -> None:
    """Swap the root logger's handlers to JSON output.

    Idempotent — calling this more than once does not stack formatters or
    filters. Existing log LEVEL / HANDLER is preserved; only the formatter
    is replaced, and a ContextFilter is attached to each handler.
    """
    root = root or logging.getLogger()
    formatter = JsonFormatter()
    ctx_filter = ContextFilter()
    for h in root.handlers:
        h.setFormatter(formatter)
        # Dedup filter by checking type.
        if not any(isinstance(f, ContextFilter) for f in h.filters):
            h.addFilter(ctx_filter)
