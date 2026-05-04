"""JSONL-backed tracker — no external deps.

Writes one JSON object per finished turn to the path configured by
``settings.tracking_file_log``. Thread-safe (per-writer lock around append).

Used as:
    * Default fallback when MLflow is not installed (via the "auto" backend).
    * Explicit backend when ``TRACKING_BACKEND=file``.
    * Extra side-channel alongside MLflow if a user wants a plain file too
      (not enabled by default; future phase if demand arises).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sql_agent.config import get_logger, settings
from sql_agent.observability.rotating_jsonl import RotatingJsonlWriter

from .base import summarize_state


_log = get_logger("tracking.file")


@dataclass
class _FileHandle:
    start_time_ns: int
    session_id: str
    user_query: str


class FileTracker:
    name: str = "file"

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is not None:
            resolved = Path(path)
        else:
            resolved = settings.resolved_path(settings.tracking_file_log)
        # Phase 8.6: size-rotated underlying writer.
        self._writer = RotatingJsonlWriter(
            resolved,
            max_bytes=settings.log_rotation_max_bytes,
            backup_count=settings.log_rotation_backup_count,
        )
        _log.info("FileTracker writing to %s", self._writer.path)

    @property
    def path(self) -> Path:
        return self._writer.path

    # ------------------------------------------------------------------

    def start(self, *, session_id: str, user_query: str) -> _FileHandle:
        return _FileHandle(
            start_time_ns=time.perf_counter_ns(),
            session_id=session_id,
            user_query=user_query,
        )

    def finish(self, handle: _FileHandle, final_state: Dict[str, Any]) -> None:
        self._write(handle, final_state=final_state, exc=None)

    def finish_error(self, handle: _FileHandle, exc: BaseException) -> None:
        self._write(handle, final_state={}, exc=exc)

    # ------------------------------------------------------------------

    def _write(
        self,
        handle: _FileHandle,
        *,
        final_state: Dict[str, Any],
        exc: Optional[BaseException],
    ) -> None:
        latency_ms = (time.perf_counter_ns() - handle.start_time_ns) / 1_000_000.0
        summary = summarize_state(
            final_state,
            user_query=handle.user_query,
            session_id=handle.session_id,
            query_max_chars=settings.tracking_query_max_chars,
        )
        summary["metrics"]["latency_ms"] = latency_ms

        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": handle.session_id,
            "params": summary["params"],
            "metrics": summary["metrics"],
            "tags": summary["tags"],
            "artifacts": summary["artifacts"],
        }
        if exc is not None:
            record["metrics"]["success"] = 0.0
            record["tags"]["error_type"] = record["tags"].get("error_type") or "exception"
            record["artifacts"]["error.txt"] = f"{type(exc).__name__}: {exc}"

        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception as e:  # pragma: no cover — defensive
            _log.warning("FileTracker JSON serialize failed: %s", e)
            return

        try:
            self._writer.append_line(line)
        except Exception as e:  # pragma: no cover
            _log.warning("FileTracker write failed: %s", e)
