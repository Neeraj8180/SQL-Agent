"""Append-only JSONL writer for routing decisions.

Thread-safe. Creates parent directories on construction.

Phase 8.6: backed by ``RotatingJsonlWriter`` so this log no longer grows
unbounded. Configured via ``LOG_ROTATION_MAX_BYTES`` /
``LOG_ROTATION_BACKUP_COUNT`` (default: 50 MB × 5 backups).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sql_agent.config import get_logger, settings
from sql_agent.observability.rotating_jsonl import RotatingJsonlWriter

from .base import RoutingDecision


_log = get_logger("routing.decision_log")


class DecisionLogWriter:
    def __init__(self, path: Optional[Path] = None) -> None:
        if path is not None:
            resolved = Path(path)
        else:
            resolved = settings.resolved_path(settings.llm_routing_decision_log)
        self._writer = RotatingJsonlWriter(
            resolved,
            max_bytes=settings.log_rotation_max_bytes,
            backup_count=settings.log_rotation_backup_count,
        )

    @property
    def path(self) -> Path:
        return self._writer.path

    def append(self, decision: RoutingDecision) -> None:
        self._writer.append_line(decision.to_json_line())
