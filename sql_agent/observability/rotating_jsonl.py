"""Append-only JSONL writer with size-based rotation.

Used by:
    * ``routing.decision_log.DecisionLogWriter`` (phase 3)
    * ``tracking.file_tracker.FileTracker`` (phase 4)

Rotation policy:
    * When the current file's size ≥ ``max_bytes``, rotate:
        file.jsonl       -> file.jsonl.1
        file.jsonl.1     -> file.jsonl.2
        ...
        file.jsonl.(N-1) -> file.jsonl.N  (oldest dropped)
      where N = ``backup_count``.
    * ``max_bytes == 0`` disables rotation (unbounded growth).

Thread-safe: each writer holds a per-instance lock; size checks are done
under that same lock so concurrent writers never interleave partial lines.
"""

from __future__ import annotations

import threading
from pathlib import Path


class RotatingJsonlWriter:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 50 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max(0, int(max_bytes))
        self._backup_count = max(0, int(backup_count))
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append_line(self, line: str) -> None:
        """Append ``line`` (which must NOT end in newline) with rotation."""
        data = line + "\n"
        with self._lock:
            self._rotate_if_needed(len(data.encode("utf-8")))
            with self._path.open("a", encoding="utf-8") as f:
                f.write(data)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self._max_bytes <= 0 or self._backup_count <= 0:
            return
        try:
            current = self._path.stat().st_size if self._path.exists() else 0
        except OSError:
            return
        if current + incoming_bytes <= self._max_bytes:
            return
        self._rotate()

    def _rotate(self) -> None:
        # Shift: .N-1 -> .N, .N-2 -> .N-1, ..., .1 -> .2, current -> .1
        base = self._path
        # Drop the oldest.
        oldest = base.with_suffix(base.suffix + f".{self._backup_count}")
        try:
            if oldest.exists():
                oldest.unlink()
        except OSError:
            pass
        for i in range(self._backup_count - 1, 0, -1):
            src = base.with_suffix(base.suffix + f".{i}")
            dst = base.with_suffix(base.suffix + f".{i + 1}")
            if src.exists():
                try:
                    src.replace(dst)
                except OSError:
                    pass
        if base.exists():
            try:
                base.replace(base.with_suffix(base.suffix + ".1"))
            except OSError:
                pass
