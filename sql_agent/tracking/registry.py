"""Singleton tracker resolver.

Resolution rules:
    * ``settings.tracking_enabled == False``   -> NoOpTracker
    * ``settings.tracking_backend == "noop"``  -> NoOpTracker
    * ``settings.tracking_backend == "file"``  -> FileTracker
    * ``settings.tracking_backend == "mlflow"``-> MLflowTracker (raises if missing)
    * ``settings.tracking_backend == "auto"``  -> try MLflow; on ImportError
                                                 fall back to FileTracker;
                                                 on failure fall back to NoOp.

Call ``reset_tracker()`` after mutating settings in tests so the next
``get_tracker()`` re-resolves.
"""

from __future__ import annotations

import threading
from typing import Optional

from sql_agent.config import get_logger, settings


_log = get_logger("tracking.registry")


_instance = None  # type: Optional[object]
_lock = threading.Lock()


def _build() -> object:
    if not settings.tracking_enabled:
        from .noop_tracker import NoOpTracker

        return NoOpTracker()

    backend = (settings.tracking_backend or "auto").strip().lower()

    if backend == "noop":
        from .noop_tracker import NoOpTracker

        return NoOpTracker()
    if backend == "file":
        from .file_tracker import FileTracker

        return FileTracker()
    if backend == "mlflow":
        from .mlflow_tracker import MLflowTracker

        return MLflowTracker()
    if backend == "auto":
        try:
            from .mlflow_tracker import MLflowTracker

            return MLflowTracker()
        except Exception as exc:
            _log.warning(
                "auto backend: MLflow unavailable (%s); falling back to FileTracker.",
                exc,
            )
            try:
                from .file_tracker import FileTracker

                return FileTracker()
            except Exception as exc2:  # pragma: no cover — filesystem perms
                _log.warning(
                    "auto backend: FileTracker init failed (%s); using NoOpTracker.",
                    exc2,
                )
                from .noop_tracker import NoOpTracker

                return NoOpTracker()

    _log.warning(
        "Unknown TRACKING_BACKEND=%r; falling back to NoOpTracker.", backend
    )
    from .noop_tracker import NoOpTracker

    return NoOpTracker()


def get_tracker():
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = _build()
            _log.info("Tracker ready: %s", getattr(_instance, "name", type(_instance).__name__))
    return _instance


def reset_tracker() -> None:
    """Drop cached tracker so the next get_tracker() re-reads settings."""
    global _instance
    with _lock:
        _instance = None
