"""ML lifecycle tracking (phase 4).

Lazy public surface so importing this package does not force ``mlflow`` to
load for users who only want the file tracker (or no tracker).

Public names:
    * ``TurnTracker`` / ``TurnHandle`` — protocols
    * ``NoOpTracker``    — default; does nothing
    * ``FileTracker``    — JSONL-backed, zero external deps
    * ``MLflowTracker``  — MLflow-backed; requires ``requirements-tracking.txt``
    * ``get_tracker``    — process-wide singleton accessor
    * ``reset_tracker``  — test hook
    * ``summarize_state`` — helper that extracts numeric/categorical fields
                            from an ``AgentState`` dict; shared by all trackers.
"""

from __future__ import annotations

__all__ = [
    "TurnTracker",
    "TurnHandle",
    "NoOpTracker",
    "FileTracker",
    "MLflowTracker",
    "get_tracker",
    "reset_tracker",
    "summarize_state",
]


def __getattr__(name: str):
    if name in {"TurnTracker", "TurnHandle"}:
        from . import base

        return getattr(base, name)
    if name == "summarize_state":
        from . import base

        return base.summarize_state
    if name == "NoOpTracker":
        from .noop_tracker import NoOpTracker

        return NoOpTracker
    if name == "FileTracker":
        from .file_tracker import FileTracker

        return FileTracker
    if name == "MLflowTracker":
        from .mlflow_tracker import MLflowTracker

        return MLflowTracker
    if name in {"get_tracker", "reset_tracker"}:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
