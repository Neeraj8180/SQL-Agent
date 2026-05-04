"""SQLAlchemy engine factory. Supports SQLite and PostgreSQL via DATABASE_URL."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Engine, create_engine

from sql_agent.config import settings


_engine: Optional[Engine] = None


def _resolve_url(url: str) -> str:
    """Resolve relative SQLite paths against the project root."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url.replace("sqlite:///", "", 1)
        path = settings.resolved_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = _resolve_url(settings.database_url)
        # Modest pool defaults; safe for both SQLite (single file) and Postgres.
        _engine = create_engine(url, future=True, pool_pre_ping=True)
    return _engine


def reset_engine() -> None:
    """Dispose cached engine (useful in tests / after config changes)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
