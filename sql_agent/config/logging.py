"""Logging configuration.

Phase 5 addition: when ``LOG_JSON=true``, the stream handler is swapped to
the ``JsonFormatter`` from ``sql_agent.observability.structured_logging``
and a ``ContextFilter`` is attached so every record picks up the current
``session_id`` / ``request_id`` from contextvars. The default remains the
human-readable format.
"""

from __future__ import annotations

import logging
import sys

from .settings import settings


_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_json:
        # Lazy import so pure-OpenAI deployments that don't enable JSON
        # logs never pay the cost of loading the observability package.
        from sql_agent.observability.structured_logging import (
            ContextFilter,
            JsonFormatter,
        )

        handler.setFormatter(JsonFormatter())
        handler.addFilter(ContextFilter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame noisy third parties.
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
