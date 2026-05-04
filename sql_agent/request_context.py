"""Request-scoped context, thread-safe and asyncio-safe via contextvars.

Two context variables live here:

    * ``session_id_var``  — the caller's chat session id. Spans multiple
      RPCs and multiple ``run_turn`` invocations within the same
      conversation. Used by the routing layer (phase 3) for sticky routing
      and by MLflow (phase 4) to group runs.

    * ``request_id_var``  — a fresh UUID per RPC. Used for tracing a
      single request through structured logs (phase 5) and for correlating
      Prometheus metrics with individual requests.

Both propagate automatically across threads because ``ContextVar`` is
thread-local AND task-local; the gRPC server's ``ThreadPoolExecutor`` sees
each value on the exact thread where it was set.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


session_id_var: ContextVar[Optional[str]] = ContextVar(
    "sql_agent_session_id", default=None
)

request_id_var: ContextVar[Optional[str]] = ContextVar(
    "sql_agent_request_id", default=None
)


# ---------------------------------------------------------------------------
# Phase 8.3: per-request LLM token accumulator.
#
# Each LLM call inside a run_turn (or an RPC) should push its usage into
# the list bound to this contextvar. Trackers read the accumulated totals
# at finish time and surface them as MLflow metrics + Prometheus counters.
#
# Shape of each entry:
#   {"provider": str, "model": str, "input_tokens": int, "output_tokens": int}
#
# Providers that don't know their usage (e.g. the Mock provider) simply
# don't append anything. This makes the feature zero-cost for deployments
# that don't care.
# ---------------------------------------------------------------------------

from typing import Any, Dict, List  # noqa: E402

token_usage_var: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "sql_agent_token_usage", default=None
)


@contextmanager
def token_usage_scope() -> Iterator[List[Dict[str, Any]]]:
    """Bind a fresh token-usage accumulator for the duration of the block."""
    accumulator: List[Dict[str, Any]] = []
    token = token_usage_var.set(accumulator)
    try:
        yield accumulator
    finally:
        token_usage_var.reset(token)


def record_token_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Append one LLM-call usage record to the active request's accumulator.

    No-op when called outside a ``token_usage_scope()`` — safe to call from
    anywhere without setup.
    """
    acc = token_usage_var.get()
    if acc is None:
        return
    acc.append(
        {
            "provider": provider,
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }
    )


@contextmanager
def session_scope(session_id: Optional[str]) -> Iterator[None]:
    """Bind ``session_id_var`` for the duration of the ``with`` block."""
    token = session_id_var.set(session_id)
    try:
        yield
    finally:
        session_id_var.reset(token)


@contextmanager
def request_scope(request_id: Optional[str] = None) -> Iterator[str]:
    """Bind ``request_id_var`` for the ``with`` block. Generates a UUID if
    ``request_id`` is not provided. Yields the resolved id so callers can
    log / propagate it."""
    rid = (request_id or str(uuid.uuid4()))
    token = request_id_var.set(rid)
    try:
        yield rid
    finally:
        request_id_var.reset(token)


__all__ = [
    "session_id_var",
    "session_scope",
    "request_id_var",
    "request_scope",
    "token_usage_var",
    "token_usage_scope",
    "record_token_usage",
]
