"""Zero-overhead tracker used when TRACKING_ENABLED=false (the default).

Returns a dummy handle and does nothing else. All three methods are cheap
enough to call unconditionally from the orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class _NoOpHandle:
    start_time_ns: int
    session_id: str


class NoOpTracker:
    name: str = "noop"

    def start(self, *, session_id: str, user_query: str) -> _NoOpHandle:
        return _NoOpHandle(start_time_ns=time.perf_counter_ns(), session_id=session_id)

    def finish(self, handle: _NoOpHandle, final_state: Dict[str, Any]) -> None:
        return None

    def finish_error(self, handle: _NoOpHandle, exc: BaseException) -> None:
        return None
