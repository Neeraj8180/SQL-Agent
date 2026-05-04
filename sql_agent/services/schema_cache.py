"""TTL cache for SchemaDiscoveryTool output.

Keyed by database URL so that switching DBs at runtime works cleanly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from sql_agent.config import settings
from sql_agent.models import SchemaInfo


@dataclass
class _Entry:
    value: SchemaInfo
    expires_at: float


class SchemaCache:
    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._ttl = ttl_seconds or settings.schema_cache_ttl_seconds
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[SchemaInfo]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.time():
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: SchemaInfo) -> None:
        with self._lock:
            self._store[key] = _Entry(
                value=value, expires_at=time.time() + self._ttl
            )

    def invalidate(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)


schema_cache = SchemaCache()
