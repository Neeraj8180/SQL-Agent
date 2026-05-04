"""JSON-per-session chat persistence.

Files live in ``sqlite_db/chat_histories/<session_uuid>.json``. Each session
stores: id, title, created_at, updated_at, memory_summary, and a list of
messages (each with role, content, optional tool call artifacts).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_agent.config import get_logger, settings
from sql_agent.models.graph_state import ChatMessage


_log = get_logger("chat_history")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    memory_summary: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls, title: str = "New chat") -> "ChatSession":
        now = _now()
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now,
            memory_summary=None,
            messages=[],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            memory_summary=data.get("memory_summary"),
            messages=list(data.get("messages") or []),
        )


class ChatHistoryStore:
    """CRUD for per-session JSON files."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = (root or settings.chat_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.json"

    def new_session(self, title: str = "New chat") -> ChatSession:
        session = ChatSession.new(title=title)
        self.save(session)
        return session

    def list_sessions(self) -> List[ChatSession]:
        sessions: List[ChatSession] = []
        for p in self._root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append(ChatSession.from_dict(data))
            except Exception as exc:
                _log.warning("skipping corrupt history file %s: %s", p, exc)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def load(self, session_id: str) -> Optional[ChatSession]:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ChatSession.from_dict(data)
        except Exception as exc:
            _log.warning("could not load session %s: %s", session_id, exc)
            return None

    def save(self, session: ChatSession) -> None:
        session.updated_at = _now()
        p = self._path(session.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)

    def delete(self, session_id: str) -> bool:
        p = self._path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False

    # ------------------------------------------------------------------

    def append_turn(
        self,
        session: ChatSession,
        user_msg: ChatMessage,
        assistant_msg: ChatMessage,
        memory_summary: Optional[str] = None,
    ) -> ChatSession:
        session.messages.append(dict(user_msg))
        session.messages.append(dict(assistant_msg))
        if memory_summary is not None:
            session.memory_summary = memory_summary
        # Auto-title from first user message if still default.
        if session.title in ("New chat", "Untitled") and user_msg.get("content"):
            content = str(user_msg["content"]).strip().splitlines()[0]
            session.title = content[:60] + ("…" if len(content) > 60 else "")
        self.save(session)
        return session
