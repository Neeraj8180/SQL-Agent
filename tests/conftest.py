"""Shared pytest fixtures.

Key responsibilities:
    1. Isolate test-time state (FAISS indices, SQLite DB) into a per-session
       tmp directory so tests NEVER touch the developer's real data.
    2. Install deterministic fake LLM + embedding seams so tests do NOT need
       an OPENAI_API_KEY and are reproducible.

These fixtures only set environment variables and monkey-patch the LLM
service module — they do not edit any sql_agent source file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when pytest is invoked from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# CRITICAL: eagerly import every module that does
# `from sql_agent.services.llm import get_chat_model` / `embed_text` at top
# level. If we don't do this *before* any fixture monkey-patches
# services.llm, the first import of an agent module inside a fixture will
# capture the already-patched (fake) function and — because monkeypatch's
# undo() restores the VALUE captured at setattr time — that poisoned
# binding survives teardown, leaking across tests.
#
# By pre-importing here, each agent module's `get_chat_model` is bound to
# the REAL function object; monkeypatch then captures the real one, patches
# to a fake for the test, and correctly restores the real one afterward.
# ---------------------------------------------------------------------------

import sql_agent.agents.intent_agent  # noqa: E402,F401
import sql_agent.agents.memory_agent  # noqa: E402,F401
import sql_agent.agents.param_builder_agent  # noqa: E402,F401
import sql_agent.services.memory_manager  # noqa: E402,F401
import sql_agent.services.llm  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Environment isolation (runs BEFORE any sql_agent module is imported).
# ---------------------------------------------------------------------------


def _isolate_environment(tmp_root: Path) -> None:
    """Point every stateful path at a throw-away tmp dir."""
    db_path = tmp_root / "test.db"
    faiss_dir = tmp_root / "faiss_index"
    chat_dir = tmp_root / "chat_histories"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    chat_dir.mkdir(parents=True, exist_ok=True)

    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["FAISS_INDEX_DIR"] = str(faiss_dir)
    os.environ["CHAT_HISTORY_DIR"] = str(chat_dir)
    # Avoid accidental real calls if a test forgets to mock the LLM.
    os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    # Insulate tests from any routing env vars the developer may have set
    # in the shell. Tests that want routing on should opt in explicitly
    # (via the routing_on fixture in tests/test_routing.py).
    os.environ["LLM_ROUTING_ENABLED"] = "false"
    os.environ["LLM_ROUTING_WEIGHTS"] = "openai:100"
    # Same for tracking — opt-in per test, not per shell environment.
    os.environ["TRACKING_ENABLED"] = "false"
    os.environ["TRACKING_BACKEND"] = "noop"
    # Phase 5: keep human-readable logs and metrics server off by default;
    # tests that need JSON logs or Prometheus opt in explicitly.
    os.environ["LOG_JSON"] = "false"
    os.environ["METRICS_ENABLED"] = "false"
    # Phase 7: keep memory writes enabled unless a test explicitly flips
    # READ_ONLY_MEMORY. Default is false; we set it explicitly here so no
    # shell-inherited value can silently break memory-write tests.
    os.environ["READ_ONLY_MEMORY"] = "false"


@pytest.fixture(scope="session", autouse=True)
def _isolated_state(tmp_path_factory):
    tmp_root = tmp_path_factory.mktemp("sql_agent_state")
    _isolate_environment(tmp_root)

    # Seed the temp SQLite DB once per session.
    from sql_agent.seed_demo import seed

    seed(force=True)

    yield tmp_root


# ---------------------------------------------------------------------------
# Deterministic fake LLM seams.
# ---------------------------------------------------------------------------


class _FakeStructuredLLM:
    """Mimics `ChatOpenAI.with_structured_output(Model).invoke(...)`."""

    def __init__(self, model_cls, factories):
        self._model_cls = model_cls
        self._factories = factories

    def invoke(self, _messages):
        factory = self._factories.get(self._model_cls.__name__)
        if factory is None:
            raise NotImplementedError(
                f"FakeChat: no factory registered for {self._model_cls.__name__}"
            )
        return factory(self._model_cls)


class _FakeResp:
    content = "(fake summary)"


class _FakeChatModel:
    temperature = 0.0

    def __init__(self, factories):
        self._factories = factories

    def with_structured_output(self, model_cls):
        return _FakeStructuredLLM(model_cls, self._factories)

    def invoke(self, _messages):
        return _FakeResp()


@pytest.fixture
def fake_llm(monkeypatch):
    """Return a callable that registers structured-output factories per test.

    Example:
        def test_count(fake_llm):
            fake_llm({
                "Intent":    lambda cls: cls(metrics=["count"], output_type="count"),
                "ParamPlan": lambda cls: cls(
                    reasoning="...",
                    params=FetchParams(
                        table_names=["orders"],
                        aggregations=[{"func": "count", "column": "*", "alias": "count"}],
                    ),
                ),
            })
    """
    factories: dict = {}

    def _register(new: dict) -> None:
        factories.update(new)

    import sql_agent.services.llm as llm_mod

    fake_chat = _FakeChatModel(factories)

    def _get_chat_model(temperature: float = 0.0):
        return fake_chat

    def _embed_text(_text: str):
        return [0.0] * 1536

    def _embed_texts(texts):
        return [[0.0] * 1536 for _ in texts]

    monkeypatch.setattr(llm_mod, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(llm_mod, "embed_text", _embed_text)
    monkeypatch.setattr(llm_mod, "embed_texts", _embed_texts)

    # The agent modules captured these symbols at import time via
    # `from sql_agent.services.llm import get_chat_model` — patch those too.
    import sql_agent.agents.intent_agent as intent_agent
    import sql_agent.agents.memory_agent as memory_agent
    import sql_agent.agents.param_builder_agent as param_builder_agent
    import sql_agent.services.memory_manager as memory_manager

    monkeypatch.setattr(intent_agent, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(param_builder_agent, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(memory_agent, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(memory_manager, "embed_text", _embed_text)

    return _register
