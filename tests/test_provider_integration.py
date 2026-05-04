"""Phase 2.2 integration tests — run the ACTUAL pipeline through the registry.

Unlike ``test_grpc_smoke.py``, these tests do NOT use the ``fake_llm`` fixture
that monkey-patches module symbols. Instead they set ``LLM_PROVIDER=mock``
and let the real ``services.llm`` / registry path execute. This verifies
that in-production agent code actually routes through the new abstraction.
"""

from __future__ import annotations

import uuid

import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry


@pytest.fixture
def mock_provider_env():
    """Switch LLM_PROVIDER and EMBEDDING_PROVIDER to 'mock' for one test.

    Deliberately mutates the pydantic Settings instance (monkeypatch.setattr
    is flaky here), resets the provider cache, and resets the MemoryManager
    singleton so it re-probes the (now-mock) embedder's dim.
    """
    from sql_agent.services.memory_manager import reset_memory_manager

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        registry.reset_caches()
        reset_memory_manager()
        yield
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        registry.reset_caches()
        reset_memory_manager()


def test_services_llm_delegates_to_registry(mock_provider_env):
    """`get_chat_model` must return MockProvider's ChatModel now."""
    from sql_agent.llm_serving.mock_provider import _MockChatModel
    from sql_agent.services.llm import embed_text, embed_texts, get_chat_model

    chat = get_chat_model()
    assert isinstance(chat, _MockChatModel)

    # Embedders return deterministic vectors at the provider's declared dim
    # (MockEmbedder defaults to 64 in phase 2.4; FAISS dim-namespacing makes
    # this safe regardless of what OpenAI-backed indices produced previously).
    from sql_agent.llm_serving.mock_embedder import _DEFAULT_DIM as mock_dim

    v = embed_text("hello")
    assert len(v) == mock_dim
    v2 = embed_text("hello")
    assert v == v2

    batch = embed_texts(["a", "b"])
    assert len(batch) == 2
    assert all(len(x) == mock_dim for x in batch)


def test_run_turn_end_to_end_with_mock_provider(mock_provider_env):
    """The full orchestrator graph runs against the seeded DB with
    LLM_PROVIDER=mock and returns the correct count."""
    from sql_agent.agents.orchestrator import run_turn

    final = run_turn(
        "How many orders are there?",
        session_id=str(uuid.uuid4()),
        prior_messages=[],
    )

    assert not final.get("error"), f"unexpected error: {final.get('error')}"
    assert final.get("success") is True
    assert final.get("tool_used") == "count"
    assert final.get("data") == [{"count": 3387}]


def test_grpc_execute_sql_with_mock_provider(mock_provider_env):
    """ExecuteSQL over the servicer, driven by LLM_PROVIDER=mock, returns
    the count via the registry path."""
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server.server import SqlAgentServicer

    # Minimal grpc context stand-in.
    class _Ctx:
        def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

        def set_code(self, _): pass

        def set_details(self, _): pass

    servicer = SqlAgentServicer()
    resp = servicer.ExecuteSQL(
        pb2.ExecuteSQLRequest(query="How many orders are there?"),
        _Ctx(),
    )
    assert resp.success is True
    assert resp.tool_used == "count"
    assert resp.row_count == 1

    import json
    rows = json.loads(resp.rows_json)
    assert rows[0]["count"] == 3387


def test_openai_provider_path_unchanged(monkeypatch):
    """Default LLM_PROVIDER=openai still returns a real ChatOpenAI.

    This test does NOT make a network call — it merely constructs the chat
    model and asserts type + wiring, proving the openai code path isn't
    broken by phase 2.2's delegation.
    """
    registry.reset_caches()

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    orig_key = settings.openai_api_key
    orig_routing = settings.llm_routing_enabled
    try:
        settings.llm_provider = "openai"
        settings.embedding_provider = "auto"
        settings.openai_api_key = "test-dummy-key"  # doesn't trigger network
        # Ensure routing is off so this test is insulated from phase-3
        # settings (incl. any env leakage). This test is specifically about
        # the default registry path, not routing.
        settings.llm_routing_enabled = False

        from langchain_openai import ChatOpenAI

        from sql_agent.llm_serving.openai_provider import _ChatOpenAIProxy
        from sql_agent.services.llm import get_chat_model

        chat = get_chat_model(temperature=0.3)
        # Phase 8.3: we now return a token-counting proxy that wraps the
        # real ChatOpenAI. External contract (with_structured_output, invoke,
        # .temperature, .model_name via __getattr__) remains the same.
        assert isinstance(chat, _ChatOpenAIProxy)
        assert isinstance(chat._inner, ChatOpenAI)
        assert chat.temperature == 0.3
        assert chat.model_name == settings.openai_chat_model
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        settings.openai_api_key = orig_key
        settings.llm_routing_enabled = orig_routing
        registry.reset_caches()
