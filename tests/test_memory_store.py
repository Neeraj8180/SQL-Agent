"""Phase 9 tests — pluggable memory-store backends (FAISS, Qdrant).

Runs every test against BOTH backends where possible, so FAISS and
Qdrant are proven to be drop-in equivalents.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry as llm_registry
from sql_agent.services.memory_manager import reset_memory_manager
from sql_agent.services.memory_store import MemoryStore
from sql_agent.services.memory_store.faiss_backend import FaissBackend


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_faiss_backend_conforms_to_protocol(tmp_path):
    be = FaissBackend(dimension=32, root_dir=tmp_path)
    assert isinstance(be, MemoryStore)
    assert be.name == "faiss"
    assert be.dimension == 32


def _qdrant_available() -> bool:
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="qdrant-client not installed",
)


@requires_qdrant
def test_qdrant_backend_conforms_to_protocol(tmp_path):
    from sql_agent.services.memory_store.qdrant_backend import QdrantBackend

    be = QdrantBackend(
        dimension=32, url="", path=tmp_path / "q",
        collection_prefix="proto_test_",
    )
    assert isinstance(be, MemoryStore)
    assert be.name == "qdrant"
    assert be.dimension == 32


# ---------------------------------------------------------------------------
# Backend equivalence — same interface, same behavior
# ---------------------------------------------------------------------------


@pytest.fixture(params=["faiss", "qdrant"])
def backend(request, tmp_path):
    """Yield a fresh backend of each type per test."""
    if request.param == "faiss":
        yield FaissBackend(dimension=16, root_dir=tmp_path)
    else:
        if not _qdrant_available():
            pytest.skip("qdrant-client not installed")
        from sql_agent.services.memory_store.qdrant_backend import QdrantBackend

        # Each test gets its own collection prefix to avoid cross-test bleed
        # since embedded Qdrant persists per-path.
        yield QdrantBackend(
            dimension=16,
            url="",
            path=tmp_path / "qdrant",
            collection_prefix=f"t_{request.node.name}_",
        )


def test_backend_sizes_start_at_zero(backend):
    assert backend.reward_size == 0
    assert backend.penalty_size == 0


def test_backend_add_reward_increases_count(backend):
    v = [0.0] * 16
    v[0] = 1.0
    backend.add_reward(v, {"query": "q1"})
    assert backend.reward_size == 1


def test_backend_search_recovers_exact_match(backend):
    v = [0.1 * i for i in range(16)]
    backend.add_reward(v, {"query": "exact", "parameters": {"a": 1}})
    hits = backend.search_rewards(v, k=3, min_score=0.0)
    assert len(hits) >= 1
    # Payload survives round-trip.
    match = hits[0]
    assert match["query"] == "exact"
    assert match["parameters"] == {"a": 1}
    assert match["kind"] == "reward"
    assert "score" in match


def test_backend_reward_penalty_are_separate(backend):
    vr = [1.0, 0.0] + [0.0] * 14
    vp = [0.0, 1.0] + [0.0] * 14
    backend.add_reward(vr, {"query": "good"})
    backend.add_penalty(vp, {"query": "bad", "reason": "syntax"})

    assert backend.reward_size == 1
    assert backend.penalty_size == 1

    r_hits = backend.search_rewards(vr, k=3, min_score=0.0)
    p_hits = backend.search_penalties(vp, k=3, min_score=0.0)
    assert any(h["query"] == "good" for h in r_hits)
    assert any(h["query"] == "bad" for h in p_hits)
    # Searching rewards doesn't return penalties.
    assert all(h["kind"] == "reward" for h in r_hits)
    assert all(h["kind"] == "penalty" for h in p_hits)


def test_backend_min_score_filters_results(backend):
    target = [1.0] + [0.0] * 15
    unrelated = [0.0] * 15 + [1.0]
    backend.add_reward(target, {"query": "target"})
    # Searching with the unrelated vector and a high threshold returns nothing.
    hits = backend.search_rewards(unrelated, k=3, min_score=0.99)
    assert hits == []


def test_backend_rejects_wrong_dim(backend):
    with pytest.raises(ValueError):
        backend.add_reward([0.0] * 5, {"query": "bad-dim"})


# ---------------------------------------------------------------------------
# Factory picks the right backend from settings
# ---------------------------------------------------------------------------


def test_factory_returns_faiss_by_default(tmp_path):
    from sql_agent.services.memory_store import build_memory_store

    orig = settings.memory_store_backend
    try:
        settings.memory_store_backend = "faiss"
        store = build_memory_store(dimension=16, root_dir=tmp_path)
        assert store.name == "faiss"
    finally:
        settings.memory_store_backend = orig


@requires_qdrant
def test_factory_returns_qdrant_when_configured(tmp_path, monkeypatch):
    """Settings.memory_store_backend='qdrant' + embedded path builds Qdrant."""
    from sql_agent.services.memory_store import build_memory_store

    orig_backend = settings.memory_store_backend
    orig_faiss_dir = settings.faiss_index_dir
    try:
        settings.memory_store_backend = "qdrant"
        settings.faiss_index_dir = str(tmp_path)
        store = build_memory_store(dimension=16, root_dir=tmp_path)
        assert store.name == "qdrant"
    finally:
        settings.memory_store_backend = orig_backend
        settings.faiss_index_dir = orig_faiss_dir


def test_factory_auto_mode_picks_faiss_when_qdrant_url_empty(tmp_path):
    from sql_agent.services.memory_store import build_memory_store

    orig_backend = settings.memory_store_backend
    orig_url = settings.qdrant_url
    try:
        settings.memory_store_backend = "auto"
        settings.qdrant_url = ""
        store = build_memory_store(dimension=16, root_dir=tmp_path)
        assert store.name == "faiss"
    finally:
        settings.memory_store_backend = orig_backend
        settings.qdrant_url = orig_url


# ---------------------------------------------------------------------------
# MemoryManager integration (unified API delegates to the chosen backend)
# ---------------------------------------------------------------------------


@pytest.fixture
def faiss_memory_env(tmp_path):
    orig_faiss_dir = settings.faiss_index_dir
    orig_backend = settings.memory_store_backend
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.faiss_index_dir = str(tmp_path / "faiss")
        settings.memory_store_backend = "faiss"
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_registry.reset_caches()
        reset_memory_manager()
        yield
    finally:
        settings.faiss_index_dir = orig_faiss_dir
        settings.memory_store_backend = orig_backend
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_registry.reset_caches()
        reset_memory_manager()


@pytest.fixture
def qdrant_memory_env(tmp_path):
    if not _qdrant_available():
        pytest.skip("qdrant-client not installed")

    orig_faiss_dir = settings.faiss_index_dir
    orig_backend = settings.memory_store_backend
    orig_url = settings.qdrant_url
    orig_prefix = settings.qdrant_collection_prefix
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.faiss_index_dir = str(tmp_path / "faiss")
        settings.memory_store_backend = "qdrant"
        settings.qdrant_url = ""  # embedded
        settings.qdrant_collection_prefix = "test_integ_"
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_registry.reset_caches()
        reset_memory_manager()
        yield
    finally:
        settings.faiss_index_dir = orig_faiss_dir
        settings.memory_store_backend = orig_backend
        settings.qdrant_url = orig_url
        settings.qdrant_collection_prefix = orig_prefix
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_registry.reset_caches()
        reset_memory_manager()


def test_memory_manager_with_faiss_records_and_recalls(faiss_memory_env):
    from sql_agent.services.memory_manager import MemoryManager

    mm = MemoryManager(root_dir=Path(settings.faiss_dir))
    assert mm.backend == "faiss"
    assert mm.reward_size == 0

    q = "how many orders?"
    mm.record_reward(q, parameters={"t": "orders"}, tool_used="count")
    assert mm.reward_size == 1

    rules = mm.recall(q, k_reward=3, k_penalty=3)
    rewards = [r for r in rules if r.get("kind") == "reward"]
    assert len(rewards) == 1
    assert rewards[0]["query"] == q


@requires_qdrant
def test_memory_manager_with_qdrant_records_and_recalls(qdrant_memory_env):
    from sql_agent.services.memory_manager import MemoryManager

    mm = MemoryManager(root_dir=Path(settings.faiss_dir))
    assert mm.backend == "qdrant"

    q = "how many orders via qdrant?"
    mm.record_reward(q, parameters={"t": "orders"}, tool_used="count")
    rules = mm.recall(q, k_reward=3, k_penalty=3)
    rewards = [r for r in rules if r.get("kind") == "reward"]
    assert len(rewards) == 1
    assert rewards[0]["query"] == q
    assert rewards[0]["tool_used"] == "count"


@requires_qdrant
def test_qdrant_backend_persistence_survives_reinitialization(tmp_path):
    """Phase 9 property: writes to an embedded Qdrant path survive re-opening.

    This proves persistence (which matters for pod restarts on a
    StatefulSet). Cross-replica sharing via a running Qdrant SERVER is
    additionally validated manually via docker-compose — embedded Qdrant
    uses a file lock and only one client can hold it at a time, which is
    why production uses the server mode, not embedded, for HPA scale-out.
    """
    from sql_agent.services.memory_store.qdrant_backend import QdrantBackend

    db_path = tmp_path / "qdrant"
    prefix = "persistence_test_"

    # First session: write, then deterministically release the client.
    be1 = QdrantBackend(
        dimension=16, url="", path=db_path, collection_prefix=prefix
    )
    v = [1.0] + [0.0] * 15
    be1.add_reward(v, {"query": "survives-restart", "parameters": {"x": 1}})
    assert be1.reward_size == 1
    be1._client.close()  # release the file lock
    del be1

    # Second session on same path — data from first session is visible.
    be2 = QdrantBackend(
        dimension=16, url="", path=db_path, collection_prefix=prefix
    )
    try:
        assert be2.reward_size == 1
        hits = be2.search_rewards(v, k=3, min_score=0.0)
        assert any(h["query"] == "survives-restart" for h in hits)
    finally:
        be2._client.close()
