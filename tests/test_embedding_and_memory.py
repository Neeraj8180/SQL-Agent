"""Phase 2.4 tests — embedding abstraction + dim-namespaced FAISS.

Two clusters:
    1. Unit tests for HuggingFaceEmbedder (torch-gated — skipped if missing).
    2. MemoryManager tests for path layout, auto-migration, dim-switching,
       and a final LLM_PROVIDER=hf + EMBEDDING_PROVIDER=hf end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry
from sql_agent.llm_serving.mock_embedder import MockEmbedder
# Phase 9: _migrate_legacy_indices moved into the FAISS backend module.
from sql_agent.services.memory_store.faiss_backend import _migrate_legacy_indices
from sql_agent.services.memory_manager import (
    MemoryManager,
    _LEGACY_DIM,
    reset_memory_manager,
)


# ---------------------------------------------------------------------------
# HuggingFaceEmbedder (torch-gated)
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


requires_local_stack = pytest.mark.skipif(
    not _torch_available(),
    reason="sentence-transformers / torch not installed; "
    "pip install -r requirements-llm-local.txt",
)


@pytest.fixture(scope="module")
def _hf_embedder():
    if not _torch_available():
        pytest.skip("local-LLM stack not installed")
    from sql_agent.llm_serving.hf_embedder import HuggingFaceEmbedder

    return HuggingFaceEmbedder(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


@requires_local_stack
@pytest.mark.timeout(600)
def test_hf_embedder_produces_384_dim_vectors(_hf_embedder):
    emb = _hf_embedder
    assert emb.name == "hf"
    assert emb.device == "cpu"
    assert emb.dimension == 384

    v = emb.embed("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)


@requires_local_stack
def test_hf_embedder_batch_matches_single(_hf_embedder):
    emb = _hf_embedder
    batch = emb.embed_batch(["alpha", "beta", "gamma"])
    assert len(batch) == 3
    assert all(len(v) == 384 for v in batch)
    # Deterministic on same input (same call, same model state):
    assert emb.embed("alpha") == emb.embed("alpha")
    # Batch-vs-single can differ by small numerical noise due to kernel
    # batching; we assert they're numerically close rather than identical.
    import math

    single_alpha = emb.embed("alpha")
    diff = sum((a - b) ** 2 for a, b in zip(batch[0], single_alpha))
    assert math.sqrt(diff) < 0.01, f"batch/single drift too large: {diff}"


@requires_local_stack
def test_hf_embedder_via_registry(_hf_embedder):
    """Registry should expose HuggingFaceEmbedder when EMBEDDING_PROVIDER=hf."""
    from sql_agent.llm_serving.hf_embedder import HuggingFaceEmbedder

    registry.reset_caches()
    with registry._lock:  # type: ignore[attr-defined]
        registry._embed_cache["hf"] = _hf_embedder  # type: ignore[attr-defined]

    got = registry.get_embedding_provider("hf")
    assert isinstance(got, HuggingFaceEmbedder)
    assert got is _hf_embedder
    registry.reset_caches()


# ---------------------------------------------------------------------------
# MemoryManager: dim-namespaced paths + migration
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_faiss_dir(tmp_path, monkeypatch):
    """Isolated FAISS root for one MemoryManager test."""
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()

    # Route settings.faiss_dir to the tmp dir for this test.
    orig = settings.faiss_index_dir
    try:
        # settings.faiss_index_dir is the raw string; resolved_path handles it.
        settings.faiss_index_dir = str(faiss_dir)
        reset_memory_manager()
        yield faiss_dir
    finally:
        settings.faiss_index_dir = orig
        reset_memory_manager()


@pytest.fixture
def mock_embedder_env():
    """Route the registry to a MockEmbedder so dim is small and predictable."""
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


def test_memory_manager_uses_dim_namespaced_subdir(
    fresh_faiss_dir, mock_embedder_env
):
    mm = MemoryManager(root_dir=fresh_faiss_dir)
    assert mm.dim == 64  # MockEmbedder default
    assert mm.backend == "faiss"
    # Phase 9: the dim{N}/ subdirectory lives inside the FAISS backend,
    # not on MemoryManager.root (which now points at the base).
    assert (fresh_faiss_dir / "dim64").exists()


def test_memory_manager_records_and_recalls(fresh_faiss_dir, mock_embedder_env):
    """Write a reward and recall it with the SAME query text.

    We use the exact query (not a semantically-similar variant) because
    MockEmbedder is hash-based, not semantic — different strings produce
    vectors with essentially random cosine similarity. Recalling the exact
    input guarantees cosine ≈ 1.0 regardless of embedder.
    """
    mm = MemoryManager(root_dir=fresh_faiss_dir)
    assert mm.reward_size == 0

    query = "How many orders are there?"
    mm.record_reward(
        query,
        parameters={"table_names": ["orders"]},
        tool_used="count",
        reasoning="count rows",
    )
    assert mm.reward_size == 1

    rules = mm.recall(query, k_reward=3, k_penalty=3)
    rewards = [r for r in rules if r.get("kind") == "reward"]
    assert len(rewards) == 1
    assert rewards[0]["query"] == query
    assert rewards[0]["tool_used"] == "count"
    # Cosine similarity to itself is ~1.0 (passes default min_score=0.3).
    assert rewards[0]["score"] > 0.9


def test_memory_manager_migrates_legacy_files(tmp_path):
    """Pre-2.4 loose {name}.{index,jsonl} get moved into dim1536/."""
    # Create a legacy faiss_dir with loose files.
    legacy_root = tmp_path / "faiss"
    legacy_root.mkdir()
    (legacy_root / "reward.index").write_bytes(b"fake-index-bytes")
    (legacy_root / "reward.jsonl").write_text('{"kind": "reward"}\n')
    (legacy_root / "penalty.index").write_bytes(b"fake-penalty-bytes")

    _migrate_legacy_indices(legacy_root, legacy_dim=_LEGACY_DIM)

    target_dir = legacy_root / f"dim{_LEGACY_DIM}"
    assert target_dir.exists()
    assert (target_dir / "reward.index").exists()
    assert (target_dir / "reward.jsonl").exists()
    assert (target_dir / "penalty.index").exists()
    # Legacy paths should be gone.
    assert not (legacy_root / "reward.index").exists()
    assert not (legacy_root / "reward.jsonl").exists()
    assert not (legacy_root / "penalty.index").exists()


def test_memory_manager_migration_is_idempotent(tmp_path):
    """Running migration twice is a no-op on the second call."""
    root = tmp_path / "faiss"
    root.mkdir()
    (root / "reward.index").write_bytes(b"once")

    _migrate_legacy_indices(root)
    first = (root / "dim1536" / "reward.index").read_bytes()

    # Second call should do nothing: legacy file is gone, target exists.
    _migrate_legacy_indices(root)
    assert (root / "dim1536" / "reward.index").read_bytes() == first


def test_memory_manager_switch_dims_preserves_old_index(
    fresh_faiss_dir, monkeypatch
):
    """Writing to dim=64 and later to dim=128 produces two separate subdirs."""
    # Round 1: dim=64
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        registry.reset_caches()
        mm64 = MemoryManager(root_dir=fresh_faiss_dir)
        mm64.record_reward(
            "q64",
            parameters={"x": 1},
            tool_used="count",
        )
        assert (fresh_faiss_dir / "dim64" / "reward.index").exists()

        # Round 2: switch to 128-dim by replacing the registry entry.
        registry.reset_caches()
        from sql_agent.llm_serving.base import EmbeddingProvider  # noqa: F401
        with registry._lock:  # type: ignore[attr-defined]
            registry._embed_cache["mock"] = MockEmbedder(dim=128)  # type: ignore[attr-defined]

        mm128 = MemoryManager(root_dir=fresh_faiss_dir)
        assert mm128.dim == 128
        mm128.record_reward("q128", parameters={"x": 2}, tool_used="count")
        assert (fresh_faiss_dir / "dim128" / "reward.index").exists()

        # Old dim=64 index still intact.
        assert (fresh_faiss_dir / "dim64" / "reward.index").exists()
        assert mm64.reward_size == 1  # unchanged in-memory
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        registry.reset_caches()
        reset_memory_manager()


def test_faiss_store_rejects_wrong_dim_vector(fresh_faiss_dir, mock_embedder_env):
    """Direct add with a wrong-dim vector raises ValueError. This is the
    safety net that previously would silently corrupt the index.

    Phase 9: accesses the FAISS backend via the public store attribute
    instead of the old private ``_reward`` field.
    """
    mm = MemoryManager(root_dir=fresh_faiss_dir)
    with pytest.raises(ValueError, match="expected dim"):
        mm._store.add_reward([0.0] * 999, {"kind": "reward"})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Phase 2.4 closing integration: LLM_PROVIDER=hf + EMBEDDING_PROVIDER=hf.
# Verifies the whole stack wires together. Does NOT assert on correctness of
# the plan because SmolLM2 is too small for reliable JSON-following.
# ---------------------------------------------------------------------------


@requires_local_stack
@pytest.mark.timeout(600)
def test_hf_chat_plus_hf_embedding_end_to_end(fresh_faiss_dir, _hf_embedder):
    """Boot the registry with real HF chat + embedding providers and verify
    run_turn() either succeeds or fails with a clean pipeline error.

    SmolLM2-135M is too weak to reliably produce valid FetchParams JSON, so
    a ``final.error`` is an acceptable outcome — we just want no crashes
    and correct dim handling end-to-end.
    """
    import uuid

    from sql_agent.llm_serving.hf_provider import HuggingFaceProvider

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "hf"
        settings.embedding_provider = "hf"
        registry.reset_caches()
        with registry._lock:  # type: ignore[attr-defined]
            registry._llm_cache["hf"] = HuggingFaceProvider(  # type: ignore[attr-defined]
                model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                device="cpu",
                max_new_tokens=64,
            )
            registry._embed_cache["hf"] = _hf_embedder  # type: ignore[attr-defined]
        reset_memory_manager()

        from sql_agent.agents.orchestrator import run_turn

        final = run_turn(
            "count orders",
            session_id=str(uuid.uuid4()),
            prior_messages=[],
        )

        # Either succeeded or failed cleanly. Crash = bad; unhandled exception
        # would already have propagated out of run_turn.
        assert isinstance(final, dict)
        assert "tool_used" in final or "error" in final

        # FAISS dir should be namespaced by the HF embedder dim (384).
        assert (Path(fresh_faiss_dir) / "dim384").exists() or final.get("error")
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        registry.reset_caches()
        reset_memory_manager()
