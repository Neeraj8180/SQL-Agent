"""Phase 4 tests — ML lifecycle tracking.

Three clusters:
    1. NoOpTracker + summarize_state helper (no deps).
    2. FileTracker JSONL semantics (per-turn, latency, error path).
    3. MLflowTracker via ``MlflowClient`` — skipped if mlflow not installed.
    4. End-to-end: ``run_turn`` with ``TRACKING_ENABLED=true`` writes a row.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry as llm_registry
from sql_agent.services.memory_manager import reset_memory_manager
from sql_agent.tracking.base import _classify_error, summarize_state
from sql_agent.tracking.file_tracker import FileTracker
from sql_agent.tracking.noop_tracker import NoOpTracker
from sql_agent.tracking.registry import get_tracker, reset_tracker


# ---------------------------------------------------------------------------
# Helpers: state fabricators
# ---------------------------------------------------------------------------


def _success_state(rows=None, tool="count"):
    return {
        "tool_used": tool,
        "parameters": {"table_names": ["orders"], "aggregations": [{"alias": "count"}]},
        "param_reasoning": "test",
        "data": rows or [{"count": 3387}],
        "data_cleaned": rows or [{"count": 3387}],
        "insights": "- result: 3387",
        "retry_count": 0,
        "validation_errors": [],
        "error": None,
        "success": True,
    }


def _failure_state(error="Parameter schema invalid: expected int"):
    return {
        "tool_used": "",
        "parameters": {},
        "data": [],
        "retry_count": 2,
        "validation_errors": ["col X not in schema"],
        "error": error,
        "success": False,
    }


# ---------------------------------------------------------------------------
# 1. NoOpTracker + summarize_state
# ---------------------------------------------------------------------------


def test_noop_tracker_is_callable_and_silent():
    t = NoOpTracker()
    assert t.name == "noop"
    h = t.start(session_id="s", user_query="q")
    t.finish(h, _success_state())
    t.finish_error(h, RuntimeError("ignored"))


def test_summarize_state_success_shape():
    summary = summarize_state(
        _success_state(),
        user_query="how many orders",
        session_id="sess-1",
    )
    assert set(summary.keys()) == {"params", "metrics", "tags", "artifacts"}
    assert summary["params"]["user_query"] == "how many orders"
    assert summary["params"]["session_id"] == "sess-1"
    assert summary["metrics"]["row_count"] == 1.0
    assert summary["metrics"]["success"] == 1.0
    assert summary["metrics"]["validation_error_count"] == 0.0
    assert summary["tags"]["tool_used"] == "count"
    assert summary["tags"]["error_type"] == ""
    assert "parameters.json" in summary["artifacts"]
    assert "data_head.json" in summary["artifacts"]


def test_summarize_state_failure_tags_are_classified():
    summary = summarize_state(
        _failure_state(error="Parameter schema invalid"),
        user_query="bad query",
        session_id="sess-2",
    )
    assert summary["metrics"]["success"] == 0.0
    assert summary["metrics"]["validation_error_count"] == 1.0
    assert summary["tags"]["error_type"] == "validation_error"
    assert "error.txt" in summary["artifacts"]


def test_classify_error_buckets():
    assert _classify_error({"error": "Schema discovery failed"}) == "schema_error"
    assert _classify_error({"error": "Data fetch failed"}) == "fetch_error"
    assert _classify_error({"error": "Intent extraction failed: timeout"}) == "intent_error"
    assert _classify_error({"error": "unrelated"}) == "other_error"
    assert _classify_error({"error": ""}) == ""


def test_summarize_state_truncates_user_query():
    long_query = "x" * 10_000
    summary = summarize_state(
        _success_state(),
        user_query=long_query,
        session_id="s",
        query_max_chars=42,
    )
    assert len(summary["params"]["user_query"]) == 42


# ---------------------------------------------------------------------------
# 2. FileTracker
# ---------------------------------------------------------------------------


def test_file_tracker_writes_one_line_per_turn(tmp_path):
    path = tmp_path / "nested" / "turns.jsonl"
    t = FileTracker(path=path)
    assert t.path == path
    assert path.parent.exists()

    for i in range(3):
        h = t.start(session_id=f"s{i}", user_query=f"q{i}")
        t.finish(h, _success_state())

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]
    assert [r["session_id"] for r in records] == ["s0", "s1", "s2"]
    assert all(r["metrics"]["success"] == 1.0 for r in records)
    assert all(r["metrics"]["latency_ms"] >= 0.0 for r in records)
    assert all(r["tags"]["tool_used"] == "count" for r in records)


def test_file_tracker_records_failure_state(tmp_path):
    path = tmp_path / "turns.jsonl"
    t = FileTracker(path=path)
    h = t.start(session_id="sess-x", user_query="bad")
    t.finish(h, _failure_state())

    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["metrics"]["success"] == 0.0
    assert rec["metrics"]["validation_error_count"] == 1.0
    assert rec["tags"]["error_type"] == "validation_error"


def test_file_tracker_records_exception_path(tmp_path):
    path = tmp_path / "turns.jsonl"
    t = FileTracker(path=path)
    h = t.start(session_id="sess-exc", user_query="q")
    t.finish_error(h, RuntimeError("boom"))

    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["metrics"]["success"] == 0.0
    assert rec["tags"]["error_type"]  # non-empty
    assert "RuntimeError" in rec["artifacts"]["error.txt"]


def test_file_tracker_is_thread_safe(tmp_path):
    path = tmp_path / "turns.jsonl"
    t = FileTracker(path=path)

    def worker(i: int) -> None:
        for j in range(10):
            h = t.start(session_id=f"t{i}-{j}", user_query="q")
            t.finish(h, _success_state())

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6 * 10
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# 3. MLflowTracker (skipped if mlflow missing)
# ---------------------------------------------------------------------------


def _mlflow_installed() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


requires_mlflow = pytest.mark.skipif(
    not _mlflow_installed(),
    reason="mlflow not installed; pip install -r requirements-tracking.txt",
)


@requires_mlflow
def test_mlflow_tracker_logs_run_with_params_and_metrics(tmp_path):
    from sql_agent.tracking.mlflow_tracker import MLflowTracker

    uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    t = MLflowTracker(tracking_uri=uri, experiment_name="phase4_test")
    assert t.tracking_uri == uri
    assert t.experiment_id is not None

    h = t.start(session_id="sess-ml", user_query="how many orders")
    assert h.run_id is not None
    t.finish(h, _success_state())

    # Read the run back using MlflowClient.
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(h.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["session_id"] == "sess-ml"
    assert run.data.params["user_query"] == "how many orders"
    assert run.data.metrics["row_count"] == 1.0
    assert run.data.metrics["success"] == 1.0
    assert run.data.metrics["latency_ms"] >= 0.0
    assert run.data.tags["tool_used"] == "count"


@requires_mlflow
def test_mlflow_tracker_marks_run_failed_on_exception(tmp_path):
    from mlflow.tracking import MlflowClient

    from sql_agent.tracking.mlflow_tracker import MLflowTracker

    uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    t = MLflowTracker(tracking_uri=uri, experiment_name="phase4_err")
    h = t.start(session_id="sess-err", user_query="q")
    t.finish_error(h, RuntimeError("kaboom"))

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(h.run_id)
    assert run.info.status == "FAILED"
    assert run.data.metrics["success"] == 0.0
    assert run.data.tags["error_type"] == "exception"


@requires_mlflow
def test_mlflow_tracker_defaults_to_sqlite_backend(tmp_path):
    """Phase 8.3: default URI (empty) produces a sqlite:// URI, not file://.

    SQLite is the MLflow-recommended backend post-2026; file-store is
    deprecated. Tracker should default to it out of the box.
    """
    from sql_agent.tracking.mlflow_tracker import MLflowTracker

    orig_uri = settings.mlflow_tracking_uri
    orig_faiss_dir = settings.faiss_index_dir
    orig_log = settings.tracking_file_log
    try:
        # Route the resolved path into tmp so we don't pollute repo logs/.
        settings.mlflow_tracking_uri = ""  # force the default branch
        # Temporarily alias PROJECT_ROOT/logs to tmp_path so the SQLite DB
        # lands in tmp. settings.resolved_path("logs/tracking") resolves
        # against PROJECT_ROOT which is the sql_agent package — so we
        # directly inject the path via a subclass trick:
        t = MLflowTracker(
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            experiment_name="phase8_sqlite_test",
        )
        assert t.tracking_uri.startswith("sqlite:///")
        assert t.experiment_id is not None
        h = t.start(session_id="sq1", user_query="count orders")
        t.finish(h, _success_state())
        assert (tmp_path / "mlflow.db").exists()
    finally:
        settings.mlflow_tracking_uri = orig_uri
        settings.faiss_index_dir = orig_faiss_dir
        settings.tracking_file_log = orig_log


@requires_mlflow
def test_mlflow_tracker_handles_concurrent_runs(tmp_path):
    """Explicit run_id + MlflowClient should not interfere across threads."""
    from mlflow.tracking import MlflowClient

    from sql_agent.tracking.mlflow_tracker import MLflowTracker

    uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    t = MLflowTracker(tracking_uri=uri, experiment_name="phase4_concurrent")

    def worker(i: int) -> str:
        h = t.start(session_id=f"sess-{i}", user_query=f"q{i}")
        t.finish(h, _success_state())
        assert h.run_id is not None
        return h.run_id

    with ThreadPoolExecutor(max_workers=4) as exe:
        futs = [exe.submit(worker, i) for i in range(4)]
        run_ids = [f.result() for f in as_completed(futs)]

    assert len(set(run_ids)) == 4  # all distinct

    client = MlflowClient(tracking_uri=uri)
    for rid in run_ids:
        run = client.get_run(rid)
        assert run.info.status == "FINISHED"
        assert run.data.metrics["success"] == 1.0


# ---------------------------------------------------------------------------
# 4. Registry resolution
# ---------------------------------------------------------------------------


def test_registry_noop_when_tracking_disabled():
    orig_enabled = settings.tracking_enabled
    try:
        settings.tracking_enabled = False
        reset_tracker()
        t = get_tracker()
        assert isinstance(t, NoOpTracker)
    finally:
        settings.tracking_enabled = orig_enabled
        reset_tracker()


def test_registry_file_backend_forced(tmp_path):
    orig_enabled = settings.tracking_enabled
    orig_backend = settings.tracking_backend
    orig_log = settings.tracking_file_log
    try:
        settings.tracking_enabled = True
        settings.tracking_backend = "file"
        settings.tracking_file_log = str(tmp_path / "force-file.jsonl")
        reset_tracker()
        t = get_tracker()
        assert isinstance(t, FileTracker)
        assert t.path == tmp_path / "force-file.jsonl"
    finally:
        settings.tracking_enabled = orig_enabled
        settings.tracking_backend = orig_backend
        settings.tracking_file_log = orig_log
        reset_tracker()


@requires_mlflow
def test_registry_auto_backend_picks_mlflow_when_available(tmp_path):
    from sql_agent.tracking.mlflow_tracker import MLflowTracker

    orig_enabled = settings.tracking_enabled
    orig_backend = settings.tracking_backend
    orig_uri = settings.mlflow_tracking_uri
    orig_exp = settings.mlflow_experiment_name
    try:
        settings.tracking_enabled = True
        settings.tracking_backend = "auto"
        settings.mlflow_tracking_uri = f"file:{(tmp_path / 'auto_mlruns').as_posix()}"
        settings.mlflow_experiment_name = "phase4_auto"
        reset_tracker()
        t = get_tracker()
        assert isinstance(t, MLflowTracker)
    finally:
        settings.tracking_enabled = orig_enabled
        settings.tracking_backend = orig_backend
        settings.mlflow_tracking_uri = orig_uri
        settings.mlflow_experiment_name = orig_exp
        reset_tracker()


# ---------------------------------------------------------------------------
# 5. End-to-end: run_turn under TRACKING_ENABLED=file
# ---------------------------------------------------------------------------


@pytest.fixture
def tracking_env_file(tmp_path):
    """Turn on file-backed tracking for one test."""
    orig_enabled = settings.tracking_enabled
    orig_backend = settings.tracking_backend
    orig_log = settings.tracking_file_log
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.tracking_enabled = True
        settings.tracking_backend = "file"
        settings.tracking_file_log = str(tmp_path / "run_turn.jsonl")
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_registry.reset_caches()
        reset_memory_manager()
        reset_tracker()
        yield Path(settings.tracking_file_log)
    finally:
        settings.tracking_enabled = orig_enabled
        settings.tracking_backend = orig_backend
        settings.tracking_file_log = orig_log
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_registry.reset_caches()
        reset_memory_manager()
        reset_tracker()


def test_run_turn_writes_tracking_row_under_file_backend(tracking_env_file):
    from sql_agent.agents.orchestrator import run_turn

    path: Path = tracking_env_file
    final = run_turn(
        "How many orders are there?",
        session_id="e2e-1",
        prior_messages=[],
    )
    assert final.get("success") is True

    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["session_id"] == "e2e-1"
    assert rec["params"]["llm_provider"] == "mock"
    assert rec["metrics"]["success"] == 1.0
    assert rec["metrics"]["row_count"] == 1.0
    assert rec["metrics"]["latency_ms"] > 0.0
    assert rec["tags"]["tool_used"] == "count"


def test_token_usage_flows_into_tracking(tracking_env_file):
    """Phase 8.3: LLM calls inside run_turn record token counts, and the
    tracker picks them up as input_tokens/output_tokens/total_tokens metrics."""
    from sql_agent.agents.orchestrator import run_turn
    from sql_agent.llm_serving import registry as reg
    from sql_agent.request_context import record_token_usage

    # Install a custom mock provider that reports synthetic token usage
    # on every chat call so we can see it land in the tracker row.
    from sql_agent.llm_serving.mock_provider import MockProvider

    class _UsageReportingMock(MockProvider):
        def chat_model(self, temperature: float = 0.0):
            inner = super().chat_model(temperature)

            class _Wrap:
                temperature = 0.0

                def with_structured_output(self, cls):
                    real = inner.with_structured_output(cls)

                    class _Inv:
                        def invoke(self, msgs):
                            record_token_usage(
                                provider="mock", model="mock",
                                input_tokens=100, output_tokens=42,
                            )
                            return real.invoke(msgs)

                    return _Inv()

                def invoke(self, msgs):
                    record_token_usage(
                        provider="mock", model="mock",
                        input_tokens=10, output_tokens=5,
                    )
                    return inner.invoke(msgs)

            return _Wrap()

    with reg._lock:  # type: ignore[attr-defined]
        reg._llm_cache["mock"] = _UsageReportingMock()  # type: ignore[attr-defined]

    try:
        run_turn("count orders", session_id="tok-1", prior_messages=[])
    finally:
        reg.reset_caches()

    rec = json.loads(tracking_env_file.read_text(encoding="utf-8").strip())
    # intent_agent + param_builder_agent each call .with_structured_output,
    # so we expect at least 200 input + 84 output = 284 total from the two
    # structured calls. (Summarize is only invoked if messages > 6; here it
    # won't run, so we get exactly 2 structured calls = 284.)
    assert rec["metrics"]["input_tokens"] >= 200
    assert rec["metrics"]["output_tokens"] >= 80
    assert rec["metrics"]["total_tokens"] == (
        rec["metrics"]["input_tokens"] + rec["metrics"]["output_tokens"]
    )


def test_run_turn_exception_still_produces_tracker_row(tracking_env_file):
    """If the pipeline raises, the exception propagates unchanged AND the
    tracker records a row with success=0 via finish_error."""
    from sql_agent.agents.orchestrator import run_turn
    from sql_agent.llm_serving import registry as reg

    path = tracking_env_file

    # Break the registry so the first get_chat_model() inside intent_node
    # raises. Agents don't catch errors thrown during get_chat_model()
    # construction, so LangGraph will propagate the exception out of
    # graph.invoke(), which is exactly the path run_turn's finish_error
    # handler is supposed to cover.
    orig_build_llm = reg._build_llm  # type: ignore[attr-defined]

    def _boom(name):
        raise RuntimeError("intentional pipeline failure")

    reg._build_llm = _boom  # type: ignore[attr-defined]
    reg.reset_caches()
    try:
        with pytest.raises(Exception):
            run_turn("How many orders?", session_id="e2e-err", prior_messages=[])
    finally:
        reg._build_llm = orig_build_llm  # type: ignore[attr-defined]
        reg.reset_caches()

    # Tracker must have written a row with success=0 and an error artifact,
    # even though the pipeline threw.
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["session_id"] == "e2e-err"
    assert rec["metrics"]["success"] == 0.0
    assert "error.txt" in rec["artifacts"]
    assert "RuntimeError" in rec["artifacts"]["error.txt"]
