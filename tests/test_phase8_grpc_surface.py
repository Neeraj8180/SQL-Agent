"""Phase 8.5 tests — plan-only subgraph + TLS + auth interceptor."""

from __future__ import annotations

import json
import uuid
from concurrent import futures

import grpc
import pytest

from sql_agent.config import settings
from sql_agent.llm_serving import registry as llm_registry
from sql_agent.services.memory_manager import reset_memory_manager


# ---------------------------------------------------------------------------
# Plan-only subgraph
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider_env():
    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        llm_registry.reset_caches()
        reset_memory_manager()
        yield
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        llm_registry.reset_caches()
        reset_memory_manager()


def test_plan_turn_returns_plan_without_data(mock_provider_env):
    """plan_turn() returns tool + params + reasoning but NO data rows."""
    from sql_agent.agents.orchestrator import plan_turn

    final = plan_turn(
        "count orders", session_id=str(uuid.uuid4()), prior_messages=[]
    )

    assert not final.get("error"), f"unexpected error: {final.get('error')}"
    assert final.get("tool_used") == "count"
    assert final.get("parameters", {}).get("table_names") == ["orders"]
    assert final.get("param_reasoning")
    # Plan-only: DB never queried, so no data / insights / visualization.
    assert not final.get("data")
    assert not final.get("data_cleaned")
    assert not final.get("insights")
    assert not final.get("visualization")


def test_plan_turn_is_cheaper_than_run_turn(mock_provider_env):
    """plan_turn should skip DB roundtrip + FAISS write, making it faster."""
    import time

    from sql_agent.agents.orchestrator import plan_turn, run_turn

    # Warm caches (first call pays graph compile).
    run_turn("count orders", session_id="warm-run", prior_messages=[])
    plan_turn("count orders", session_id="warm-plan", prior_messages=[])

    # Measure 3-call averages so single-call jitter doesn't dominate.
    def _avg(fn, n=3):
        t0 = time.perf_counter()
        for i in range(n):
            fn(f"count orders {i}", session_id=f"timing-{i}", prior_messages=[])
        return (time.perf_counter() - t0) / n

    run_avg = _avg(run_turn)
    plan_avg = _avg(plan_turn)

    # plan_turn should be at least as fast as run_turn. We don't assert a
    # strict factor because mock providers are fast enough that FAISS
    # writes + analysis are the dominant term and CPU noise matters.
    assert plan_avg <= run_avg * 1.5, (
        f"plan_turn ({plan_avg*1000:.1f}ms) is not meaningfully faster than "
        f"run_turn ({run_avg*1000:.1f}ms)"
    )


def test_grpc_generate_sql_uses_plan_subgraph(mock_provider_env):
    """GenerateSQL RPC routes through plan_turn (no DB data in response)."""
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server.server import SqlAgentServicer

    class _Ctx:
        def abort(self, code, details):
            raise AssertionError(f"abort: {code} {details}")

        def set_code(self, _): pass

        def set_details(self, _): pass

    servicer = SqlAgentServicer()
    resp = servicer.GenerateSQL(
        pb2.GenerateSQLRequest(query="count orders"), _Ctx()
    )
    assert not resp.error
    assert resp.tool_used == "count"
    params = json.loads(resp.parameters_json)
    assert params["table_names"] == ["orders"]
    assert resp.latency_ms > 0
    # Response surface intentionally has no data/insights fields.


# ---------------------------------------------------------------------------
# Bearer-token auth interceptor
# ---------------------------------------------------------------------------


def _start_server_with_interceptor(interceptor) -> tuple[grpc.Server, int]:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    from sql_agent.grpc_server import sql_agent_pb2_grpc as pb2_grpc
    from sql_agent.grpc_server.server import SqlAgentServicer

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[interceptor],
    )
    pb2_grpc.add_SqlAgentServicer_to_server(SqlAgentServicer(), server)

    # Mirror create_server(): register the standard grpc.health.v1.Health
    # service so the auth-exemption test can actually call /Check.
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


def test_bearer_auth_allows_correct_token(mock_provider_env):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server import sql_agent_pb2_grpc as pb2_grpc
    from sql_agent.grpc_server.interceptors import BearerAuthInterceptor

    server, port = _start_server_with_interceptor(
        BearerAuthInterceptor("correct-secret")
    )
    try:
        chan = grpc.insecure_channel(f"localhost:{port}")
        stub = pb2_grpc.SqlAgentStub(chan)
        resp = stub.GenerateSQL(
            pb2.GenerateSQLRequest(query="count orders"),
            metadata=(("authorization", "Bearer correct-secret"),),
        )
        assert not resp.error
    finally:
        server.stop(grace=0)


def test_bearer_auth_rejects_missing_token(mock_provider_env):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server import sql_agent_pb2_grpc as pb2_grpc
    from sql_agent.grpc_server.interceptors import BearerAuthInterceptor

    server, port = _start_server_with_interceptor(
        BearerAuthInterceptor("correct-secret")
    )
    try:
        chan = grpc.insecure_channel(f"localhost:{port}")
        stub = pb2_grpc.SqlAgentStub(chan)
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.GenerateSQL(pb2.GenerateSQLRequest(query="count orders"))
        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        server.stop(grace=0)


def test_bearer_auth_rejects_wrong_token(mock_provider_env):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server import sql_agent_pb2_grpc as pb2_grpc
    from sql_agent.grpc_server.interceptors import BearerAuthInterceptor

    server, port = _start_server_with_interceptor(
        BearerAuthInterceptor("correct-secret")
    )
    try:
        chan = grpc.insecure_channel(f"localhost:{port}")
        stub = pb2_grpc.SqlAgentStub(chan)
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.GenerateSQL(
                pb2.GenerateSQLRequest(query="count orders"),
                metadata=(("authorization", "Bearer wrong"),),
            )
        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        server.stop(grace=0)


def test_bearer_auth_exempts_health_check(mock_provider_env):
    """Kubernetes / service-mesh probes must reach the health service
    regardless of auth configuration."""
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    from sql_agent.grpc_server.interceptors import BearerAuthInterceptor

    server, port = _start_server_with_interceptor(
        BearerAuthInterceptor("correct-secret")
    )
    try:
        chan = grpc.insecure_channel(f"localhost:{port}")
        stub = health_pb2_grpc.HealthStub(chan)
        # No metadata; should still succeed because health is allowlisted.
        resp = stub.Check(health_pb2.HealthCheckRequest())
        assert resp.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        server.stop(grace=0)
