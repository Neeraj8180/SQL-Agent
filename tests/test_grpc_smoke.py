"""Phase 1 smoke tests for the gRPC gateway.

We run the servicer in-process (no network port) to keep CI deterministic and
fast. The orchestrator still executes its full graph — only the LLM calls are
faked via the `fake_llm` fixture in `tests/conftest.py`.

Coverage:
    - HealthCheck
    - GenerateSQL (plan-only fields; no rows in response)
    - ExecuteSQL  (end-to-end against the seeded SQLite DB)
    - Invalid-input guard
"""

from __future__ import annotations

import json

import grpc
import pytest

from sql_agent.models import FetchParams, Intent


# ---------------------------------------------------------------------------
# Fake-intent factories used across tests.
# ---------------------------------------------------------------------------


def _intent_count(cls):
    return cls(
        metrics=["order_count"],
        dimensions=[],
        filters=[],
        time_range=None,
        output_type="count",
        visualize=False,
        notes="count of orders",
    )


def _plan_count(cls):
    return cls(
        reasoning="Count all rows in 'orders'.",
        params=FetchParams(
            table_names=["orders"],
            aggregations=[{"func": "count", "column": "*", "alias": "count"}],
            limit=100,
        ),
    )


COUNT_FACTORIES = {"Intent": _intent_count, "ParamPlan": _plan_count}


# ---------------------------------------------------------------------------
# Servicer fixture — built AFTER fake_llm has patched the LLM seams.
# ---------------------------------------------------------------------------


@pytest.fixture
def servicer(fake_llm):
    fake_llm(COUNT_FACTORIES)
    # Import here so that the orchestrator's `build_graph()` sees the patched
    # LLM module (it doesn't actually care, but keeps the import order
    # explicit for future readers).
    from sql_agent.grpc_server.server import SqlAgentServicer

    return SqlAgentServicer()


class _FakeContext:
    """Minimal grpc.ServicerContext stand-in for in-process tests."""

    def __init__(self):
        self.code = None
        self.details = None

    def abort(self, code, details):
        self.code = code
        self.details = details
        raise grpc.RpcError(details)

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthcheck_reports_serving(servicer):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2

    resp = servicer.HealthCheck(pb2.HealthCheckRequest(), _FakeContext())
    assert resp.status == "SERVING"
    assert resp.version  # non-empty


def test_execute_sql_count_returns_seeded_row_count(servicer):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2

    resp = servicer.ExecuteSQL(
        pb2.ExecuteSQLRequest(query="How many orders are there?"),
        _FakeContext(),
    )

    assert not resp.error, f"unexpected error: {resp.error}"
    assert resp.success is True
    assert resp.tool_used == "count"
    assert resp.row_count == 1

    rows = json.loads(resp.rows_json)
    assert len(rows) == 1
    assert int(rows[0]["count"]) > 0  # seeded DB has ~3000+ orders
    assert resp.session_id  # server generated one
    assert resp.latency_ms > 0


def test_generate_sql_returns_plan_without_data(servicer):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2

    resp = servicer.GenerateSQL(
        pb2.GenerateSQLRequest(query="How many orders are there?"),
        _FakeContext(),
    )

    assert not resp.error
    assert resp.tool_used == "count"
    assert resp.param_reasoning
    params = json.loads(resp.parameters_json)
    assert params["table_names"] == ["orders"]
    assert params["aggregations"][0]["func"] == "count"


def test_execute_sql_rejects_empty_query(servicer):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2

    ctx = _FakeContext()
    with pytest.raises(grpc.RpcError):
        servicer.ExecuteSQL(pb2.ExecuteSQLRequest(query="   "), ctx)
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


def test_session_id_is_echoed(servicer):
    from sql_agent.grpc_server import sql_agent_pb2 as pb2

    resp = servicer.ExecuteSQL(
        pb2.ExecuteSQLRequest(
            query="How many orders are there?",
            session_id="fixed-session-abc",
        ),
        _FakeContext(),
    )
    assert resp.session_id == "fixed-session-abc"
