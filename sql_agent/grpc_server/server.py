"""gRPC servicer wrapping the existing `run_turn` orchestrator.

Design constraints (phase 1):
    - ZERO changes to orchestrator, agents, tools, or services.
    - `run_turn` is the only entrypoint we call.
    - LLM calls, DB access, FAISS writes — all handled by existing code.
    - Server is additive: not importing this module leaves the rest of the
      system byte-for-byte identical.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent import futures
from typing import Any, Dict, List, Optional

import grpc

from sql_agent import __version__ as _pkg_version
from sql_agent.config import configure_logging, get_logger, settings
from sql_agent.models.graph_state import ChatMessage
from sql_agent.request_context import request_scope, session_scope

from . import sql_agent_pb2 as pb2
from . import sql_agent_pb2_grpc as pb2_grpc


def _record_rpc_metrics(rpc: str, status: str, latency_ms: float) -> None:
    """Emit Prometheus counters + histogram. Never raises."""
    try:
        from sql_agent.observability.metrics import get_metrics

        get_metrics().record_rpc(rpc, status, latency_ms)
    except Exception as exc:  # pragma: no cover — defensive
        _log.debug("metrics record_rpc failed: %s", exc)


def _record_turn_metrics(state: Dict[str, Any]) -> None:
    """Record tool-used and validation-error metrics from final state."""
    try:
        from sql_agent.observability.metrics import get_metrics

        m = get_metrics()
        tool = state.get("tool_used") or ""
        if tool:
            m.record_tool(tool)
        ve = state.get("validation_errors") or []
        if ve:
            m.record_validation_errors(len(ve))
    except Exception as exc:  # pragma: no cover
        _log.debug("metrics record_turn failed: %s", exc)


_log = get_logger("grpc_server")


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class SqlAgentServicer(pb2_grpc.SqlAgentServicer):
    """Thin gRPC adapter over `sql_agent.agents.orchestrator.run_turn`.

    The servicer is stateless (per Phase 7 readiness). All state lives in:
      - SQLAlchemy engine (cached in `services.db`)
      - FAISS memory stores (on-disk in settings.faiss_dir)
      - Compiled LangGraph (lru_cache in `orchestrator.build_graph`)
    """

    def __init__(self) -> None:
        # Import here so that unit tests can monkey-patch LLM seams BEFORE
        # the orchestrator module gets imported (its agent modules capture
        # `get_chat_model` at import time via `from sql_agent.services.llm
        # import get_chat_model`).
        from sql_agent.agents.orchestrator import (  # noqa: F401
            build_graph,
            build_plan_graph,
            plan_turn,
            run_turn,
        )

        # Warm both compiled graphs so first requests are not slow.
        build_graph()
        build_plan_graph()
        self._run_turn = run_turn
        self._plan_turn = plan_turn
        _log.info("SqlAgentServicer ready (sql_agent v%s)", _pkg_version)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_id(raw: str) -> str:
        return raw.strip() or str(uuid.uuid4())

    @staticmethod
    def _prior_messages(proto_msgs) -> List[ChatMessage]:
        out: List[ChatMessage] = []
        for m in proto_msgs:
            role = (m.role or "user").strip() or "user"
            out.append({"role": role, "content": m.content or ""})
        return out

    def _invoke(
        self,
        *,
        query: str,
        session_id: str,
        memory_summary: str,
        prior_messages: List[ChatMessage],
    ) -> Dict[str, Any]:
        # `run_turn` already handles empty / None summary.
        return self._run_turn(
            query,
            session_id=session_id,
            prior_messages=prior_messages,
            memory_summary=memory_summary or None,
        )

    def _invoke_plan(
        self,
        *,
        query: str,
        session_id: str,
        memory_summary: str,
        prior_messages: List[ChatMessage],
    ) -> Dict[str, Any]:
        """Plan-only variant — used by GenerateSQL. Skips DB execution."""
        return self._plan_turn(
            query,
            session_id=session_id,
            prior_messages=prior_messages,
            memory_summary=memory_summary or None,
        )

    @staticmethod
    def _params_json(state: Dict[str, Any]) -> str:
        return json.dumps(state.get("parameters") or {}, default=str)

    @staticmethod
    def _rows_json(state: Dict[str, Any]) -> str:
        rows = state.get("data_cleaned") or state.get("data") or []
        return json.dumps(rows, default=str)

    # ------------------------------------------------------------------
    # RPC implementations
    # ------------------------------------------------------------------

    def GenerateSQL(
        self, request: pb2.GenerateSQLRequest, context: grpc.ServicerContext
    ) -> pb2.GenerateSQLResponse:
        if not request.query or not request.query.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")

        session_id = self._session_id(request.session_id)
        t0 = time.perf_counter()
        # Bind session_id (phase 3) AND a fresh request_id (phase 5) for
        # this RPC's scope. Both propagate via contextvars to downstream
        # services without touching agent code.
        with session_scope(session_id), request_scope() as request_id:
            try:
                # Phase 8.5: plan-only subgraph. Skips DB roundtrip /
                # analysis / memory writes — ~2x faster than ExecuteSQL.
                state = self._invoke_plan(
                    query=request.query,
                    session_id=session_id,
                    memory_summary=request.memory_summary,
                    prior_messages=self._prior_messages(request.prior_messages),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                _log.exception(
                    "GenerateSQL failed",
                    extra={
                        "rpc": "GenerateSQL",
                        "request_id": request_id,
                        "latency_ms": latency_ms,
                    },
                )
                _record_rpc_metrics("GenerateSQL", "error", latency_ms)
                return pb2.GenerateSQLResponse(
                    error=f"pipeline error: {exc}",
                    latency_ms=latency_ms,
                    session_id=session_id,
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "success" if not state.get("error") else "pipeline_error"
            _log.info(
                "GenerateSQL complete",
                extra={
                    "rpc": "GenerateSQL",
                    "request_id": request_id,
                    "tool_used": state.get("tool_used"),
                    "error": state.get("error"),
                    "latency_ms": latency_ms,
                },
            )
            _record_rpc_metrics("GenerateSQL", status, latency_ms)
            _record_turn_metrics(state)
            return pb2.GenerateSQLResponse(
                tool_used=state.get("tool_used") or "",
                parameters_json=self._params_json(state),
                param_reasoning=state.get("param_reasoning") or "",
                error=state.get("error") or "",
                latency_ms=latency_ms,
                session_id=session_id,
            )

    def ExecuteSQL(
        self, request: pb2.ExecuteSQLRequest, context: grpc.ServicerContext
    ) -> pb2.ExecuteSQLResponse:
        if not request.query or not request.query.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")

        session_id = self._session_id(request.session_id)
        t0 = time.perf_counter()
        with session_scope(session_id), request_scope() as request_id:
            try:
                state = self._invoke(
                    query=request.query,
                    session_id=session_id,
                    memory_summary=request.memory_summary,
                    prior_messages=self._prior_messages(request.prior_messages),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                _log.exception(
                    "ExecuteSQL failed",
                    extra={
                        "rpc": "ExecuteSQL",
                        "request_id": request_id,
                        "latency_ms": latency_ms,
                    },
                )
                _record_rpc_metrics("ExecuteSQL", "error", latency_ms)
                return pb2.ExecuteSQLResponse(
                    error=f"pipeline error: {exc}",
                    success=False,
                    latency_ms=latency_ms,
                    session_id=session_id,
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            rows = state.get("data_cleaned") or state.get("data") or []
            success = bool(state.get("success")) and not state.get("error")
            status = "success" if success else "pipeline_error"

            _log.info(
                "ExecuteSQL complete",
                extra={
                    "rpc": "ExecuteSQL",
                    "request_id": request_id,
                    "tool_used": state.get("tool_used"),
                    "row_count": len(rows),
                    "success": success,
                    "latency_ms": latency_ms,
                },
            )
            _record_rpc_metrics("ExecuteSQL", status, latency_ms)
            _record_turn_metrics(state)
            return pb2.ExecuteSQLResponse(
                tool_used=state.get("tool_used") or "",
                parameters_json=self._params_json(state),
                param_reasoning=state.get("param_reasoning") or "",
                row_count=len(rows),
                rows_json=self._rows_json(state),
                insights=state.get("insights") or "",
                visualization_b64=state.get("visualization") or "",
                error=state.get("error") or "",
                success=success,
                latency_ms=latency_ms,
                session_id=session_id,
            )

    def HealthCheck(
        self, request: pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> pb2.HealthCheckResponse:
        return pb2.HealthCheckResponse(status="SERVING", version=_pkg_version)


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------


def create_server(
    port: Optional[int] = None,
    *,
    max_workers: int = 10,
) -> grpc.Server:
    """Build (but do NOT start) a gRPC server with SqlAgentServicer attached.

    Phase 8.5 additions (all opt-in via settings):
        * TLS when ``GRPC_TLS_CERT_FILE`` + ``GRPC_TLS_KEY_FILE`` are set.
        * Bearer-token auth interceptor when ``GRPC_AUTH_TOKEN`` is set.
    """
    configure_logging()

    interceptors = []
    if settings.grpc_auth_token:
        from .interceptors import BearerAuthInterceptor

        interceptors.append(BearerAuthInterceptor(settings.grpc_auth_token))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
        interceptors=interceptors or None,
    )
    pb2_grpc.add_SqlAgentServicer_to_server(SqlAgentServicer(), server)

    # Phase 6: register the standard gRPC health-check service so tools
    # like `grpc_health_probe` / Kubernetes gRPC probes can check liveness
    # without knowing our custom RPC surface. Best-effort: if
    # grpcio-health-checking isn't installed we log and skip.
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        # Empty string key = overall service status (grpc_health_probe default).
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(
            "sql_agent.v1.SqlAgent", health_pb2.HealthCheckResponse.SERVING
        )
        _log.info("Standard grpc.health.v1.Health service registered")
    except ImportError as exc:
        _log.warning(
            "grpcio-health-checking not installed; grpc_health_probe will "
            "not work. Install requirements-grpc.txt to enable. (%s)",
            exc,
        )

    bind_port = port if port is not None else settings.grpc_port

    # Phase 8.5: TLS when both cert + key are configured; insecure otherwise.
    cert_file = settings.grpc_tls_cert_file
    key_file = settings.grpc_tls_key_file
    if cert_file and key_file:
        try:
            cert_bytes = open(cert_file, "rb").read()
            key_bytes = open(key_file, "rb").read()
            credentials = grpc.ssl_server_credentials(
                [(key_bytes, cert_bytes)]
            )
            server.add_secure_port(f"[::]:{bind_port}", credentials)
            _log.info("gRPC server bound to [::]:%d (TLS)", bind_port)
        except Exception as exc:
            _log.error(
                "Failed to load TLS cert/key (%s); falling back to insecure.",
                exc,
            )
            server.add_insecure_port(f"[::]:{bind_port}")
            _log.info("gRPC server bound to [::]:%d (insecure fallback)", bind_port)
    else:
        server.add_insecure_port(f"[::]:{bind_port}")
        _log.info("gRPC server bound to [::]:%d", bind_port)

    # Phase 5: optional Prometheus exposer. Never fatal — if the port is
    # taken or prometheus_client is missing we log and keep the gRPC server
    # running.
    if settings.metrics_enabled:
        try:
            from sql_agent.observability.metrics import start_metrics_server

            start_metrics_server()
        except Exception as exc:
            _log.warning("Metrics server failed to start: %s", exc)

    # Phase 8.7: SIGHUP reloads routing (POSIX). No-op on Windows.
    try:
        from sql_agent.routing.router import install_sighup_reload_handler

        install_sighup_reload_handler()
    except Exception as exc:  # pragma: no cover — defensive
        _log.debug("SIGHUP handler install skipped: %s", exc)

    return server


def serve(port: Optional[int] = None, *, max_workers: int = 10) -> None:
    """Run a blocking gRPC server until Ctrl-C / SIGTERM."""
    server = create_server(port=port, max_workers=max_workers)
    server.start()
    _log.info("gRPC server started; awaiting termination")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        _log.info("gRPC server shutting down (KeyboardInterrupt)")
        server.stop(grace=5)
