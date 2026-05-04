"""gRPC gateway for the SQL Agent.

Phase 1 goal: expose the existing `sql_agent.agents.orchestrator.run_turn`
over gRPC without modifying any other part of the codebase.

Public surface:
    - SqlAgentServicer  : gRPC servicer wrapping run_turn.
    - serve             : convenience function to run a blocking server.

Generated protobuf modules are imported lazily so that `import sql_agent`
does not require grpcio for users who only need the CLI / Streamlit UI.
"""

from __future__ import annotations

__all__ = ["SqlAgentServicer", "serve"]


def __getattr__(name: str):
    if name in ("SqlAgentServicer", "serve"):
        from .server import SqlAgentServicer, serve

        return {"SqlAgentServicer": SqlAgentServicer, "serve": serve}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
