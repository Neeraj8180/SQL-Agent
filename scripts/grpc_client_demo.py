"""Minimal gRPC client for manual smoke-testing the SQL Agent gateway.

Usage:
    python scripts/grpc_client_demo.py "How many orders are there?"
    python scripts/grpc_client_demo.py --rpc generate "Revenue by country"
    python scripts/grpc_client_demo.py --rpc health
    python scripts/grpc_client_demo.py --target localhost:60000 "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQL Agent gRPC client demo")
    p.add_argument("query", nargs="*", help="Natural-language question.")
    p.add_argument(
        "--rpc",
        choices=("execute", "generate", "health"),
        default="execute",
        help="Which RPC to invoke.",
    )
    p.add_argument("--target", default="localhost:50051", help="host:port")
    p.add_argument("--session-id", default="", help="Session id (server generates if empty).")
    return p.parse_args()


def _print_generate(resp) -> None:
    print("--- GenerateSQL ---")
    print(f"session_id      : {resp.session_id}")
    print(f"tool_used       : {resp.tool_used}")
    print(f"param_reasoning : {resp.param_reasoning}")
    print(f"error           : {resp.error or '(none)'}")
    print(f"latency_ms      : {resp.latency_ms:.1f}")
    try:
        params = json.loads(resp.parameters_json or "{}")
        print("parameters      :")
        print(json.dumps(params, indent=2, default=str))
    except Exception:
        print(f"parameters_json : {resp.parameters_json}")


def _print_execute(resp) -> None:
    print("--- ExecuteSQL ---")
    print(f"session_id    : {resp.session_id}")
    print(f"tool_used     : {resp.tool_used}")
    print(f"success       : {resp.success}")
    print(f"row_count     : {resp.row_count}")
    print(f"error         : {resp.error or '(none)'}")
    print(f"latency_ms    : {resp.latency_ms:.1f}")
    if resp.insights:
        print(f"insights      :\n{resp.insights}")
    try:
        rows = json.loads(resp.rows_json or "[]")
        preview = rows[:5]
        print(f"rows (first {len(preview)}):")
        for r in preview:
            print(f"  {r}")
    except Exception:
        print(f"rows_json     : {resp.rows_json[:200]}...")


def main() -> int:
    args = _parse_args()

    import grpc

    from sql_agent.grpc_server import sql_agent_pb2 as pb2
    from sql_agent.grpc_server import sql_agent_pb2_grpc as pb2_grpc

    channel = grpc.insecure_channel(args.target)
    stub = pb2_grpc.SqlAgentStub(channel)

    if args.rpc == "health":
        resp = stub.HealthCheck(pb2.HealthCheckRequest())
        print(f"status={resp.status} version={resp.version}")
        return 0

    query = " ".join(args.query).strip()
    if not query:
        print("error: query must not be empty", file=sys.stderr)
        return 2

    if args.rpc == "generate":
        resp = stub.GenerateSQL(
            pb2.GenerateSQLRequest(query=query, session_id=args.session_id)
        )
        _print_generate(resp)
        return 1 if resp.error else 0

    resp = stub.ExecuteSQL(
        pb2.ExecuteSQLRequest(query=query, session_id=args.session_id)
    )
    _print_execute(resp)
    return 0 if resp.success else 1


if __name__ == "__main__":
    sys.exit(main())
