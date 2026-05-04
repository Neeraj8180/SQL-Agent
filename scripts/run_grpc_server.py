"""Run the SQL Agent gRPC gateway.

Usage:
    python scripts/run_grpc_server.py                 # uses GRPC_PORT (default 50051)
    python scripts/run_grpc_server.py --port 60000
    python scripts/run_grpc_server.py --workers 16

This script does NOT seed the demo DB — run `python -m sql_agent.seed_demo`
first if you're using the default SQLite backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQL Agent gRPC gateway")
    p.add_argument("--port", type=int, default=None, help="Override GRPC_PORT setting.")
    p.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Thread-pool workers for concurrent RPCs.",
    )
    p.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Install a deterministic fake LLM (for phase 1 demos without an "
            "OpenAI key). Do NOT use in production."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mock_llm:
        # Phase 2.2: this is now equivalent to setting LLM_PROVIDER=mock and
        # EMBEDDING_PROVIDER=mock in the environment. Kept as a convenience
        # for demos.
        import os

        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        print("[run_grpc_server] LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock — demo mode.")

    # Import after any env-var tweaks so pydantic-settings picks them up.
    from sql_agent.config import get_logger, settings

    log = get_logger("run_grpc_server")

    # Phase 6: optional demo-DB seeding. Production should set
    # SEED_DEMO_ON_BOOT=false (default) and seed explicitly; the docker
    # compose profile turns it on for zero-config demos.
    if settings.seed_demo_on_boot and settings.database_url.startswith("sqlite:///"):
        try:
            from sql_agent.seed_demo import seed

            seed(force=False)
            log.info("Demo DB seeded (SEED_DEMO_ON_BOOT=true)")
        except Exception as exc:
            log.warning("Demo-seed skipped: %s", exc)

    from sql_agent.grpc_server.server import serve

    serve(port=args.port, max_workers=args.workers)


if __name__ == "__main__":
    main()
