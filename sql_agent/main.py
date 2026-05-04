"""CLI entrypoint — useful for smoke-testing the pipeline without Streamlit.

Usage:
    python -m sql_agent.main "Show monthly revenue for last 6 months by country"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any, Dict

from sql_agent.agents.orchestrator import run_turn
from sql_agent.config import configure_logging, get_logger, settings
from sql_agent.seed_demo import seed as seed_demo_db


_log = get_logger("main")


def _truncate(rows, n: int = 5):
    if not rows:
        return rows
    return rows[:n]


def _render(final_state: Dict[str, Any]) -> None:
    if final_state.get("error"):
        print(f"[error] {final_state['error']}")
        return

    print(f"\n=== Tool used: {final_state.get('tool_used')} ===")
    print("Parameters:")
    print(json.dumps(final_state.get("parameters"), indent=2, default=str))

    if final_state.get("param_reasoning"):
        print(f"\nPlan: {final_state['param_reasoning']}")

    data = final_state.get("data_cleaned") or final_state.get("data") or []
    print(f"\nRows returned: {len(data)}")
    for row in _truncate(data):
        print(f"  {row}")

    insights = final_state.get("insights")
    if insights:
        print(f"\nInsights:\n{insights}")

    if final_state.get("visualization"):
        print("\n[chart rendered as base64 PNG — omitted in CLI]")


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL Agent CLI")
    parser.add_argument("query", nargs="+", help="Natural language question.")
    parser.add_argument(
        "--skip-seed", action="store_true", help="Do not seed the demo SQLite DB."
    )
    args = parser.parse_args()

    configure_logging()

    if not args.skip_seed and settings.database_url.startswith("sqlite:///"):
        try:
            seed_demo_db(force=False)
        except Exception as exc:
            _log.warning("Seeding skipped: %s", exc)

    query = " ".join(args.query)
    session_id = str(uuid.uuid4())

    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY is not set. Create .env with your key (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(2)

    final = run_turn(query, session_id=session_id, prior_messages=[])
    _render(final)


if __name__ == "__main__":
    main()
