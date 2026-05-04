"""Streamlit chat interface for the SQL Agent.

Run with:
    streamlit run sql_agent/ui/app.py
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is importable when running via `streamlit run`.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from sql_agent.agents.orchestrator import build_graph, run_turn
from sql_agent.config import configure_logging, settings
from sql_agent.models.graph_state import ChatMessage
from sql_agent.seed_demo import seed as seed_demo_db
from sql_agent.ui.chat_history import ChatHistoryStore
from sql_agent.ui.sidebar import render_sidebar


st.set_page_config(
    page_title="SQL Agent",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _bootstrap():
    configure_logging()
    # Seed demo DB when using the default SQLite URL.
    if settings.database_url.startswith("sqlite:///"):
        try:
            seed_demo_db(force=False)
        except Exception as exc:  # never block UI
            st.warning(f"Demo seeding skipped: {exc}")
    build_graph()  # warm the compiled graph
    return ChatHistoryStore()


def _init_state() -> None:
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory_summary", None)


def _render_assistant_message(msg: Dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        content = msg.get("content") or ""
        if content:
            st.markdown(content)

        rows = msg.get("data_preview")
        if rows:
            with st.expander("Data preview", expanded=True):
                try:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception:
                    st.json(rows)

        chart_b64 = msg.get("chart_b64")
        if chart_b64:
            try:
                st.image(base64.b64decode(chart_b64), use_container_width=True)
            except Exception:
                pass

        insights = msg.get("insights")
        if insights:
            with st.expander("Insights", expanded=False):
                st.markdown(insights)

        tool_used = msg.get("tool_used")
        params = msg.get("parameters")
        if tool_used or params:
            with st.expander("Tool call details", expanded=False):
                st.markdown(f"**Tool used:** `{tool_used or '-'}`")
                st.markdown("**Parameters:**")
                st.code(json.dumps(params or {}, indent=2, default=str), language="json")


def _render_user_message(msg: Dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(msg.get("content") or "")


def _render_transcript(messages: List[Dict[str, Any]]) -> None:
    for m in messages:
        if m.get("role") == "user":
            _render_user_message(m)
        else:
            _render_assistant_message(m)


def _format_assistant_content(final_state: Dict[str, Any]) -> str:
    if final_state.get("error"):
        return f"⚠️ {final_state['error']}"

    data = final_state.get("data_cleaned") or final_state.get("data") or []
    tool = final_state.get("tool_used") or "data_fetch"
    if tool == "count" and data:
        return f"**Count:** {data[0].get('count', 0)}"
    if not data:
        return "_No rows matched._"

    lines = [f"Returned **{len(data)}** row(s) via `{tool}`."]
    reasoning = final_state.get("param_reasoning")
    if reasoning:
        lines.append(f"\n**Plan:** {reasoning}")
    return "\n".join(lines)


def _run_pipeline(
    user_query: str,
    store: ChatHistoryStore,
) -> None:
    session_id = st.session_state.session_id
    if not session_id:
        session = store.new_session()
        session_id = session.id
        st.session_state.session_id = session_id

    prior_messages: List[ChatMessage] = list(st.session_state.messages)
    summary = st.session_state.memory_summary

    # Render the user turn immediately.
    user_msg: ChatMessage = {"role": "user", "content": user_query}
    _render_user_message(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

        try:
            final_state = run_turn(
                user_query,
                session_id=session_id,
                prior_messages=prior_messages,
                memory_summary=summary,
            )
        except Exception as exc:
            placeholder.error(f"Pipeline failed: {exc}")
            st.code(traceback.format_exc())
            return

        placeholder.empty()

        content = _format_assistant_content(final_state)
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": content,
            "data_preview": final_state.get("data_cleaned")
            or final_state.get("data")
            or final_state.get("data_preview"),
            "chart_b64": final_state.get("visualization"),
            "tool_used": final_state.get("tool_used"),
            "parameters": final_state.get("parameters"),
            "insights": final_state.get("insights"),
        }

        _render_assistant_message(dict(assistant_msg))

    # Persist turn.
    st.session_state.messages.append(dict(user_msg))
    st.session_state.messages.append(dict(assistant_msg))
    st.session_state.memory_summary = final_state.get("memory_summary") or summary

    session = store.load(session_id) or store.new_session()
    store.append_turn(
        session,
        user_msg,
        assistant_msg,
        memory_summary=st.session_state.memory_summary,
    )


def main() -> None:
    store = _bootstrap()
    _init_state()

    render_sidebar(store)

    st.title("SQL Agent")
    st.caption(
        "Ask questions in natural language. The LLM plans parameters — "
        "SQL is executed only by SQLAlchemy tools with bound parameters."
    )

    _render_transcript(st.session_state.messages)

    if not settings.openai_api_key:
        st.warning(
            "`OPENAI_API_KEY` is not set. Copy `.env.example` to `.env` and add your key."
        )

    user_query = st.chat_input("Ask about your data…")
    if user_query:
        _run_pipeline(user_query, store)


if __name__ == "__main__":
    main()
