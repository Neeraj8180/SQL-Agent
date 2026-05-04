"""Streamlit sidebar: session list, new chat, delete."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from sql_agent.services.memory_manager import get_memory_manager
from sql_agent.ui.chat_history import ChatHistoryStore


def render_sidebar(store: ChatHistoryStore) -> Optional[str]:
    """Render the sidebar. Returns the currently selected session_id."""
    with st.sidebar:
        st.markdown("### SQL Agent")
        st.caption("Tool-driven multi-agent system. LLM never writes SQL.")

        if st.button("+ New chat", use_container_width=True, type="primary"):
            new_session = store.new_session()
            st.session_state.session_id = new_session.id
            st.session_state.messages = []
            st.session_state.memory_summary = None
            st.rerun()

        st.divider()
        st.markdown("#### Conversations")

        sessions = store.list_sessions()
        if not sessions:
            st.caption("No chats yet. Ask a question below.")

        for s in sessions:
            is_active = s.id == st.session_state.get("session_id")
            label = f"{'▶ ' if is_active else ''}{s.title}"
            col_a, col_b = st.columns([0.85, 0.15])
            with col_a:
                if st.button(
                    label,
                    key=f"sess_{s.id}",
                    use_container_width=True,
                    type="secondary",
                ):
                    _load(store, s.id)
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{s.id}", help="Delete"):
                    store.delete(s.id)
                    if st.session_state.get("session_id") == s.id:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.session_state.memory_summary = None
                    st.rerun()

        st.divider()
        with st.expander("Memory stats", expanded=False):
            mm = get_memory_manager()
            st.caption(f"Reward rules: {mm.reward_size}")
            st.caption(f"Penalty rules: {mm.penalty_size}")

    return st.session_state.get("session_id")


def _load(store: ChatHistoryStore, session_id: str) -> None:
    session = store.load(session_id)
    if session is None:
        return
    st.session_state.session_id = session.id
    st.session_state.messages = list(session.messages)
    st.session_state.memory_summary = session.memory_summary
