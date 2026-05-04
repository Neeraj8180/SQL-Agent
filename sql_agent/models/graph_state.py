"""LangGraph state container."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ChatMessage(TypedDict, total=False):
    role: str  # "user" | "assistant" | "system"
    content: str
    # Optional assistant turn metadata:
    data_preview: Optional[List[Dict[str, Any]]]
    chart_b64: Optional[str]
    tool_used: Optional[str]
    parameters: Optional[Dict[str, Any]]
    insights: Optional[str]


class MemoryRule(TypedDict, total=False):
    kind: str  # "reward" | "penalty"
    query: str
    reason: str
    params: Optional[Dict[str, Any]]
    tool_used: Optional[str]
    reasoning: Optional[str]
    score: float  # similarity score
    timestamp: str


class AgentState(TypedDict, total=False):
    """State threaded through every LangGraph node.

    `total=False` keeps node handlers simple — each returns only the keys it
    wants to update and LangGraph merges.
    """

    # Inputs
    user_query: str
    session_id: str
    messages: List[ChatMessage]

    # Discovery
    schema: Optional[Dict[str, Any]]

    # Memory
    memory_rules: List[MemoryRule]
    memory_summary: Optional[str]

    # Intent / datetime
    intent: Optional[Dict[str, Any]]
    datetime_resolved: Optional[Dict[str, Any]]

    # Parameters & tool
    parameters: Optional[Dict[str, Any]]
    param_reasoning: Optional[str]
    tool_used: Optional[str]

    # Validation / retry
    validation_errors: List[str]
    retry_count: int

    # Execution artifacts
    data_preview: Optional[List[Dict[str, Any]]]
    data: Optional[List[Dict[str, Any]]]
    data_cleaned: Optional[List[Dict[str, Any]]]

    # Post-processing
    analysis: Optional[Dict[str, Any]]
    visualization: Optional[str]  # base64 PNG
    insights: Optional[str]

    # Outcome
    error: Optional[str]
    success: bool


def empty_state(user_query: str, session_id: str) -> AgentState:
    return AgentState(
        user_query=user_query,
        session_id=session_id,
        messages=[],
        schema=None,
        memory_rules=[],
        memory_summary=None,
        intent=None,
        datetime_resolved=None,
        parameters=None,
        param_reasoning=None,
        tool_used=None,
        validation_errors=[],
        retry_count=0,
        data_preview=None,
        data=None,
        data_cleaned=None,
        analysis=None,
        visualization=None,
        insights=None,
        error=None,
        success=False,
    )
