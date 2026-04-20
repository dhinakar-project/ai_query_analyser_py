"""Conditional edges for LangGraph query analysis pipeline."""

from graph.state import QueryState


def should_skip_responder(state: QueryState) -> str:
    """Route to escalation_response for Critical+escalate, else responder."""
    if state.get("should_escalate") and state.get("priority") == "Critical":
        return "escalation_response"
    return "responder"