"""LangGraph builder for query analysis pipeline."""

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import QueryState
from graph.nodes import combined_analysis_node, responder_node

logger = logging.getLogger(__name__)


def should_skip_responder(state: QueryState) -> str:
    """
    If escalation is needed AND priority is Critical, use a fast escalation
    response path. Otherwise go to the normal responder.
    """
    if state.get("should_escalate") and state.get("priority") == "Critical":
        return "escalation_response"
    return "responder"


def escalation_response_node(state: QueryState) -> dict:
    """
    Fast path for critical escalations — no LLM call needed,
    returns a templated escalation message immediately.
    Saves one LLM call for the most urgent cases.
    """
    team = state.get("suggested_team") or "our specialist team"
    reason = state.get("escalation_reason") or "the urgency of your situation"

    response = (
        f"I understand this is urgent and I sincerely apologise for the difficulty "
        f"you're experiencing. Due to {reason}, I'm immediately connecting you with "
        f"{team} who can resolve this right away. Please hold — someone will be with "
        f"you shortly."
    )

    existing_trace = state.get("reasoning_trace", [])
    return {
        "response": response,
        "rag_sources": [],
        "reasoning_trace": existing_trace + [
            f"[escalation_response] fast-path used for Critical+escalate case"
        ]
    }


def build_graph() -> StateGraph:
    """
    Build the LangGraph pipeline.

    Flow:
      START
        └─► combined_analysis_node   (1 LLM call: lang + classify + sentiment + priority + escalation)
              ├─► escalation_response_node  (0 LLM calls: template, Critical+escalate only)
              └─► responder_node            (1 LLM call: response generation)
                    └─► END

    Total LLM calls per query: 2 (down from 6-7)
    """
    checkpointer = MemorySaver()
    graph = StateGraph(QueryState)

    graph.add_node("combined_analysis", combined_analysis_node)
    graph.add_node("responder", responder_node)
    graph.add_node("escalation_response", escalation_response_node)

    graph.add_edge(START, "combined_analysis")

    graph.add_conditional_edges(
        "combined_analysis",
        should_skip_responder,
        {
            "responder": "responder",
            "escalation_response": "escalation_response",
        }
    )

    graph.add_edge("responder", END)
    graph.add_edge("escalation_response", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph query analysis pipeline built successfully")
    return compiled


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph