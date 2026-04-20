"""LangGraph builder for query analysis pipeline."""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import QueryState
from graph.nodes import (
    language_node,
    classify_node,
    sentiment_node,
    priority_node,
    escalation_node,
    responder_node,
    escalation_responder_node,
    retry_classify_node,
)
from graph.edges import (
    should_retry_classify,
    should_escalate,
)

logger = logging.getLogger(__name__)


def create_initial_state(query: str) -> QueryState:
    """Create initial state for the graph.
    
    Args:
        query: The customer query to analyze.
        
    Returns:
        Initial QueryState dictionary.
    """
    return {
        "query": query,
        "category": None,
        "sentiment": None,
        "priority": None,
        "confidence_category": None,
        "confidence_sentiment": None,
        "should_escalate": None,
        "escalation_reason": None,
        "suggested_team": None,
        "language": None,
        "response": None,
        "reasoning_trace": [],
        "retry_count": 0,
        "processing_time_ms": 0
    }


def build_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build the LangGraph query analysis pipeline.
    
    Args:
        checkpointer: Optional MemorySaver for cross-session persistence.
        
    Returns:
        Compiled StateGraph for query analysis.
    """
    workflow = StateGraph(QueryState)
    
    workflow.add_node("language_node", language_node)
    workflow.add_node("classify_node", classify_node)
    workflow.add_node("retry_classify", retry_classify_node)
    workflow.add_node("sentiment_node", sentiment_node)
    workflow.add_node("priority_node", priority_node)
    workflow.add_node("escalation_node", escalation_node)
    workflow.add_node("responder_node", responder_node)
    workflow.add_node("escalation_responder", escalation_responder_node)
    
    workflow.set_entry_point("language_node")
    
    workflow.add_edge("language_node", "classify_node")
    workflow.add_conditional_edges(
        "classify_node",
        should_retry_classify,
        {
            "retry_classify": "retry_classify",
            "sentiment_node": "sentiment_node"
        }
    )
    workflow.add_conditional_edges(
        "retry_classify",
        lambda state: "sentiment_node" if state.get("retry_count", 0) >= 2 else "classify_node",
        {
            "classify_node": "classify_node",
            "sentiment_node": "sentiment_node"
        }
    )
    workflow.add_edge("sentiment_node", "priority_node")
    workflow.add_edge("priority_node", "escalation_node")
    workflow.add_conditional_edges(
        "escalation_node",
        should_escalate,
        {
            "escalation_responder": "escalation_responder",
            "responder_node": "responder_node"
        }
    )
    workflow.add_edge("escalation_responder", END)
    workflow.add_edge("responder_node", END)
    
    if checkpointer:
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()
    
    logger.info("LangGraph query analysis pipeline built successfully")
    return compiled


def get_default_checkpointer() -> MemorySaver:
    """Get a default MemorySaver checkpointer for the graph.
    
    Returns:
        MemorySaver instance for state persistence.
    """
    return MemorySaver()


default_graph = build_graph(get_default_checkpointer())