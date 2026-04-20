"""Conditional edges for LangGraph query analysis pipeline."""

from typing import Literal

from graph.state import QueryState


def should_retry_classify(state: QueryState) -> Literal["retry_classify", "sentiment_node"]:
    """Decide whether to retry classification based on confidence.
    
    If confidence_category < 60 and retry_count < 2, retry classification.
    Otherwise, proceed to sentiment analysis.
    
    Args:
        state: Current query state.
        
    Returns:
        Next node to visit.
    """
    confidence = state.get("confidence_category", 100)
    retry_count = state.get("retry_count", 0)
    
    if confidence < 60 and retry_count < 2:
        return "retry_classify"
    return "sentiment_node"


def should_escalate(state: QueryState) -> Literal["escalation_responder", "responder_node"]:
    """Decide whether to skip responder and go to escalation response.
    
    If should_escalate is True, go to escalation_responder_node.
    Otherwise, go to normal responder_node.
    
    Args:
        state: Current query state.
        
    Returns:
        Next node to visit.
    """
    should_escalate = state.get("should_escalate", False)
    
    if should_escalate:
        return "escalation_responder"
    return "responder_node"


def should_retry_classify_direct(state: QueryState) -> Literal["classify_node", "sentiment_node"]:
    """Decide whether to retry classification directly.
    
    If confidence_category < 60 and retry_count < 2, retry classification.
    Otherwise, proceed to sentiment analysis.
    
    Args:
        state: Current query state.
        
    Returns:
        Next node to visit.
    """
    confidence = state.get("confidence_category", 100)
    retry_count = state.get("retry_count", 0)
    
    if confidence < 60 and retry_count < 2:
        return "classify_node"
    return "sentiment_node"