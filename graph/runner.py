"""Async runner for LangGraph query analysis."""

import logging
import time
import uuid
from typing import Dict, Any, Optional

from graph.builder import build_graph, get_default_checkpointer
from graph.state import QueryState
from observability.logger import get_query_hash, get_trace_id, log_graph_run

logger = logging.getLogger(__name__)


async def run_graph(
    query: str,
    thread_id: Optional[str] = None,
    preferred_language: str = "English"
) -> QueryState:
    """Run the query analysis graph.
    
    Args:
        query: The customer query to analyze.
        thread_id: Optional thread ID for conversation persistence.
        preferred_language: Preferred response language.
        
    Returns:
        Final QueryState with all analysis results.
    """
    start_time = time.time()
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    checkpointer = get_default_checkpointer()
    graph = build_graph(checkpointer)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state: QueryState = {
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
    
    try:
        final_state = await graph.ainvoke(
            initial_state,
            config=config
        )
        
        end_time = time.time()
        processing_time = int((end_time - start_time) * 1000)
        
        final_state["processing_time_ms"] = processing_time
        
        trace_id = get_trace_id()
        query_hash = get_query_hash(query)
        
        log_graph_run(
            trace_id=trace_id,
            total_nodes=len(final_state.get("reasoning_trace", [])),
            total_latency_ms=processing_time,
            final_category=final_state.get("category", "Unknown"),
            final_sentiment=final_state.get("sentiment", "Unknown"),
            final_priority=final_state.get("priority", "Medium"),
            escalated=final_state.get("should_escalate", False),
            confidence_category=final_state.get("confidence_category", 0),
            confidence_sentiment=final_state.get("confidence_sentiment", 0),
            query_hash=query_hash
        )
        
        return final_state
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise


def run_graph_sync(
    query: str,
    thread_id: Optional[str] = None,
    preferred_language: str = "English"
) -> QueryState:
    """Synchronous wrapper for run_graph.
    
    Args:
        query: The customer query to analyze.
        thread_id: Optional thread ID for conversation persistence.
        preferred_language: Preferred response language.
        
    Returns:
        Final QueryState with all analysis results.
    """
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_graph(query, thread_id, preferred_language))
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(run_graph(query, thread_id, preferred_language))