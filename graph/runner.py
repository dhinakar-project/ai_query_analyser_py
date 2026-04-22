"""Async runner for LangGraph query analysis."""

import logging
import time

from graph.builder import get_graph
from graph.state import QueryState

logger = logging.getLogger(__name__)


async def run_graph(query: str, thread_id: str, preferred_language: str = "English") -> dict:
    """
    Run the compiled LangGraph pipeline for a query.
    Returns the final QueryState as a plain dict.
    """
    graph = get_graph()

    config = {"configurable": {"thread_id": thread_id}}

    initial_state: QueryState = {
        "query": query,
        "category": None,
        "sentiment": None,
        "priority": None,
        "category_confidence": 0,
        "sentiment_confidence": 0,
        "should_escalate": False,
        "escalation_reason": None,
        "suggested_team": None,
        "language": None,
        "language_code": None,
        "response": None,
        "reasoning_trace": [],
        "retry_count": 0,
        "processing_time_ms": 0,
        "preferred_language": preferred_language,
    }

    start = time.time()

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        elapsed_ms = int((time.time() - start) * 1000)
        final_state["processing_time_ms"] = elapsed_ms
        logger.info(
            f"run_graph completed: thread={thread_id} "
            f"category={final_state.get('category')} "
            f"latency={elapsed_ms}ms"
        )
        
        try:
            from storage.db import get_database
            from observability.costs import get_session_cost
            from observability.logger import log_graph_run
            
            db = await get_database()
            await db.record_query(
                query=query,
                category=final_state.get("category", "General Inquiry"),
                sentiment=final_state.get("sentiment", "Neutral"),
                priority=final_state.get("priority", "Medium"),
                escalated=final_state.get("should_escalate", False),
                language=final_state.get("language", "English"),
                latency_ms=elapsed_ms,
                cost_usd=get_session_cost(),
                category_confidence=final_state.get("category_confidence", 0),
                sentiment_confidence=final_state.get("sentiment_confidence", 0),
            )
            logger.info(f"Query recorded to database: {thread_id}")
            
            log_graph_run(
                trace_id=thread_id,
                total_nodes=len(final_state.get("reasoning_trace", [])),
                total_latency_ms=elapsed_ms,
                final_category=final_state.get("category", "General Inquiry"),
                final_sentiment=final_state.get("sentiment", "Neutral"),
                final_priority=final_state.get("priority", "Medium"),
                escalated=final_state.get("should_escalate", False),
                confidence_category=final_state.get("category_confidence", 0),
                confidence_sentiment=final_state.get("sentiment_confidence", 0)
            )
        except Exception as _err:
            logger.warning(f"DB/Logger record failed (non-fatal): {_err}")
            
        return dict(final_state)
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        logger.error(f"run_graph failed after {elapsed_ms}ms: {e}")
        return {
            "query": query,
            "category": "General Inquiry",
            "sentiment": "Neutral",
            "priority": "Medium",
            "category_confidence": 0,
            "sentiment_confidence": 0,
            "should_escalate": False,
            "escalation_reason": None,
            "suggested_team": None,
            "language": "English",
            "language_code": "en",
            "response": "I'm sorry, something went wrong. Please try again.",
            "reasoning_trace": [f"GRAPH FAILED: {str(e)}"],
            "retry_count": 0,
            "processing_time_ms": elapsed_ms,
        }