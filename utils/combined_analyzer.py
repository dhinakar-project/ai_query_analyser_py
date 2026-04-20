"""Combined analyzer module - now uses LangGraph pipeline."""

import asyncio
import time
import uuid


def run_combined_analysis(query: str, thread_id: str = None) -> dict:
    """
    Drop-in replacement for the old run_combined_analysis.
    Now runs the full LangGraph pipeline (combined_analysis_node + responder_node).
    Returns the same dict shape as before so app.py needs zero changes.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    start = time.time()

    try:
        from graph.runner import run_graph
    except ImportError:
        raise RuntimeError("graph.runner import failed")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            final_state = loop.run_until_complete(
                run_graph(query=query, thread_id=thread_id)
            )
        finally:
            loop.close()
    except RuntimeError:
        final_state = asyncio.run(run_graph(query=query, thread_id=thread_id))

    processing_time_ms = int((time.time() - start) * 1000)

    return {
        "category": final_state.get("category", "General Inquiry"),
        "sentiment": final_state.get("sentiment", "Neutral"),
        "priority": final_state.get("priority", "Medium"),
        "category_confidence": final_state.get("category_confidence", 50),
        "sentiment_confidence": final_state.get("sentiment_confidence", 50),
        "should_escalate": final_state.get("should_escalate", False),
        "escalation_reason": final_state.get("escalation_reason"),
        "suggested_team": final_state.get("suggested_team"),
        "language": final_state.get("language", "English"),
        "language_code": final_state.get("language_code", "en"),
        "response": final_state.get("response", ""),
        "reasoning_trace": final_state.get("reasoning_trace", []),
        "processing_time_ms": processing_time_ms,
    }
