"""Transcript processor - bridges voice transcripts to LangGraph pipeline."""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_customer_utterances(transcript: str) -> str:
    """Extract only customer utterances from transcript.

    Args:
        transcript: Full transcript with "Customer:" and "Agent:" prefixes

    Returns:
        Cleaned transcript with only customer messages
    """
    lines = transcript.split("\n")
    customer_lines = []

    for line in lines:
        if line.startswith("Customer:"):
            content = line[len("Customer:"):].strip()
            if content:
                customer_lines.append(content)
        elif line.startswith("Customer "):
            match = re.match(r"Customer\s+(.+)", line)
            if match:
                customer_lines.append(match.group(1).strip())

    return " ".join(customer_lines) if customer_lines else transcript


async def process_voice_transcript(
    transcript: str,
    call_id: str,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a voice transcript through the LangGraph pipeline.

    Args:
        transcript: Raw voice transcript text
        call_id: Vapi call ID
        thread_id: Optional thread ID for LangGraph state

    Returns:
        Standard analysis result dict with all fields
    """
    thread_id = thread_id or f"voice_{call_id}"

    cleaned_query = extract_customer_utterances(transcript)

    if not cleaned_query.strip():
        logger.warning(f"Empty transcript for call {call_id}")
        return {
            "query": transcript,
            "category": "General Inquiry",
            "sentiment": "Neutral",
            "priority": "Medium",
            "should_escalate": False,
            "response": "No voice input detected.",
            "call_id": call_id,
            "transcript": transcript,
        }

    logger.info(f"Processing voice transcript for call {call_id}: {cleaned_query[:100]}...")

    try:
        from graph.runner import run_graph
        result = await run_graph(query=cleaned_query, thread_id=thread_id)

        result["call_id"] = call_id
        result["transcript"] = transcript

        try:
            from storage.db import get_database
            db = await get_database()
            await db.record_query(
                query=cleaned_query,
                category=result.get("category", "General Inquiry"),
                sentiment=result.get("sentiment", "Neutral"),
                priority=result.get("priority", "Medium"),
                escalated=result.get("should_escalate", False),
                language=result.get("language", "English"),
                latency_ms=result.get("processing_time_ms", 0),
                cost_usd=0.0,
                category_confidence=result.get("category_confidence", 0),
                sentiment_confidence=result.get("sentiment_confidence", 0),
            )
            logger.debug(f"Recorded voice query to database")
        except Exception as e:
            logger.warning(f"Failed to record to database: {e}")

        return result

    except Exception as e:
        logger.error(f"Failed to process voice transcript: {e}")
        return {
            "query": cleaned_query,
            "category": "General Inquiry",
            "sentiment": "Neutral",
            "priority": "Medium",
            "should_escalate": False,
            "response": "Analysis failed. Please try again.",
            "reasoning_trace": [f"PROCESSING ERROR: {str(e)}"],
            "call_id": call_id,
            "transcript": transcript,
            "error": str(e),
        }