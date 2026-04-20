"""Node functions for LangGraph query analysis pipeline."""

import logging
import time
from typing import Any

from utils.llm import get_classifier_llm, get_responder_llm, _MODEL_NAME
from utils.rate_limiter import throttle

logger = logging.getLogger(__name__)

COMBINED_ANALYSIS_SYSTEM_PROMPT = """You are an expert customer support analyst.
Given a customer query, you must analyse it and return a structured JSON response.

Think step by step before committing to each field:
1. What language is this written in?
2. What is the primary topic? Choose the single best category.
3. What is the emotional tone? Look for urgency markers, caps, exclamation marks,
   repeated complaints, and explicit frustration words.
4. How urgent is this? Consider sentiment + business impact.
5. Does this need a human agent? Escalate if: Critical priority, Frustrated/Urgent
   sentiment with unresolved repeated issues, legal threats, safety concerns,
   or explicit escalation request.

Category rules:
- Billing: payment, charge, invoice, bill, refund related to money
- Technical Support: errors, bugs, crashes, not working, broken features
- Returns & Refunds: return, exchange, damaged item, wrong item
- Shipping & Delivery: tracking, delivery, package, shipping
- Account Management: login, password, profile, subscription, account access
- General Inquiry: everything else, product questions, company info

Sentiment rules:
- Urgent: words like immediately, asap, emergency, critical, deadline
- Frustrated: repeated issues, "again", "still broken", "third time", "unacceptable"
- Negative: unhappy but not yet at frustrated/urgent level
- Positive: grateful, happy, compliment
- Neutral: plain factual question, no emotional charge

Few-shot examples:
Query: "I've been charged twice this month and nobody is helping me. This is the THIRD time!"
→ category: Billing, sentiment: Frustrated, priority: High, should_escalate: true

Query: "Hi, what are your business hours?"
→ category: General Inquiry, sentiment: Neutral, priority: Low, should_escalate: false

Query: "My entire system is down and I'm losing money every minute. URGENT"
→ category: Technical Support, sentiment: Urgent, priority: Critical, should_escalate: true

Query: "Gracias, el producto llegó perfecto!"
→ language: Spanish, language_code: es, category: General Inquiry, sentiment: Positive,
   priority: Low, should_escalate: false
"""


RESPONDER_SYSTEM_PROMPT_TEMPLATE = """You are an empathetic customer support agent.

Query category: {category}
Customer sentiment: {sentiment}
Priority level: {priority}
Respond in: {language}

Tone guidelines by sentiment:
- Frustrated/Urgent: lead with empathy and apology, be direct about resolution
- Negative: acknowledge the issue, show understanding, offer clear next steps
- Neutral: professional and helpful
- Positive: warm and appreciative

Keep response concise (3-5 sentences). Do not mention internal categories or scores.
Do not start with "I". End with a clear next step or offer of further help.
"""


def combined_analysis_node(state: dict) -> dict:
    """
    Single LLM call replacing: language_node, classify_node, sentiment_node,
    priority_node, escalation_node. Reduces calls from 5 to 1 for analysis phase.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from models.analysis_result import CombinedAnalysis

    logger.info(f"combined_analysis_node: analysing query length={len(state['query'])}")
    throttle()

    start = time.time()
    try:
        llm = get_classifier_llm()
        structured_llm = llm.with_structured_output(CombinedAnalysis)

        result: CombinedAnalysis = structured_llm.invoke([
            SystemMessage(content=COMBINED_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyse this customer query:\n\n{state['query']}")
        ])

        latency_ms = int((time.time() - start) * 1000)

        trace_entry = (
            f"[combined_analysis] category={result.category} "
            f"({result.category_confidence}% conf) | "
            f"sentiment={result.sentiment} ({result.sentiment_confidence}% conf) | "
            f"priority={result.priority} | escalate={result.should_escalate} | "
            f"lang={result.language} | latency={latency_ms}ms"
        )

        logger.info(trace_entry)

        existing_trace = state.get("reasoning_trace", [])

        return {
            "language": result.language,
            "language_code": result.language_code,
            "category": result.category,
            "category_confidence": result.category_confidence,
            "sentiment": result.sentiment,
            "sentiment_confidence": result.sentiment_confidence,
            "priority": result.priority,
            "should_escalate": result.should_escalate,
            "escalation_reason": result.escalation_reason,
            "suggested_team": result.suggested_team,
            "reasoning_trace": existing_trace + [trace_entry],
        }

    except Exception as e:
        logger.error(f"combined_analysis_node failed: {e}")
        existing_trace = state.get("reasoning_trace", [])
        return {
            "language": "English",
            "language_code": "en",
            "category": "General Inquiry",
            "category_confidence": 40,
            "sentiment": "Neutral",
            "sentiment_confidence": 40,
            "priority": "Medium",
            "should_escalate": False,
            "escalation_reason": None,
            "suggested_team": None,
            "reasoning_trace": existing_trace + [f"[combined_analysis] FAILED: {str(e)}"],
        }


def responder_node(state: dict) -> dict:
    """Generate customer-facing response using RAG."""
    from langchain_core.messages import SystemMessage, HumanMessage
    import asyncio

    throttle()
    start = time.time()

    system_prompt = RESPONDER_SYSTEM_PROMPT_TEMPLATE.format(
        category=state.get("category", "General Inquiry"),
        sentiment=state.get("sentiment", "Neutral"),
        priority=state.get("priority", "Medium"),
        language=state.get("preferred_language") or state.get("language", "English"),
    )

    try:
        llm = get_responder_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["query"])
        ]
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        latency_ms = int((time.time() - start) * 1000)
        existing_trace = state.get("reasoning_trace", [])

        return {
            "response": response_text,
            "reasoning_trace": existing_trace + [
                f"[responder] generated response in {latency_ms}ms"
            ]
        }
    except Exception as e:
        logger.error(f"responder_node failed: {e}")
        existing_trace = state.get("reasoning_trace", [])
        return {
            "response": "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment or contact our support team directly.",
            "reasoning_trace": existing_trace + [f"[responder] FAILED: {str(e)}"]
        }