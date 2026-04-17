"""Combined analyzer module for single-call query analysis.

This module provides a unified analysis function that extracts all query metadata
(category, sentiment, priority, escalation, language) in a single LLM call,
reducing API usage and improving response time.
"""

import json
import logging
import time
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_gemini

logger = logging.getLogger(__name__)

_last_call_time = 0
MIN_INTERVAL = 12


def _rate_limit_guard() -> None:
    """Enforce rate limiting to stay under free tier limits (5 req/min).
    
    Sleeps if necessary to ensure at least MIN_INTERVAL seconds between calls.
    """
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        sleep_time = MIN_INTERVAL - elapsed
        logger.info(f"Rate limit: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)
    _last_call_time = time.time()


@tool
def combined_query_analysis(query: str) -> str:
    """Analyze a customer query comprehensively in a single LLM call.
    
    This tool extracts all query metadata (category, sentiment, priority, 
    escalation, language) in one call for efficiency.
    
    Args:
        query: The customer's query text to analyze.
        
    Returns:
        A JSON string containing all analysis results.
    """
    _rate_limit_guard()
    
    logger.debug(f"Running combined analysis for: {query[:50]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert customer query analyzer. Analyze the query and return ALL metadata in ONE JSON response.

Return EXACTLY this JSON structure (no markdown, no explanation, no text outside the JSON):
{
    "category": "Billing|Technical Support|Returns & Refunds|Shipping & Delivery|Account Management|General Inquiry",
    "category_confidence": 0-100,
    "sentiment": "Positive|Neutral|Negative|Urgent|Frustrated",
    "sentiment_confidence": 0-100,
    "priority": "Critical|High|Medium|Low",
    "should_escalate": true|false,
    "escalation_reason": "string or null",
    "suggested_team": "Billing Support Team|Technical Support Team|Returns & Refunds Team|Shipping & Logistics Team|Account Security Team|Customer Service Team|null",
    "language": "detected language name"
}

CATEGORY RULES:
- Billing: Payment issues, charges, invoices, pricing, billing disputes
- Technical Support: Software/hardware problems, login errors, bugs, crashes
- Returns & Refunds: Product returns, money back, exchanges, damaged items
- Shipping & Delivery: Order tracking, delivery delays, lost packages
- Account Management: Login, password, profile, subscription changes
- General Inquiry: Questions about products, services, company info

SENTIMENT RULES:
- Positive: Satisfied, happy, grateful, complimenting
- Neutral: Normal question, no strong emotion
- Negative: Unhappy, disappointed
- Urgent: Needs immediate attention, time-sensitive
- Frustrated: Annoyed, repeated issues, losing patience

PRIORITY RULES:
- Critical: Urgent/Frustrated + Billing, or emergency keywords
- High: Frustrated + any category, Urgent + Technical Support, Negative + Billing
- Medium: Negative in most categories, standard technical questions
- Low: Positive/Neutral sentiment, simple inquiries

ESCALATION RULES:
- true if: Critical priority, or High+Frustrated/Urgent, or legal/safety/security keywords
- false if: Low/Medium priority with positive/neutral sentiment

Respond with ONLY the JSON object."""),
        ("human", "Analyze this customer query:\n\n{query}")
    ])
    
    llm = get_gemini(temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": query})
        result = result.strip()
        
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(line for line in lines if not line.strip().startswith("```"))
            result = result.strip()
        
        parsed = json.loads(result)
        
        return json.dumps({
            "category": parsed.get("category", "General Inquiry"),
            "category_confidence": parsed.get("category_confidence", 50),
            "sentiment": parsed.get("sentiment", "Neutral"),
            "sentiment_confidence": parsed.get("sentiment_confidence", 50),
            "priority": parsed.get("priority", "Medium"),
            "should_escalate": parsed.get("should_escalate", False),
            "escalation_reason": parsed.get("escalation_reason"),
            "suggested_team": parsed.get("suggested_team"),
            "language": parsed.get("language", "English")
        })
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}, result: {result}")
        return json.dumps(_get_safe_defaults())
    except Exception as e:
        logger.error(f"Combined analysis error: {e}")
        return json.dumps(_get_safe_defaults())


def _get_safe_defaults() -> Dict[str, Any]:
    """Return safe default values for all fields on error."""
    return {
        "category": "General Inquiry",
        "category_confidence": 50,
        "sentiment": "Neutral",
        "sentiment_confidence": 50,
        "priority": "Medium",
        "should_escalate": False,
        "escalation_reason": None,
        "suggested_team": None,
        "language": "English"
    }


def run_combined_analysis(query: str) -> Dict[str, Any]:
    """Run combined analysis on a customer query.
    
    Makes a single LLM call to extract all metadata:
    category, sentiment, priority, escalation, language.
    
    Args:
        query: The customer query to analyze.
        
    Returns:
        Dictionary containing all analysis results.
    """
    logger.info(f"Running combined analysis for query: {query[:50]}...")
    
    try:
        result_str = combined_query_analysis.invoke({"query": query})
        result = json.loads(result_str)
        
        logger.debug(f"Combined analysis complete: category={result['category']}, "
                    f"sentiment={result['sentiment']}, priority={result['priority']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Combined analysis failed: {e}")
        return _get_safe_defaults()
