"""State definitions for LangGraph query analysis pipeline."""

from typing import TypedDict, Optional, Literal


Category = Literal[
    "Billing",
    "Technical Support",
    "Returns & Refunds",
    "Shipping & Delivery",
    "Account Management",
    "General Inquiry"
]

Sentiment = Literal[
    "Positive",
    "Neutral",
    "Negative",
    "Urgent",
    "Frustrated"
]

Priority = Literal["Critical", "High", "Medium", "Low"]


class QueryState(TypedDict, total=False):
    """State passed through the LangGraph query analysis pipeline."""

    query: str
    category: Optional[str]
    sentiment: Optional[str]
    priority: Optional[str]
    category_confidence: int
    sentiment_confidence: int
    should_escalate: bool
    escalation_reason: Optional[str]
    suggested_team: Optional[str]
    language: Optional[str]
    language_code: Optional[str]
    preferred_language: str
    response: Optional[str]
    reasoning_trace: list[str]
    retry_count: int
    processing_time_ms: int