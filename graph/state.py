"""State definitions for LangGraph query analysis pipeline."""

from typing import TypedDict, Optional, Literal, List


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


class QueryState(TypedDict):
    """State passed through the LangGraph query analysis pipeline.
    
    Attributes:
        query: The original customer query text.
        category: Classified category of the query.
        sentiment: Detected sentiment of the query.
        priority: Assessed priority level.
        confidence_category: Confidence score (0-100) for category classification.
        confidence_sentiment: Confidence score (0-100) for sentiment analysis.
        should_escalate: Whether query should be escalated to human agent.
        escalation_reason: Reason for escalation if applicable.
        suggested_team: Suggested team for escalation.
        language: Detected language of query.
        response: Generated AI response to customer.
        reasoning_trace: List of human-readable reasoning strings.
        retry_count: Number of retries for low-confidence classification.
        processing_time_ms: Total processing time in milliseconds.
    """
    
    query: str
    category: Optional[Category]
    sentiment: Optional[Sentiment]
    priority: Optional[Priority]
    confidence_category: Optional[int]
    confidence_sentiment: Optional[int]
    should_escalate: Optional[bool]
    escalation_reason: Optional[str]
    suggested_team: Optional[str]
    language: Optional[str]
    response: Optional[str]
    reasoning_trace: List[str]
    retry_count: int
    processing_time_ms: int