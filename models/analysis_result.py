"""Structured output model for query analysis results."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Pydantic model for structured query analysis results.
    
    This model defines the complete output structure for analyzing
    a customer query, including classification, sentiment, priority,
    confidence scores, and escalation decisions.
    
    Attributes:
        category: The classified category of the customer query.
        sentiment: The analyzed emotional sentiment of the query.
        priority: The assessed priority level based on content and sentiment.
        confidence_category: Confidence score (0-100) for the category classification.
        confidence_sentiment: Confidence score (0-100) for the sentiment analysis.
        should_escalate: Whether this query should be escalated to a human agent.
        escalation_reason: Reason for escalation, if applicable.
        suggested_team: Suggested team for escalation (e.g., "Billing Team", "Technical Support").
        language: Detected language of the query (ISO code or name).
        response: The generated AI response to the customer query.
        processing_time_ms: Time taken to process the query in milliseconds.
    """
    
    category: Literal[
        "Billing",
        "Technical Support",
        "Returns & Refunds",
        "Shipping & Delivery",
        "Account Management",
        "General Inquiry"
    ] = Field(
        description="The classified category of the customer query"
    )
    
    sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative",
        "Urgent",
        "Frustrated"
    ] = Field(
        description="The analyzed emotional sentiment of the customer query"
    )
    
    priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="The assessed priority level based on content and sentiment"
    )
    
    confidence_category: int = Field(
        ge=0,
        le=100,
        description="Confidence score (0-100) for the category classification"
    )
    
    confidence_sentiment: int = Field(
        ge=0,
        le=100,
        description="Confidence score (0-100) for the sentiment analysis"
    )
    
    should_escalate: bool = Field(
        description="Whether this query should be escalated to a human agent"
    )
    
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation, if applicable"
    )
    
    suggested_team: Optional[str] = Field(
        default=None,
        description="Suggested team for escalation"
    )
    
    language: str = Field(
        description="Detected language of the query"
    )
    
    response: str = Field(
        description="The generated AI response to the customer query"
    )
    
    processing_time_ms: int = Field(
        ge=0,
        description="Time taken to process the query in milliseconds"
    )


class CategoryClassification(BaseModel):
    """Structured output for category classification with confidence.
    
    Attributes:
        category: The classified category.
        confidence: Confidence score (0-100) for the classification.
        reasoning: Brief reasoning for the classification decision.
    """
    
    category: Literal[
        "Billing",
        "Technical Support",
        "Returns & Refunds",
        "Shipping & Delivery",
        "Account Management",
        "General Inquiry"
    ] = Field(description="The classified category of the query")
    
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score (0-100) for the classification"
    )
    
    reasoning: str = Field(
        description="Brief reasoning for the classification decision"
    )


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis with confidence.
    
    Attributes:
        sentiment: The detected sentiment.
        confidence: Confidence score (0-100) for the analysis.
        reasoning: Brief reasoning for the sentiment detection.
    """
    
    sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative",
        "Urgent",
        "Frustrated"
    ] = Field(description="The detected sentiment of the query")
    
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score (0-100) for the sentiment analysis"
    )
    
    reasoning: str = Field(
        description="Brief reasoning for the sentiment detection"
    )


class PriorityAssessment(BaseModel):
    """Structured output for priority assessment.
    
    Attributes:
        priority: The assessed priority level.
        reasoning: Brief reasoning for the priority decision.
    """
    
    priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="The assessed priority level"
    )
    
    reasoning: str = Field(
        description="Brief reasoning for the priority decision"
    )


class EscalationDecision(BaseModel):
    """Structured output for escalation decisions.
    
    Attributes:
        should_escalate: Whether escalation is recommended.
        reason: Explanation for the escalation decision.
        suggested_team: Recommended team for handling if escalated.
    """
    
    should_escalate: bool = Field(
        description="Whether escalation to a human agent is recommended"
    )
    
    reason: str = Field(
        description="Explanation for the escalation decision"
    )
    
    suggested_team: Optional[str] = Field(
        default=None,
        description="Recommended team for handling the escalated query"
    )


class LanguageDetection(BaseModel):
    """Structured output for language detection.
    
    Attributes:
        language_name: Full name of the detected language.
        language_code: ISO 639-1 language code.
        confidence: Confidence score for the detection.
    """
    
    language_name: str = Field(
        description="Full name of the detected language (e.g., 'English', 'Spanish')"
    )
    
    language_code: str = Field(
        description="ISO 639-1 language code (e.g., 'en', 'es')"
    )
    
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score for the language detection"
    )


class Article(BaseModel):
    """Structured output for support article from knowledge base.
    
    Attributes:
        id: Unique identifier for the article.
        title: Title of the support article.
        category: Category the article belongs to.
        content: Full content of the article.
        tags: List of tags for the article.
    """
    
    id: str = Field(description="Unique identifier for the article")
    
    title: str = Field(description="Title of the support article")
    
    category: str = Field(description="Category the article belongs to")
    
    content: str = Field(description="Full content of the support article")
    
    tags: list[str] = Field(default_factory=list, description="List of tags for the article")
