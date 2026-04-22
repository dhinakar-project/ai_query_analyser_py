"""Unit tests for Pydantic output models."""
import pytest
from models.analysis_result import CombinedAnalysis, Article


def test_combined_analysis_valid():
    obj = CombinedAnalysis(
        language="English",
        language_code="en",
        category="Billing",
        category_confidence=90,
        sentiment="Frustrated",
        sentiment_confidence=85,
        priority="High",
        should_escalate=True,
        escalation_reason="Customer frustrated with repeated billing issue",
        suggested_team="Billing Support Team",
        category_reasoning="Query mentions billing and charges",
        sentiment_reasoning="Customer used frustrated language",
    )
    assert obj.category == "Billing"
    assert obj.should_escalate is True


def test_combined_analysis_invalid_category():
    with pytest.raises(Exception):
        CombinedAnalysis(
            language="English",
            language_code="en",
            category="InvalidCategory",
            category_confidence=90,
            sentiment="Neutral",
            sentiment_confidence=80,
            priority="Low",
            should_escalate=False,
            category_reasoning="test",
            sentiment_reasoning="test",
        )


def test_article_model():
    article = Article(
        id="art_001",
        title="How to reset your password",
        category="Account Management",
        content="To reset your password, go to the login page...",
        tags=["password", "account", "login"]
    )
    assert article.id == "art_001"
    assert len(article.tags) == 3


def test_confidence_bounds():
    with pytest.raises(Exception):
        CombinedAnalysis(
            language="English",
            language_code="en",
            category="Billing",
            category_confidence=150,
            sentiment="Neutral",
            sentiment_confidence=80,
            priority="Low",
            should_escalate=False,
            category_reasoning="test",
            sentiment_reasoning="test",
        )
