"""Agents package for query classification, sentiment analysis, and response generation.

This package contains LangChain agents for:
- ClassifierAgent: Categorizes customer queries with confidence scoring
- SentimentAgent: Analyzes emotional tone with confidence scoring
- ResponderAgent: Generates contextual responses
- PriorityAgent: Assesses query priority for handling decisions
- EscalationAgent: Determines when to escalate to human agents
"""

from agents.classifier_agent import (
    run_classifier_agent,
    run_classifier_agent_with_confidence
)
from agents.sentiment_agent import (
    run_sentiment_agent,
    run_sentiment_agent_with_confidence
)
from agents.responder_agent import (
    run_responder_agent
)
from agents.priority_agent import (
    run_priority_agent
)
from agents.escalation_agent import (
    run_escalation_agent,
    get_suggested_team_for_category
)

__all__ = [
    "run_classifier_agent",
    "run_classifier_agent_with_confidence",
    "run_sentiment_agent",
    "run_sentiment_agent_with_confidence",
    "run_responder_agent",
    "run_priority_agent",
    "run_escalation_agent",
    "get_suggested_team_for_category"
]
