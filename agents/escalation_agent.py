"""Escalation decision agent using real LangChain agent with tools."""

import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_escalation_llm

logger = logging.getLogger(__name__)


SUGGESTED_TEAMS = {
    "Billing": "Billing Support Team",
    "Technical Support": "Technical Support Team",
    "Returns & Refunds": "Returns & Refunds Team",
    "Shipping & Delivery": "Shipping & Logistics Team",
    "Account Management": "Account Security Team",
    "General Inquiry": "Customer Service Team"
}


@tool
def check_escalation(priority: str, sentiment: str, query: str) -> str:
    """Determine if a customer query should be escalated to a human agent.
    
    This tool analyzes the priority level, sentiment, and query content to
    decide whether human intervention is needed.
    
    Args:
        priority: The assessed priority level (Critical, High, Medium, Low).
        sentiment: The detected sentiment (Positive, Neutral, Negative, Urgent, Frustrated).
        query: The customer's original query text.
    
    Returns:
        A JSON-formatted string containing:
        - escalate: Boolean indicating if escalation is recommended
        - reason: Explanation for the escalation decision
        - suggested_team: Recommended team to handle the escalated query
    
    Escalation Triggers:
        - Critical priority: Always escalate
        - High priority + Frustrated/Urgent sentiment: Escalate
        - Any priority + keywords: "lawsuit", "lawyer", "attorney", "legal": Escalate
        - Any priority + keywords: "suicide", "self-harm", "emergency": Escalate (emergency team)
        - High priority + security concerns: Escalate
    """
    logger.debug(f"Checking escalation: priority={priority}, sentiment={sentiment}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an escalation decision specialist for customer support. Your task is to determine when a query needs human agent intervention.

Escalation Guidelines:
- CRITICAL Priority: ALWAYS escalate to human agent
- HIGH Priority + Frustrated/Urgent: Escalate
- HIGH Priority + Negative sentiment: Consider escalation
- MEDIUM Priority: Generally auto-handle, but escalate if customer seems unsatisfied
- LOW Priority: Auto-handle with AI, no escalation needed

CRITICAL Escalation Situations (ALWAYS escalate):
- Queries containing: "lawsuit", "lawyer", "attorney", "legal action", "court"
- Queries containing: "suicide", "self-harm", "harm myself", "hurt myself"
- Queries containing: "emergency", "life-threatening", "medical emergency"
- Queries containing: "data breach", "security incident", "hacked account"
- Critical priority level

HIGH Escalation Situations:
- Frustrated customers who have contacted multiple times
- High priority queries with negative sentiment
- Billing disputes over $500
- Complex technical issues not resolved after troubleshooting
- Customer explicitly requests supervisor

AUTO-HANDLE (No escalation):
- Positive/Neutral sentiment
- Low/Medium priority with standard queries
- Simple information requests
- Thank you messages
- Routine transactions

Suggested Teams:
- Billing: "Billing Support Team"
- Technical Support: "Technical Support Team"
- Returns & Refunds: "Returns & Refunds Team"
- Shipping & Delivery: "Shipping & Logistics Team"
- Account Management: "Account Security Team"
- General Inquiry: "Customer Service Team"

Respond with ONLY a JSON object in this format (no markdown, no explanation):
{{"escalate": true/false, "reason": "Brief explanation (1-2 sentences)", "suggested_team": "Team Name or null"}}"""),
        ("human", """Analyze this query for escalation decision:

Priority Level: {priority}
Customer Sentiment: {sentiment}
Query Content: {query}

Should this query be escalated to a human agent?""")
    ])
    
    llm = get_escalation_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "priority": priority,
            "sentiment": sentiment,
            "query": query
        })
        
        result = result.strip()
        
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(line for line in lines if not line.strip().startswith("```"))
            result = result.strip()
        
        logger.debug(f"Escalation check result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Escalation check error: {e}")
        return '{"escalate": false, "reason": "Error in escalation check, defaulting to auto-handle", "suggested_team": null}'

def run_escalation_agent(priority: str, sentiment: str, query: str) -> Dict[str, Any]:
    """Run the escalation decision agent on a query.
    
    Args:
        priority: The assessed priority level.
        sentiment: The detected sentiment.
        query: The customer's query text.
    
    Returns:
        Dictionary containing escalation decision, reason, and suggested team.
    """
    logger.info(f"Running escalation check: priority={priority}, sentiment={sentiment}")
    
    try:
        result_str = check_escalation.invoke({
            "priority": priority,
            "sentiment": sentiment,
            "query": query
        })
        
        import json
        try:
            result = json.loads(result_str)
            should_escalate = result.get("escalate", False)
            reason = result.get("reason", "Standard escalation check")
            suggested_team = result.get("suggested_team")
        except json.JSONDecodeError:
            should_escalate = "true" in result_str.lower() and "false" not in result_str.split("true")[0]
            reason = "Escalation determined by keyword analysis"
            suggested_team = None
        
        if should_escalate and not suggested_team:
            suggested_team = "Customer Service Team"
        
        logger.debug(f"Escalation decision: {should_escalate}, team: {suggested_team}")
        return {
            "should_escalate": should_escalate,
            "reason": reason,
            "suggested_team": suggested_team
        }
        
    except Exception as e:
        logger.error(f"Escalation agent error: {e}")
        return {
            "should_escalate": False,
            "reason": "Error in escalation check",
            "suggested_team": None
        }


def get_suggested_team_for_category(category: str) -> str:
    """Get the suggested team name for a given category.
    
    Args:
        category: The query category.
    
    Returns:
        The suggested team name.
    """
    return SUGGESTED_TEAMS.get(category, "Customer Service Team")
