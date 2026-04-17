"""Priority assessment agent using real LangChain agent with tools."""

import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_priority_llm

logger = logging.getLogger(__name__)


@tool
def assess_priority(query: str, sentiment: str, category: str) -> str:
    """Assess the priority level of a customer query based on content, sentiment, and category.
    
    This tool analyzes the query details to determine the appropriate priority level
    for response and escalation handling.
    
    Args:
        query: The customer's original query text.
        sentiment: The detected sentiment (Positive, Neutral, Negative, Urgent, Frustrated).
        category: The classified category (Billing, Technical Support, etc.).
    
    Returns:
        A JSON-formatted string containing:
        - priority: One of Critical, High, Medium, Low
        - reasoning: Brief explanation for the priority decision
    
    Priority Rules:
        - Critical: Urgent/Frustrated + Billing, or Technical issues with keywords like "emergency", "down", "outage"
        - High: Frustrated + any category, or Urgent + Technical Support, or Negative + Billing
        - Medium: Negative sentiment in most categories, or any query from Technical Support
        - Low: Positive or Neutral sentiment, General Inquiry category
    """
    logger.debug(f"Assessing priority: sentiment={sentiment}, category={category}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a priority assessment specialist for customer support. Your task is to determine the appropriate priority level for a customer query.

Priority Levels:
- Critical: Requires immediate human attention. Examples: urgent billing disputes, system outages affecting many users, security issues
- High: Should be handled soon but not immediately. Examples: frustrated customers, urgent technical issues, negative billing feedback
- Medium: Standard priority. Examples: general complaints, standard technical questions, shipping concerns
- Low: Can be handled when convenient. Examples: positive feedback, simple inquiries, general questions

Critical Priority Triggers:
- Urgent OR Frustrated + Billing (disputes, charges, payment issues)
- Technical Support + keywords: "emergency", "down", "outage", "crash", "not working", "broken", "urgent"
- Account Management + security concerns: "hacked", "unauthorized", "breach"

High Priority Triggers:
- Frustrated + any category
- Urgent + Technical Support
- Negative + Billing
- Technical Support + complex issues

Medium Priority Triggers:
- Negative + most categories (except Billing which may be High)
- Technical Support + standard questions
- Shipping & Delivery + delays

Low Priority Triggers:
- Positive sentiment
- Neutral sentiment + General Inquiry
- Simple questions

Respond with ONLY a JSON object in this format (no markdown, no explanation):
{{"priority": "Level", "reasoning": "Brief 1-2 sentence explanation"}}"""),
        ("human", """Analyze this customer query and determine priority:

Query: {query}
Detected Sentiment: {sentiment}
Classified Category: {category}

What is the priority level and reasoning?""")
    ])
    
    llm = get_priority_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "query": query,
            "sentiment": sentiment,
            "category": category
        })
        
        result = result.strip()
        
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(line for line in lines if not line.strip().startswith("```"))
            result = result.strip()
        
        logger.debug(f"Priority assessment result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Priority assessment error: {e}")
        return '{"priority": "Medium", "reasoning": "Unable to assess priority due to processing error"}'

def run_priority_agent(query: str, sentiment: str, category: str) -> Dict[str, Any]:
    """Run the priority assessment agent on a customer query.
    
    Args:
        query: The customer's query text.
        sentiment: The detected sentiment.
        category: The classified category.
    
    Returns:
        Dictionary containing priority and reasoning.
    """
    logger.info(f"Running priority agent for: category={category}, sentiment={sentiment}")
    
    try:
        result_str = assess_priority.invoke({
            "query": query,
            "sentiment": sentiment,
            "category": category
        })
        
        import json
        try:
            result = json.loads(result_str)
            priority = result.get("priority", "Medium")
            reasoning = result.get("reasoning", "Standard priority assessment")
        except json.JSONDecodeError:
            if "Critical" in result_str:
                priority = "Critical"
            elif "High" in result_str:
                priority = "High"
            elif "Low" in result_str:
                priority = "Low"
            else:
                priority = "Medium"
            reasoning = "Priority determined by keyword analysis"
        
        logger.debug(f"Priority determined: {priority}")
        return {
            "priority": priority,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"Priority agent error: {e}")
        return {
            "priority": "Medium",
            "reasoning": "Error in priority assessment, defaulting to Medium"
        }
