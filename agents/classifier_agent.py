"""Classifier agent using real LangChain ReAct agent with tool calling."""

import logging
import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_classifier_llm

logger = logging.getLogger(__name__)


VALID_CATEGORIES = [
    "Billing",
    "Technical Support",
    "Returns & Refunds",
    "Shipping & Delivery",
    "Account Management",
    "General Inquiry"
]


@tool
def classify_query_with_confidence(query: str) -> str:
    """Classify a customer query into exactly one category with confidence scoring.
    
    This tool analyzes the customer query to determine its category and
    provides a confidence score for the classification.
    
    Args:
        query: The customer's query text to classify.
    
    Returns:
        A JSON-formatted string containing:
        - category: The classified category (Billing, Technical Support, etc.)
        - confidence: Confidence score (0-100)
        - reasoning: Brief explanation for the classification
    
    Categories:
        - Billing: Payment issues, charges, invoices, refunds, pricing
        - Technical Support: Software/hardware issues, troubleshooting, errors, bugs
        - Returns & Refunds: Product returns, money back, exchanges
        - Shipping & Delivery: Order tracking, delivery issues, delays
        - Account Management: Login, password, profile, settings, subscription
        - General Inquiry: Questions about products, services, company info
    """
    logger.debug(f"Classifying query: {query[:50]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert customer query classifier. Your task is to classify customer queries into EXACTLY ONE category.

Available Categories:
1. Billing - Payment issues, charges, invoices, billing disputes, pricing questions, subscription billing, refunds related to charges
2. Technical Support - Software problems, hardware issues, login errors, bugs, crashes, troubleshooting, app not working, integration issues
3. Returns & Refunds - Product returns, money back guarantees, exchanges, damaged items, wrong items received
4. Shipping & Delivery - Order tracking, delivery delays, lost packages, shipping costs, delivery address changes
5. Account Management - Login issues, password reset, profile updates, account security, subscription changes, account deletion
6. General Inquiry - Questions about products, services, company policies, business hours, general questions

Classification Rules:
- "how much" "price" "charged" "bill" "payment" → Billing
- "login" "password" "account" "profile" "subscription" → Account Management
- "return" "refund" "exchange" "damaged" "wrong item" → Returns & Refunds
- "shipping" "delivery" "tracking" "package" "arrived" → Shipping & Delivery
- "error" "crash" "bug" "not working" "broken" "fix" → Technical Support
- Questions about products/services → General Inquiry

Also assess your confidence in this classification from 0-100.

Respond with ONLY a JSON object in this format (no markdown, no explanation):
{{"category": "CategoryName", "confidence": 85, "reasoning": "Brief 1-2 sentence explanation"}}"""),
        ("human", "Classify this customer query:\n\n{query}")
    ])
    
    llm = get_classifier_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": query})
        result = result.strip()
        
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(line for line in lines if not line.strip().startswith("```"))
            result = result.strip()
        
        logger.debug(f"Classification result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return '{"category": "General Inquiry", "confidence": 50, "reasoning": "Error in classification, defaulting to General Inquiry"}'

def run_classifier_agent(query: str) -> str:
    """Run the classifier agent to categorize a customer query.
    
    Args:
        query: The customer's query text.
    
    Returns:
        The classified category string.
    """
    logger.info(f"Running classifier agent for query: {query[:50]}...")
    
    try:
        result_str = classify_query_with_confidence.invoke({"query": query})
        
        try:
            result = json.loads(result_str)
            category = result.get("category", "General Inquiry")
            
            if category not in VALID_CATEGORIES:
                logger.warning(f"Invalid category returned: {category}, defaulting to General Inquiry")
                category = "General Inquiry"
            
            logger.debug(f"Query classified as: {category}")
            return category
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse classification result: {result_str}")
            return "General Inquiry"
            
    except Exception as e:
        logger.error(f"Classifier agent error: {e}")
        return "General Inquiry"


def run_classifier_agent_with_confidence(query: str) -> Dict[str, Any]:
    """Run the classifier agent and return category with confidence score.
    
    Args:
        query: The customer's query text.
    
    Returns:
        Dictionary containing category, confidence, and reasoning.
    """
    logger.info(f"Running classifier agent with confidence for query: {query[:50]}...")
    
    try:
        result_str = classify_query_with_confidence.invoke({"query": query})
        
        try:
            result = json.loads(result_str)
            return {
                "category": result.get("category", "General Inquiry"),
                "confidence": result.get("confidence", 50),
                "reasoning": result.get("reasoning", "Standard classification")
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse classification result: {result_str}")
            return {
                "category": "General Inquiry",
                "confidence": 50,
                "reasoning": "Error in classification"
            }
            
    except Exception as e:
        logger.error(f"Classifier agent error: {e}")
        return {
            "category": "General Inquiry",
            "confidence": 50,
            "reasoning": "Error in classification"
        }
