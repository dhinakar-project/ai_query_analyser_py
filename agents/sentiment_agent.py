"""Sentiment analysis agent using real LangChain ReAct agent with tool calling."""

import logging
import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_sentiment_llm

logger = logging.getLogger(__name__)


VALID_SENTIMENTS = [
    "Positive",
    "Neutral",
    "Negative",
    "Urgent",
    "Frustrated"
]


@tool
def analyze_sentiment_with_confidence(query: str) -> str:
    """Analyze the sentiment of a customer query with confidence scoring.
    
    This tool detects the emotional tone of a customer query and provides
    a confidence score for the analysis.
    
    Args:
        query: The customer's query text to analyze.
    
    Returns:
        A JSON-formatted string containing:
        - sentiment: The detected sentiment (Positive, Neutral, Negative, Urgent, Frustrated)
        - confidence: Confidence score (0-100)
        - reasoning: Brief explanation for the sentiment detection
    
    Sentiment Definitions:
        - Positive: Customer is satisfied, happy, expressing gratitude, complimenting
        - Neutral: Customer is asking a normal question, no strong emotion expressed
        - Negative: Customer is unhappy, disappointed, dissatisfied with service/product
        - Urgent: Customer needs immediate attention, time-sensitive issue, distress
        - Frustrated: Customer is annoyed, angry, has experienced repeated issues
    """
    logger.debug(f"Analyzing sentiment for query: {query[:50]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert sentiment analysis specialist. Your task is to analyze the emotional tone of customer queries.

Sentiment Categories:
1. Positive - Customer expresses satisfaction, happiness, gratitude, or compliments
   Examples: "Thank you so much!", "I love this product!", "Great service!"
   
2. Neutral - Customer asks a normal question or provides information without emotion
   Examples: "What are your business hours?", "I'd like to know more about..."
   
3. Negative - Customer is unhappy, disappointed, or dissatisfied
   Examples: "I'm not happy with this", "This is disappointing", "I expected better"
   
4. Urgent - Customer needs immediate attention, situation is time-sensitive or distressing
   Examples: "I need this fixed NOW!", "This is an emergency", "My business is down!"
   Triggers: "immediately", "urgent", "emergency", "asap", "critical", "deadline"
   
5. Frustrated - Customer is annoyed, has experienced repeated issues, or is losing patience
   Examples: "I've already contacted you 3 times about this!", "This keeps happening!"
   Triggers: "again", "still", "another", "waste", "ridiculous", "unacceptable"

Analysis Guidelines:
- Look for emotional keywords and intensity
- Consider CAPS_LOCK usage (SHOUTING indicates stronger emotion)
- Check for repeated exclamation marks
- Note repeated issues (frustration indicator)
- Assess if immediate action is requested (urgency indicator)
- Be conservative with Positive - make sure there's genuine satisfaction

Also assess your confidence in this sentiment analysis from 0-100.

Respond with ONLY a JSON object in this format (no markdown, no explanation):
{{"sentiment": "SentimentType", "confidence": 85, "reasoning": "Brief 1-2 sentence explanation"}}"""),
        ("human", "Analyze the sentiment of this customer query:\n\n{query}")
    ])
    
    llm = get_sentiment_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": query})
        result = result.strip()
        
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(line for line in lines if not line.strip().startswith("```"))
            result = result.strip()
        
        logger.debug(f"Sentiment analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return '{"sentiment": "Neutral", "confidence": 50, "reasoning": "Error in analysis, defaulting to Neutral"}'

def run_sentiment_agent(query: str) -> str:
    """Run the sentiment agent to analyze a customer query.
    
    Args:
        query: The customer's query text.
    
    Returns:
        The detected sentiment string.
    """
    logger.info(f"Running sentiment agent for query: {query[:50]}...")
    
    try:
        result_str = analyze_sentiment_with_confidence.invoke({"query": query})
        
        try:
            result = json.loads(result_str)
            sentiment = result.get("sentiment", "Neutral")
            
            if sentiment not in VALID_SENTIMENTS:
                logger.warning(f"Invalid sentiment returned: {sentiment}, defaulting to Neutral")
                sentiment = "Neutral"
            
            logger.debug(f"Query sentiment: {sentiment}")
            return sentiment
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse sentiment result: {result_str}")
            return "Neutral"
            
    except Exception as e:
        logger.error(f"Sentiment agent error: {e}")
        return "Neutral"


def run_sentiment_agent_with_confidence(query: str) -> Dict[str, Any]:
    """Run the sentiment agent and return sentiment with confidence score.
    
    Args:
        query: The customer's query text.
    
    Returns:
        Dictionary containing sentiment, confidence, and reasoning.
    """
    logger.info(f"Running sentiment agent with confidence for query: {query[:50]}...")
    
    try:
        result_str = analyze_sentiment_with_confidence.invoke({"query": query})
        
        try:
            result = json.loads(result_str)
            return {
                "sentiment": result.get("sentiment", "Neutral"),
                "confidence": result.get("confidence", 50),
                "reasoning": result.get("reasoning", "Standard sentiment analysis")
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse sentiment result: {result_str}")
            return {
                "sentiment": "Neutral",
                "confidence": 50,
                "reasoning": "Error in sentiment analysis"
            }
            
    except Exception as e:
        logger.error(f"Sentiment agent error: {e}")
        return {
            "sentiment": "Neutral",
            "confidence": 50,
            "reasoning": "Error in sentiment analysis"
        }
