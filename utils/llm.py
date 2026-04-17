"""LLM utility module for Google Gemini integration with retry logic."""

import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Raised when LLM configuration is invalid."""
    pass


class LLMConnectionError(Exception):
    """Raised when there's a connection issue with the LLM service."""
    pass


def _get_api_key() -> str:
    """Retrieve and validate the Gemini API key from environment.
    
    Returns:
        The API key string.
        
    Raises:
        LLMConfigurationError: If API key is not found or empty.
    """
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        raise LLMConfigurationError(
            "GEMINI_API_KEY not found. Please add your API key to the .env file."
        )
    if not api_key.strip():
        logger.error("GEMINI_API_KEY is empty")
        raise LLMConfigurationError("GEMINI_API_KEY is empty. Please provide a valid API key.")
    return api_key.strip()


@lru_cache(maxsize=4)
def _create_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Create and cache an LLM instance.
    
    Args:
        model: The model name to use.
        temperature: The temperature for generation (0.0-1.0).
        
    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=_get_api_key()
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True
)
def get_gemini(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Get a Gemini Flash LLM instance with retry logic.
    
    This function returns a cached LLM instance using gemini-2.5-flash model.
    It includes automatic retry with exponential backoff for transient failures.
    
    Args:
        temperature: The creativity/temperature level (0.0-1.0).
            - 0.0: Deterministic, factual responses (good for classification)
            - 0.3: Balanced (default)
            - 0.6-0.7: Creative responses (good for response generation)
            
    Returns:
        ChatGoogleGenerativeAI instance configured for gemini-2.5-flash.
        
    Raises:
        LLMConfigurationError: If API key is not configured.
    """
    logger.debug(f"Getting Gemini Flash LLM with temperature={temperature}")
    return _create_llm(model="gemini-2.5-flash", temperature=temperature)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True
)
def get_gemini_pro(temperature: float = 0.6) -> ChatGoogleGenerativeAI:
    """Get a Gemini LLM instance optimized for response generation.
    
    Falls back to gemini-2.5-flash since gemini-2.5-pro requires a paid tier.
    Use this for higher quality responses like the responder agent.
    
    Args:
        temperature: The creativity/temperature level (0.0-1.0).
            - 0.0-0.3: Factual, structured responses
            - 0.6-0.7: Creative, conversational responses (recommended)
            
    Returns:
        ChatGoogleGenerativeAI instance (uses gemini-2.5-flash for free tier).
        
    Raises:
        LLMConfigurationError: If API key is not configured.
    """
    logger.debug(f"Getting responder LLM with temperature={temperature}")
    return get_gemini(temperature=temperature)


def get_classifier_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for classification tasks.
    
    Uses low temperature for deterministic, accurate classifications.
    
    Returns:
        ChatGoogleGenerativeAI instance with temperature=0.0.
    """
    return get_gemini(temperature=0.0)


def get_sentiment_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for sentiment analysis tasks.
    
    Uses very low temperature for consistent sentiment detection.
    
    Returns:
        ChatGoogleGenerativeAI instance with temperature=0.1.
    """
    return get_gemini(temperature=0.1)


def get_priority_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for priority assessment tasks.
    
    Uses low temperature for consistent priority determination.
    
    Returns:
        ChatGoogleGenerativeAI instance with temperature=0.0.
    """
    return get_gemini(temperature=0.0)


def get_responder_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for response generation.
    
    Uses higher temperature for natural, empathetic responses.
    
    Returns:
        ChatGoogleGenerativeAI instance with temperature=0.6.
    """
    return get_gemini_pro(temperature=0.6)


def get_escalation_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for escalation decisions.
    
    Uses low temperature for consistent, logical escalation reasoning.
    
    Returns:
        ChatGoogleGenerativeAI instance with temperature=0.1.
    """
    return get_gemini(temperature=0.1)
