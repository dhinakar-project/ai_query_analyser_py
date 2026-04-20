"""LLM utility module for Google Gemini integration with retry logic."""

import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger(__name__)
_retry_logger = logging.getLogger("llm_retry")

_MODEL_NAME = "gemini-2.5-flash"
_MODEL_NAME_PRO = "gemini-2.5-flash"


class LLMConfigurationError(Exception):
    """Raised when LLM configuration is invalid."""
    pass


class LLMConnectionError(Exception):
    """Raised when there's a connection issue with the LLM service."""
    pass


def _get_api_key() -> str:
    """Retrieve and validate the Gemini API key from environment."""
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
    """Create and cache an LLM instance."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=_get_api_key()
    )


_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

try:
    import google.api_core.exceptions as _gex
    _RETRY_EXCEPTIONS = _RETRY_EXCEPTIONS + (
        _gex.ResourceExhausted,
        _gex.ServiceUnavailable,
        _gex.InternalServerError,
        _gex.DeadlineExceeded,
    )
except ImportError:
    pass


_retry_policy = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
    reraise=True,
)


@_retry_policy
def get_gemini(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Get a Gemini Flash LLM instance with retry logic."""
    logger.debug(f"Getting Gemini Flash LLM with temperature={temperature}")
    return _create_llm(model=_MODEL_NAME, temperature=temperature)


@_retry_policy
def get_gemini_pro(temperature: float = 0.6) -> ChatGoogleGenerativeAI:
    """Get a Gemini LLM instance optimized for response generation."""
    logger.debug(f"Getting responder LLM with temperature={temperature}")
    return get_gemini(temperature=temperature)


def get_classifier_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for classification tasks."""
    return get_gemini(temperature=0.0)


def get_sentiment_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for sentiment analysis tasks."""
    return get_gemini(temperature=0.1)


def get_priority_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for priority assessment tasks."""
    return get_gemini(temperature=0.0)


def get_responder_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for response generation."""
    return get_gemini_pro(temperature=0.6)


def get_escalation_llm() -> ChatGoogleGenerativeAI:
    """Get an LLM optimized for escalation decisions."""
    return get_gemini(temperature=0.1)