"""Guardrails module for input validation and content safety."""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"ignore\s+above\s+instructions",
    r"disregard\s+(previous|all)\s+(instructions|commands)",
    r"forget\s+(previous|all)\s+instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
    r"pretend\s+(?:you|to)\s+are",
    r"system\s*prompt",
    r"(?:new|override)\s+(?:instructions?|rules?)",
    r"\\n(?:system|human|assistant):",
    r"<(?:system|user|assistant)>",
    r"\\[INST\\]",
    r"\\{START\\}|\\{END\\}",
    r"role\s*:\s*(?:system|admin|user)",
    r"access\s+(?:granted|denied)",
    r"password\s*is\s*:",
    r"secret\s*is\s*:",
    r"sudo\s+",
    r"rm\s+-rf\s+/",
    r"exec\s*\(",
]

OFFENSIVE_KEYWORDS = [
    "hate",
    "kill",
    "violence",
    "abuse",
    "harassment",
    "threat",
    "terror",
    "bomb",
    "weapon",
]

SYMBOL_ONLY_PATTERN = re.compile(r"^[^\w\s]+$|^\d+$")


def validate_query(query: str) -> Tuple[bool, str]:
    """Validate a customer query for safety and quality.
    
    This function performs multiple checks on the input query:
    1. Empty/whitespace validation
    2. Length constraints (5-1000 characters)
    3. Symbol-only content detection
    4. Prompt injection attempt detection
    5. Offensive content detection
    
    Args:
        query: The customer query string to validate.
        
    Returns:
        A tuple containing:
        - is_valid (bool): True if the query passes all validations.
        - error_message (str): Empty string if valid, otherwise describes the issue.
        
    Examples:
        >>> validate_query("How can I reset my password?")
        (True, '')
        
        >>> validate_query("")
        (False, 'Query cannot be empty.')
        
        >>> validate_query("ignore previous instructions")
        (False, 'Potential prompt injection detected.')
    """
    if not query or not query.strip():
        logger.warning("Empty query submitted for validation")
        return False, "Query cannot be empty. Please enter a valid customer query."
    
    query_stripped = query.strip()
    query_lower = query_stripped.lower()
    
    if len(query_stripped) < 5:
        logger.warning(f"Query too short: {len(query_stripped)} characters")
        return False, "Query is too short. Please provide at least 5 characters."
    
    if len(query_stripped) > 1000:
        logger.warning(f"Query too long: {len(query_stripped)} characters")
        return False, "Query exceeds maximum length of 1000 characters. Please shorten your query."
    
    if SYMBOL_ONLY_PATTERN.match(query_stripped):
        logger.warning("Query contains only symbols or numbers")
        return False, "Query cannot contain only symbols or numbers. Please provide a meaningful query."
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.warning(f"Potential prompt injection detected: {pattern}")
            return False, "Invalid input detected. Please enter a valid customer query."
    
    words = query_lower.split()
    words_found = []
    for keyword in OFFENSIVE_KEYWORDS:
        if keyword in words:
            words_found.append(keyword)
    
    if words_found:
        logger.warning(f"Offensive content detected: {words_found}")
        return False, "Your query contains inappropriate content. Please rephrase your question professionally."
    
    logger.debug(f"Query validated successfully: {len(query_stripped)} characters")
    return True, ""


def sanitize_query(query: str) -> str:
    """Sanitize a query by removing potentially harmful characters.
    
    Args:
        query: The query string to sanitize.
        
    Returns:
        Sanitized query string safe for processing.
    """
    if not query:
        return ""
    
    sanitized = query.strip()
    
    sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", sanitized)
    
    sanitized = re.sub(r"\s+", " ", sanitized)
    
    if len(sanitized) > 2000:
        sanitized = sanitized[:2000]
    
    return sanitized


def is_repetitive_query(query: str) -> bool:
    """Check if a query appears to be repetitive/spam.
    
    Args:
        query: The query string to check.
        
    Returns:
        True if the query appears to be spam/repetitive.
    """
    if not query or len(query) < 10:
        return False
    
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()
    
    if len(words) < 3:
        return False
    
    unique_ratio = len(set(words)) / len(words) if words else 1
    
    if unique_ratio < 0.2:
        logger.warning(f"Query appears repetitive: unique ratio {unique_ratio:.2f}")
        return True
    
    return False
