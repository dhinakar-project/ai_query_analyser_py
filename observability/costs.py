"""Cost tracking for LLM API calls."""

import os
from typing import Optional

GEMINI_COST_PER_1K_INPUT = 0.000075
GEMINI_COST_PER_1K_OUTPUT = 0.0003

_session_cost: float = 0.0


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call.
    
    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        
    Returns:
        Cost in USD.
    """
    input_cost = (input_tokens / 1000) * GEMINI_COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * GEMINI_COST_PER_1K_OUTPUT
    return input_cost + output_cost


def add_cost(cost: float) -> None:
    """Add cost to the session total.
    
    Args:
        cost: Cost to add in USD.
    """
    global _session_cost
    _session_cost += cost


def get_session_cost() -> float:
    """Get the current session cost total.
    
    Returns:
        Total cost in USD.
    """
    return _session_cost


def reset_session_cost() -> None:
    """Reset the session cost to zero."""
    global _session_cost
    _session_cost = 0.0


def format_cost(cost: float) -> str:
    """Format cost as a dollar string.
    
    Args:
        cost: Cost in USD.
        
    Returns:
        Formatted string like "$0.0023".
    """
    return f"${cost:.4f}"


def estimate_cost_from_tokens(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost and add to session total.
    
    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        
    Returns:
        Estimated cost in USD.
    """
    cost = calculate_cost(input_tokens, output_tokens)
    add_cost(cost)
    return cost