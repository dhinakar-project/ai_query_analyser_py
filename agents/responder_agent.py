"""Responder agent for generating contextual customer support responses."""

from tools.response_tool import generate_response


def run_responder_agent(query: str, category: str, sentiment: str) -> str:
    """Run the responder tool to generate a response.
    
    Args:
        query: The original customer query.
        category: The classified category.
        sentiment: The analyzed sentiment.
    
    Returns:
        The generated response text.
    """
    try:
        result = generate_response.invoke({
            "query": query, 
            "category": category, 
            "sentiment": sentiment
        })
        return result.content.strip() if hasattr(result, 'content') else str(result).strip()
    except Exception:
        return "I apologize for any inconvenience. Our team is here to help you with your inquiry. Please provide more details about your concern, and we'll do our best to assist you promptly."
