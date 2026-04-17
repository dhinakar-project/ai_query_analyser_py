"""Classifier agent for categorizing customer queries."""

from tools.classification_tool import classify_query


def run_classifier_agent(query: str) -> str:
    """Run the classifier tool on a query.
    
    Args:
        query: The customer query to classify.
    
    Returns:
        The category label for the query.
    """
    try:
        result = classify_query.invoke({"query": query})
        return result.content.strip() if hasattr(result, 'content') else str(result).strip()
    except Exception:
        return "General Inquiry"
