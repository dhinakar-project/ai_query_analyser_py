"""Sentiment agent for analyzing customer query sentiment."""

from tools.sentiment_tool import analyze_sentiment


def run_sentiment_agent(query: str) -> str:
    """Run the sentiment analysis tool on a query.
    
    Args:
        query: The customer query to analyze.
    
    Returns:
        The sentiment label for the query.
    """
    try:
        result = analyze_sentiment.invoke({"query": query})
        return result.content.strip() if hasattr(result, 'content') else str(result).strip()
    except Exception:
        return "Neutral"
