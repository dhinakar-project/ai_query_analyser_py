"""Sentiment agent for analyzing customer query sentiment."""

from tools.sentiment_tool import analyze_sentiment


def run_sentiment_agent(query: str) -> str:
    try:
        result = analyze_sentiment.invoke({"query": query})
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()
    except Exception as e:
        print(f"[DEBUG] Sentiment error: {e}")
        return "Neutral"
