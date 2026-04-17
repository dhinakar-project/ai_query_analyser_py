"""Classifier agent for categorizing customer queries."""

from tools.classification_tool import classify_query


def run_classifier_agent(query: str) -> str:
    try:
        result = classify_query.invoke({"query": query})
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()
    except Exception as e:
        print(f"[DEBUG] Classifier error: {e}")
        return "General Inquiry"
