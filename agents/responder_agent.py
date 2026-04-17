"""Responder agent for generating contextual customer support responses."""

from tools.response_tool import generate_response


def run_responder_agent(query: str, category: str, sentiment: str) -> str:
    try:
        result = generate_response.invoke({
            "query": query,
            "category": category,
            "sentiment": sentiment
        })
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()
    except Exception as e:
        print(f"[DEBUG] Responder error: {e}")
        return "I apologize for any inconvenience. Our team is here to help you with your inquiry. Please provide more details about your concern, and we'll do our best to assist you promptly."
