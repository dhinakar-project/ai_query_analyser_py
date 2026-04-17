"""Sentiment analysis tool for analyzing customer query sentiment."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_gemini


@tool
def analyze_sentiment(query: str) -> str:
    """Analyze the sentiment of a customer query. Returns one of: Positive, Neutral, Negative, Urgent, Frustrated.
    
    Args:
        query: The customer's query text to analyze.
    
    Returns:
        The sentiment label for the query.
    """
    llm = get_gemini(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analysis specialist. Analyze the emotional tone of the customer query and return EXACTLY ONE sentiment label:\n- Positive: Customer is satisfied, happy, or expressing gratitude\n- Neutral: Customer is asking a normal question without strong emotion\n- Negative: Customer is unhappy or disappointed\n- Urgent: Customer needs immediate attention or is in distress\n- Frustrated: Customer is annoyed or expressing repeated issues\n\nRespond with ONLY the sentiment label, nothing else. No punctuation, no explanation."),
        ("human", "Customer Query: {query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})
