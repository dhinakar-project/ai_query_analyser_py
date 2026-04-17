"""Classification tool for categorizing customer queries."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_gemini


@tool
def classify_query(query: str) -> str:
    """Classify a customer query into one of these categories: Billing, Technical Support, Returns & Refunds, Shipping & Delivery, Account Management, General Inquiry.
    
    Args:
        query: The customer's query text to classify.
    
    Returns:
        The category label for the query.
    """
    llm = get_gemini(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer query classifier. Your task is to classify the given query into EXACTLY ONE of the following categories:\n- Billing\n- Technical Support\n- Returns & Refunds\n- Shipping & Delivery\n- Account Management\n- General Inquiry\n\nRespond with ONLY the category name, nothing else. No punctuation, no explanation, just the category."),
        ("human", "Customer Query: {query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})
