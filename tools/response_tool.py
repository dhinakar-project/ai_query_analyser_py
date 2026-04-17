"""Response generation tool for creating empathetic customer support responses."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_gemini


@tool
def generate_response(query: str, category: str, sentiment: str) -> str:
    """Generate a professional, empathetic customer support response given the query, its category, and its sentiment.
    
    Args:
        query: The original customer query.
        category: The classified category of the query.
        sentiment: The analyzed sentiment of the query.
    
    Returns:
        A contextual, empathetic response tailored to the query, category, and sentiment.
    """
    llm = get_gemini(temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional customer support specialist for our company. Your role is to generate helpful, empathetic responses that address customer concerns.

Guidelines:
1. Always address the customer's concern directly and professionally
2. Match your tone to the sentiment:
   - Urgent: Provide immediate reassurance and clear next steps
   - Frustrated: Show extra empathy, apologize for the inconvenience, and offer solutions
   - Negative: Be understanding and offer compensation or solutions
   - Positive: Be friendly, appreciative, and reinforce their positive experience
   - Neutral: Be helpful and informative
3. Use the category to provide relevant, specific help (e.g., billing issues should include account details, technical issues should include troubleshooting steps)
4. Keep responses between 3-5 sentences
5. Always end with an offer for further assistance
6. Never make up policy details - stay general but helpful"""),
        ("human", """Query: {query}
Category: {category}
Sentiment: {sentiment}""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "category": category, "sentiment": sentiment})
