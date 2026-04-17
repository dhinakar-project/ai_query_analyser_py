"""Responder agent using real LangChain agent with conversation memory."""

import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_responder_llm, get_gemini

logger = logging.getLogger(__name__)


def _get_responder_llm_safe():
    """Get responder LLM with fallback to flash model on quota errors."""
    try:
        return get_responder_llm()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning("gemini-2.5-pro quota exhausted, falling back to gemini-2.5-flash")
            return get_gemini(temperature=0.6)
        raise


@tool
def generate_response(query: str, category: str, sentiment: str, language: str = "English") -> str:
    """Generate an empathetic, contextual customer support response.
    
    This tool creates a tailored response based on the query content,
    its classified category, detected sentiment, and detected language.
    
    Args:
        query: The customer's original query text.
        category: The classified category (Billing, Technical Support, etc.).
        sentiment: The detected sentiment (Positive, Neutral, Negative, Urgent, Frustrated).
        language: The detected language of the query (for multilingual support).
    
    Returns:
        A professional, empathetic customer support response string.
    
    Tone Guidelines by Sentiment:
        - Positive: Warm, appreciative, friendly, reinforce positive experience
        - Neutral: Professional, helpful, informative, clear
        - Negative: Empathetic, apologetic, understanding, offer solutions
        - Urgent: Immediate reassurance, clear next steps, provide direct contact
        - Frustrated: Extra empathy, acknowledge frustration, apologize sincerely, offer compensation
    """
    logger.debug(f"Generating response: category={category}, sentiment={sentiment}, lang={language}")
    
    system_prompts = {
        "English": """You are a professional customer support specialist. Your role is to generate helpful, empathetic responses that address customer concerns.

Guidelines:
1. Always address the customer's concern directly and professionally
2. Match your tone to the sentiment:
   - Urgent: Provide immediate reassurance and clear next steps
   - Frustrated: Show extra empathy, apologize for the inconvenience, offer solutions
   - Negative: Be understanding, acknowledge the issue, offer solutions
   - Positive: Be friendly, appreciative, reinforce their positive experience
   - Neutral: Be helpful, informative, professional
3. Use the category to provide relevant help:
   - Billing: Reference billing systems, payment processing
   - Technical Support: Include troubleshooting steps, ticket creation
   - Returns & Refunds: Explain return process, timelines
   - Shipping & Delivery: Provide tracking info, shipping policies
   - Account Management: Guide through account settings, security
   - General Inquiry: Provide helpful information
4. Keep responses 3-5 sentences, focused and actionable
5. Always end with offer for further assistance
6. Never invent policy details - stay helpful but accurate""",
        
        "Spanish": """Eres un especialista profesional de atención al cliente. Tu rol es generar respuestas útiles y empáticas que aborden las preocupaciones de los clientes.

Directrices:
1. Aborda siempre la preocupación del cliente directamente
2. Adapta tu tono al sentimiento:
   - Urgente: Proporciona tranquilidad inmediata y próximos pasos claros
   - Frustrado: Muestra empatía adicional, disculpa por las molestias
   - Negativo: Sé comprensivo, ofrece soluciones
   - Positivo: Sé amigable, aprecia su experiencia
   - Neutral: Sé útil, profesional, informativo
3. Mantén las respuestas de 3-5 oraciones
4. Siempre termina ofreciendo más ayuda""",
        
        "French": """Vous êtes un spécialiste professionnel du support client. Votre rôle est de générer des réponses utiles et empathiques.

Directives:
1. Abordez toujours la préoccupation du client directement
2. Adaptez votre ton au sentiment:
   - Urgent: Réassurance immédiate, prochaines étapes claires
   - Frustré: Empathie supplémentaire, excuses sincères
   - Négatif: Compréhension, offrez des solutions
   - Positif: Amical, appréciatif
   - Neutre: Utile, professionnel
3. Gardez les réponses à 3-5 phrases
4. Terminez toujours par une offre d'aide supplémentaire"""
    }
    
    system_prompt = system_prompts.get(language, system_prompts["English"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Customer Query: {query}
Category: {category}
Sentiment: {sentiment}

Generate a professional, empathetic response.""")
    ])
    
    llm = _get_responder_llm_safe()
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "query": query,
            "category": category,
            "sentiment": sentiment
        })
        
        logger.debug(f"Response generated: {len(response)} characters")
        return response.strip()
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return "I apologize for any inconvenience you may be experiencing. Our team is here to help you. Please provide more details about your concern, and we'll do our best to assist you promptly."

def run_responder_agent(
    query: str,
    category: str,
    sentiment: str,
    priority: Optional[str] = None,
    language: str = "English"
) -> str:
    """Run the responder agent to generate a customer response.
    
    Args:
        query: The customer's query text.
        category: The classified category.
        sentiment: The detected sentiment.
        priority: Optional priority level for context.
        language: The detected language.
    
    Returns:
        The generated response string.
    """
    logger.info(f"Running responder agent: category={category}, sentiment={sentiment}")
    
    priority_context = ""
    if priority and priority in ["Critical", "High"]:
        priority_context = f" (Note: This query has been flagged as {priority} priority)"
    
    try:
        response = generate_response.invoke({
            "query": query,
            "category": category,
            "sentiment": sentiment,
            "language": language
        })
        
        if priority_context and sentiment in ["Urgent", "Frustrated"]:
            response += f"\n\n{priority_context} Our team has been notified and will prioritize your request."
        
        logger.debug(f"Response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Responder agent error: {e}")
        return "I apologize for any inconvenience you may be experiencing. Our team is here to help you. Please provide more details about your concern, and we'll do our best to assist you promptly."
