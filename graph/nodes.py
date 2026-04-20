"""Node functions for LangGraph query analysis pipeline."""

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage

from models.analysis_result import (
    CategoryClassification,
    SentimentAnalysis,
    PriorityAssessment,
    EscalationDecision,
    LanguageDetection,
)
from utils.llm import (
    get_classifier_llm,
    get_sentiment_llm,
    get_priority_llm,
    get_escalation_llm,
    get_responder_llm,
)
from observability.logger import log_llm_call
from graph.state import QueryState

logger = logging.getLogger(__name__)

CATEGORY_SYSTEM_PROMPT = """You are an expert customer query classifier. Your task is to categorize customer queries into one of six categories.

CATEGORIES:
- Billing: Payment issues, charges, invoices, pricing, billing disputes, refunds, subscriptions
- Technical Support: Software/hardware problems, login errors, bugs, crashes, app issues
- Returns & Refunds: Product returns, money back, exchanges, damaged items, warranty claims
- Shipping & Delivery: Order tracking, delivery delays, lost packages, shipping info
- Account Management: Login, password, profile updates, subscription changes, account security
- General Inquiry: Questions about products, services, company info, how-to questions

Think step by step before outputting your answer. First reason through the evidence, then commit to the most defensible classification.

RATE YOUR CONFIDENCE 0-100. Be conservative: only give 90+ if the answer is unambiguous. If two categories are nearly tied, report the lower confidence.

FEW-SHOT EXAMPLES:

Example 1 (ambiguous billing+technical):
Query: "I tried to pay my bill online but the payment keeps failing and the website shows an error."
Category: Billing
Reasoning: While the issue involves a website error, the core problem is a payment/billing failure. The technical issue is secondary to the billing dispute.
Confidence: 75

Example 2 (passive-aggressive that looks neutral):
Query: "Oh wonderful, another month where I have to remind you about the same issue. Love waiting on hold."
Category: Billing
Reasoning: Despite the sarcastic tone, the mention of "another month" and "remind you" suggests an ongoing billing issue that requires attention.
Confidence: 60

Example 3 (multilingual):
Query: "Hola, necesito ayuda con mi cuenta de usuario. No puedo iniciar sesion."
Category: Account Management
Reasoning: The Spanish query clearly mentions account/login issues. Language doesn't affect the classification.
Confidence: 95

Respond with the category and your confidence level."""


SENTIMENT_SYSTEM_PROMPT = """You are an expert sentiment analyst for customer support. Your task is to detect the emotional tone of customer queries.

SENTIMENTS:
- Positive: Satisfied, happy, grateful, complimenting, thanking
- Neutral: Normal question, no strong emotion, factual inquiry
- Negative: Unhappy, disappointed, dissatisfied
- Urgent: Needs immediate attention, time-sensitive, emergency
- Frustrated: Annoyed, repeated issues, losing patience, sarcasm

Think step by step before outputting your answer. First reason through the evidence, then commit to the most defensible sentiment.

RATE YOUR CONFIDENCE 0-100. Be conservative: only give 90+ if the answer is unambiguous. If two sentiments are nearly tied, report the lower confidence.

FEW-SHOT EXAMPLES:

Example 1 (passive-aggressive that looks neutral):
Query: "Oh wonderful, another month where I have to remind you about the same issue. Love waiting on hold."
Sentiment: Frustrated
Reasoning: The sarcasm ("Oh wonderful", "Love waiting") and mention of repeated issues ("another month") clearly indicate frustration despite no direct negative words.
Confidence: 85

Example 2 (urgent disguised as neutral):
Query: "I need to change my shipping address before tomorrow morning. Is that possible?"
Sentiment: Urgent
Reasoning: The explicit deadline ("before tomorrow morning") and question about possibility indicates urgency.
Confidence: 80

Example 3 (negative but polite):
Query: "I was hoping for a different outcome but I understand. Could you explain the options?"
Sentiment: Negative
Reasoning: "Hoping for a different outcome" indicates disappointment, though the tone remains polite.
Confidence: 70

Respond with the sentiment and your confidence level."""


PRIORITY_SYSTEM_PROMPT = """You are a priority assessment expert for customer support. Your task is to determine the urgency level of customer queries.

PRIORITIES:
- Critical: Immediate human attention needed, legal/safety issues, potential account compromise
- High: Urgent sentiment with serious issues, frustrated with repeated problems
- Medium: Standard issues with negative sentiment, regular complaints
- Low: Positive/neutral sentiment, simple inquiries, general questions

Think step by step before outputting your answer. Consider both the content and sentiment together.

Respond with the priority level and brief reasoning."""


ESCALATION_SYSTEM_PROMPT = """You are an escalation decision expert. Your task is to determine if a customer query requires human agent intervention.

ESCALATION CRITERIA:
- Escalate: Critical priority, legal/safety issues, security concerns, repeated failures, extreme frustration
- Don't Escalate: Simple inquiries, positive sentiment, standard issues already handled

Think step by your answer. Provide a clear reason for your decision.

Respond with whether escalation is recommended and the suggested team if applicable."""


LANGUAGE_SYSTEM_PROMPT = """You are a language detection expert. Your task is to identify the language of customer queries.

Detect the language name and ISO 639-1 code. Rate your confidence 0-100.

Respond with language name, code, and confidence."""


RESPONDER_SYSTEM_PROMPT = """You are a professional AI customer support agent. Your task is to generate helpful, empathetic responses to customer queries.

CONTEXT:
- Category: {category}
- Sentiment: {sentiment}
- Priority: {priority}
- Language: {language}

INSTRUCTIONS:
1. Acknowledge the customer's concern with empathy
2. Provide a clear, actionable response
3. If relevant, reference the support knowledge provided
4. Keep the response concise but complete
5. Always maintain a professional, helpful tone

{preferred_language_instruction}

{rag_context}

Respond to the customer's query now."""


async def language_node(state: QueryState):
    """Detect the language of the query."""
    start_time = time.time()
    query = state["query"]
    
    llm = get_classifier_llm()
    structured_llm = llm.with_structured_output(LanguageDetection, method="json_schema")
    
    messages = [
        ("system", LANGUAGE_SYSTEM_PROMPT),
        ("human", f"Detect the language of this query:\n\n{query}")
    ]
    
    try:
        result = structured_llm.invoke(messages)
        log_llm_call(
            node_name="language_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Detected language: {result.language_name} ({result.language_code}) with confidence {result.confidence}%"
        
        return {
            "language": result.language_name,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {
            "language": "English",
            "reasoning_trace": state["reasoning_trace"] + ["Language detection failed, defaulting to English"]
        }


async def classify_node(state: QueryState):
    """Classify the query into a category."""
    start_time = time.time()
    query = state["query"]
    
    llm = get_classifier_llm()
    structured_llm = llm.with_structured_output(CategoryClassification, method="json_schema")
    
    messages = [
        ("system", CATEGORY_SYSTEM_PROMPT),
        ("human", f"Classify this customer query:\n\n{query}")
    ]
    
    try:
        result = structured_llm.invoke(messages)
        log_llm_call(
            node_name="classify_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Classified as {result.category} with confidence {result.confidence}%. Reasoning: {result.reasoning}"
        
        return {
            "category": result.category,
            "confidence_category": result.confidence,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "category": "General Inquiry",
            "confidence_category": 50,
            "reasoning_trace": state["reasoning_trace"] + ["Classification failed, defaulting to General Inquiry"]
        }


async def sentiment_node(state: QueryState):
    """Analyze the sentiment of the query."""
    start_time = time.time()
    query = state["query"]
    
    llm = get_sentiment_llm()
    structured_llm = llm.with_structured_output(SentimentAnalysis, method="json_schema")
    
    messages = [
        ("system", SENTIMENT_SYSTEM_PROMPT),
        ("human", f"Analyze the sentiment of this query:\n\n{query}")
    ]
    
    try:
        result = structured_llm.invoke(messages)
        log_llm_call(
            node_name="sentiment_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Detected sentiment: {result.sentiment} with confidence {result.confidence}%. Reasoning: {result.reasoning}"
        
        return {
            "sentiment": result.sentiment,
            "confidence_sentiment": result.confidence,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "sentiment": "Neutral",
            "confidence_sentiment": 50,
            "reasoning_trace": state["reasoning_trace"] + ["Sentiment analysis failed, defaulting to Neutral"]
        }


async def priority_node(state: QueryState):
    """Assess the priority of the query."""
    start_time = time.time()
    query = state["query"]
    category = state.get("category", "General Inquiry")
    sentiment = state.get("sentiment", "Neutral")
    
    llm = get_priority_llm()
    structured_llm = llm.with_structured_output(PriorityAssessment, method="json_schema")
    
    context = f"Query: {query}\nCategory: {category}\nSentiment: {sentiment}"
    
    messages = [
        ("system", PRIORITY_SYSTEM_PROMPT),
        ("human", f"Assess priority for:\n\n{context}")
    ]
    
    try:
        result = structured_llm.invoke(messages)
        log_llm_call(
            node_name="priority_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Priority assessed as {result.priority}. Reasoning: {result.reasoning}"
        
        return {
            "priority": result.priority,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Priority assessment failed: {e}")
        return {
            "priority": "Medium",
            "reasoning_trace": state["reasoning_trace"] + ["Priority assessment failed, defaulting to Medium"]
        }


async def escalation_node(state: QueryState):
    """Determine if the query should be escalated."""
    start_time = time.time()
    query = state["query"]
    category = state.get("category", "General Inquiry")
    sentiment = state.get("sentiment", "Neutral")
    priority = state.get("priority", "Medium")
    
    llm = get_escalation_llm()
    structured_llm = llm.with_structured_output(EscalationDecision, method="json_schema")
    
    context = f"Query: {query}\nCategory: {category}\nSentiment: {sentiment}\nPriority: {priority}"
    
    messages = [
        ("system", ESCALATION_SYSTEM_PROMPT),
        ("human", f"Determine escalation for:\n\n{context}")
    ]
    
    try:
        result = structured_llm.invoke(messages)
        log_llm_call(
            node_name="escalation_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Escalation: {result.should_escalate}. Reason: {result.reasoning}"
        
        return {
            "should_escalate": result.should_escalate,
            "escalation_reason": result.reasoning if result.should_escalate else None,
            "suggested_team": result.suggested_team,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Escalation decision failed: {e}")
        return {
            "should_escalate": False,
            "escalation_reason": None,
            "suggested_team": None,
            "reasoning_trace": state["reasoning_trace"] + ["Escalation decision failed, defaulting to no escalation"]
        }


async def responder_node(state: QueryState, preferred_language: str = "English"):
    """Generate a response to the customer query using RAG."""
    from rag.retriever import retrieve
    
    start_time = time.time()
    query = state["query"]
    category = state.get("category", "General Inquiry")
    sentiment = state.get("sentiment", "Neutral")
    priority = state.get("priority", "Medium")
    language = state.get("language", "English")
    
    articles = await retrieve(query, category, k=3)
    
    rag_context = ""
    if articles:
        rag_context = "Relevant support knowledge:\n"
        for article in articles:
            rag_context += f"- {article.title}: {article.content[:400]}\n"
    
    if preferred_language == "same as query":
        preferred_language_instruction = ""
    else:
        preferred_language_instruction = f"Always respond in {preferred_language}, regardless of the query language."
    
    llm = get_responder_llm()
    
    prompt = RESPONDER_SYSTEM_PROMPT.format(
        category=category,
        sentiment=sentiment,
        priority=priority,
        language=language,
        preferred_language_instruction=preferred_language_instruction,
        rag_context=rag_context
    )
    
    messages = [
        ("system", prompt),
        ("human", f"Customer query:\n\n{query}")
    ]
    
    try:
        result = llm.invoke(messages)
        log_llm_call(
            node_name="responder_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        reasoning = f"Generated response for category {category}"
        
        return {
            "response": result.content,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {
            "response": "I apologize, but I'm having trouble generating a response right now. Please try again.",
            "reasoning_trace": state["reasoning_trace"] + ["Response generation failed"]
        }


async def escalation_responder_node(state: QueryState):
    """Generate an escalation response when query should be handled by human."""
    start_time = time.time()
    query = state["query"]
    category = state.get("category", "General Inquiry")
    sentiment = state.get("sentiment", "Neutral")
    escalation_reason = state.get("escalation_reason", "Requires human attention")
    suggested_team = state.get("suggested_team", "Customer Support")
    
    llm = get_responder_llm()
    
    prompt = f"""You are generating a response to a customer query that will be escalated to a human agent.

CATEGORY: {category}
SENTIMENT: {sentiment}
ESCALATION REASON: {escalation_reason}
SUGGESTED TEAM: {suggested_team}

Generate a professional response that:
1. Acknowledges the issue
2. Explains that a human agent will be in touch
3. Thanks them for their patience

Keep it concise and empathetic."""

    messages = [
        ("system", prompt),
        ("human", f"Customer query:\n\n{query}")
    ]
    
    try:
        result = llm.invoke(messages)
        log_llm_call(
            node_name="escalation_responder_node",
            model=llm.model_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            query_hash=hash(query)
        )
        
        return {
            "response": result.content,
            "reasoning_trace": state["reasoning_trace"] + ["Generated escalation response"]
        }
    except Exception as e:
        logger.error(f"Escalation response failed: {e}")
        return {
            "response": f"Your query has been escalated to our {suggested_team}. A representative will contact you shortly.",
            "reasoning_trace": state["reasoning_trace"] + ["Escalation response failed, using fallback"]
        }


async def retry_classify_node(state: QueryState):
    """Retry classification with incremented retry count when confidence is low."""
    retry_count = state.get("retry_count", 0)
    
    query = state["query"]
    current_category = state.get("category", "General Inquiry")
    current_confidence = state.get("confidence_category", 0)
    
    llm = get_classifier_llm()
    structured_llm = llm.with_structured_output(CategoryClassification, method="json_schema")
    
    retry_prompt = f"""This is retry #{retry_count + 1}. Previous classification was '{current_category}' with confidence {current_confidence}.

Your task is to reclassify this customer query. Think more carefully about the evidence.

{state['query']}

Classify into one of: Billing, Technical Support, Returns & Refunds, Shipping & Delivery, Account Management, General Inquiry."""

    messages = [
        ("system", CATEGORY_SYSTEM_PROMPT),
        ("human", retry_prompt)
    ]
    
    try:
        result = structured_llm.invoke(messages)
        
        reasoning = f"Retry #{retry_count + 1}: Reclassified as {result.category} with confidence {result.confidence}%. Reasoning: {result.reasoning}"
        
        return {
            "category": result.category,
            "confidence_category": result.confidence,
            "retry_count": retry_count + 1,
            "reasoning_trace": state["reasoning_trace"] + [reasoning]
        }
    except Exception as e:
        logger.error(f"Retry classification failed: {e}")
        new_retry_count = retry_count + 1
        return {
            "retry_count": new_retry_count,
            "reasoning_trace": state["reasoning_trace"] + [f"Retry #{new_retry_count} failed"]
        }