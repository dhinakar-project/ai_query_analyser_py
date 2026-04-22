import asyncio
import concurrent.futures
import re
import time
import uuid
from typing import Dict, Any, List

import streamlit as st

from graph.runner import run_graph
from utils.guardrails import validate_query, sanitize_query, redact_pii
from observability.costs import get_session_cost


def run_async_in_thread(coro):
    """Run an async coroutine safely from a sync Streamlit context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


async def run_analysis_streaming(query: str, thread_id: str, preferred_language: str = "English") -> Dict[str, Any]:
    """Run analysis with streaming response."""
    start_time_ms = int(time.time() * 1000)
    
    result = await run_graph(
        query,
        thread_id=thread_id,
        preferred_language=preferred_language
    )
    
    end_time_ms = int(time.time() * 1000)
    processing_time_ms = end_time_ms - start_time_ms
    
    return {
        "category": result.get("category", "General Inquiry"),
        "sentiment": result.get("sentiment", "Neutral"),
        "priority": result.get("priority", "Medium"),
        "confidence_category": result.get("confidence_category", 50),
        "confidence_sentiment": result.get("confidence_sentiment", 50),
        "should_escalate": result.get("should_escalate", False),
        "escalation_reason": result.get("escalation_reason"),
        "suggested_team": result.get("suggested_team"),
        "language": result.get("language", "English"),
        "response": result.get("response", ""),
        "processing_time_ms": processing_time_ms,
        "reasoning_trace": result.get("reasoning_trace", []),
        "rag_sources": result.get("rag_sources", [])
    }


def get_priority_color(priority: str) -> str:
    """Get color for priority level."""
    PRIORITY_COLORS = {
        "Critical": "#e53935",
        "High": "#fb8c00",
        "Medium": "#fdd835",
        "Low": "#43a047"
    }
    return PRIORITY_COLORS.get(priority, "#a0aec0")


def get_priority_emoji(priority: str) -> str:
    """Get emoji for priority level."""
    emojis = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    return emojis.get(priority, "⚪")


def get_confidence_bar(confidence: int, label: str) -> None:
    """Render a confidence bar."""
    st.markdown(f"**{label}**")
    st.progress(confidence / 100, text=f"{confidence}%")


def format_reasoning_trace(reasoning_trace: List[str]) -> None:
    """Render the reasoning trace as a visual timeline."""
    if reasoning_trace:
        with st.expander("🧠 Reasoning Trace"):
            for i, reason in enumerate(reasoning_trace):
                icon = "🔵" if "combined_analysis" in reason else "🟢" if "responder" in reason else "⚡"
                time_match = re.search(r'latency=(\d+)ms', reason)
                latency = f" · {time_match.group(1)}ms" if time_match else ""
                st.markdown(f"{icon} **Step {i+1}**{latency}")
                st.caption(reason)
                if i < len(reasoning_trace) - 1:
                    st.markdown('<br>', unsafe_allow_html=True)


def record_analytics(result: Dict[str, Any]) -> None:
    """Record query results in session state."""
    st.session_state.history.insert(0, {
        "query": result.get("query", ""),
        "category": result.get("category"),
        "sentiment": result.get("sentiment"),
        "priority": result.get("priority"),
        "confidence_category": result.get("confidence_category"),
        "confidence_sentiment": result.get("confidence_sentiment"),
        "should_escalate": result.get("should_escalate"),
        "escalation_reason": result.get("escalation_reason"),
        "suggested_team": result.get("suggested_team"),
        "language": result.get("language"),
        "response": result.get("response"),
        "time": f"{result.get('processing_time_ms', 0)} ms"
    })


def render_confidence_bars(confidence_cat: int, confidence_sent: int) -> None:
    """Render confidence score progress bars."""
    if confidence_cat < 40:
        st.warning("⚠️ Low confidence classification", icon="⚠️")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category Confidence**")
        st.progress(confidence_cat / 100, text=f"{confidence_cat}%")
    with col2:
        st.markdown("**Sentiment Confidence**")
        st.progress(confidence_sent / 100, text=f"{confidence_sent}%")


def render_escalation_alert(should_escalate: bool, reason: str, team: str) -> None:
    """Render escalation alert if needed."""
    if should_escalate:
        st.error(
            f"🚨 **This query requires human escalation**\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Suggested Team:** {team}",
            icon="🚨"
        )


SENTIMENT_COLORS = {
    "Positive": "#81c784",
    "Neutral": "#4fc3f7",
    "Negative": "#f06292",
    "Urgent": "#ff8a65",
    "Frustrated": "#ffb74d"
}

CATEGORY_INFO = {
    "Billing": {"color": "#ffb74d", "emoji": "💰"},
    "Technical Support": {"color": "#4fc3f7", "emoji": "🔧"},
    "Returns & Refunds": {"color": "#f06292", "emoji": "📦"},
    "Shipping & Delivery": {"color": "#81c784", "emoji": "🚚"},
    "Account Management": {"color": "#ce93d8", "emoji": "👤"},
    "General Inquiry": {"color": "#a0aec0", "emoji": "💬"}
}


def render_result_cards(result: Dict[str, Any]) -> None:
    """Render the analysis result cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = result["category"]
        color = CATEGORY_INFO.get(category, {}).get("color", "#a0aec0")
        emoji = CATEGORY_INFO.get(category, {}).get("emoji", "💬")
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
            f'border-radius: 8px; padding: 16px; text-align: center;">'
            f'<div style="color: #a0aec0; font-size: 0.85rem;">📂 Category</div>'
            f'<div style="color: {color}; font-size: 1.2rem; font-weight: 600;">'
            f'{emoji} {category}</div></div>',
            unsafe_allow_html=True
        )
        
        conf = result.get("confidence_category", 0)
        st.caption(f"Confidence: {conf}%")
        st.progress(conf / 100)
    
    with col2:
        sentiment = result["sentiment"]
        color = SENTIMENT_COLORS.get(sentiment, "#a0aec0")
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
            f'border-radius: 8px; padding: 16px; text-align: center;">'
            f'<div style="color: #a0aec0; font-size: 0.85rem;">💬 Sentiment</div>'
            f'<div style="color: {color}; font-size: 1.2rem; font-weight: 600;">{sentiment}</div></div>',
            unsafe_allow_html=True
        )
        
        sent_conf = result.get("confidence_sentiment", 0)
        st.caption(f"Confidence: {sent_conf}%")
        st.progress(sent_conf / 100)
    
    with col3:
        priority = result["priority"]
        color = get_priority_color(priority)
        emoji = get_priority_emoji(priority)
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
            f'border-radius: 8px; padding: 16px; text-align: center;">'
            f'<div style="color: #a0aec0; font-size: 0.85rem;">⚡ Priority</div>'
            f'<div style="color: {color}; font-size: 1.2rem; font-weight: 600;">'
            f'{emoji} {priority}</div></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        lang = result["language"]
        lang_code = lang[:2].lower() if lang else "en"
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-left: 4px solid #667eea; '
            f'border-radius: 8px; padding: 16px; text-align: center;">'
            f'<div style="color: #a0aec0; font-size: 0.85rem;">🌐 Language</div>'
            f'<div style="color: #667eea; font-size: 1.2rem; font-weight: 600;">'
            f'{lang} ({lang_code})</div></div>',
            unsafe_allow_html=True
        )


def render_response_card(response: str, sentiment: str, streaming: bool = True) -> None:
    """Render the AI response card with optional streaming."""
    color = SENTIMENT_COLORS.get(sentiment, "#667eea")
    st.markdown(
        f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
        f'border-radius: 8px; padding: 20px; margin: 16px 0;">'
        f'<div style="color: #667eea; font-size: 0.85rem; font-weight: 600; margin-bottom: 12px;">'
        f'🤖 AI SUPPORT AGENT</div>',
        unsafe_allow_html=True
    )
    
    if streaming and response:
        with st.empty():
            displayed = ""
            for word in response.split():
                displayed += word + " "
                st.markdown(displayed + "▌")
                time.sleep(0.03)
            st.markdown(displayed)
    else:
        st.markdown(
            f'<div style="color: #e0e0e0; font-size: 1rem; line-height: 1.6;">{response}</div></div>',
            unsafe_allow_html=True
        )


def render_analyzer_tab() -> None:
    """Render the query analyzer tab."""
    with st.expander("ℹ️ How this AI system works", expanded=False):
        st.markdown("""
        **Pipeline (runs on every query):**
        1. **Guardrails** — validates input, detects PII and prompt injection
        2. **Combined Analysis Node** — single Gemini call classifies category,
           detects sentiment, assesses priority, decides escalation
        3. **RAG Retrieval** — ChromaDB finds top-3 relevant support articles
        4. **Responder Node** — Gemini generates empathetic response using RAG context
        5. **Observability** — logs cost, latency, reasoning trace to SQLite

        **Total: 2 LLM calls per query** (down from 6 before optimisation)
        """)
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "response_language" not in st.session_state:
        st.session_state.response_language = "English"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_cost" not in st.session_state:
        st.session_state.session_cost = 0.0
    
    with st.expander("🌐 Try multilingual examples"):
        cols = st.columns(4)
        examples = [
            ("🇮🇳 Hindi", "मेरा बिल गलत है। इस महीने मुझसे दो बार चार्ज किया गया।"),
            ("🇪🇸 Spanish", "Mi pedido llegó dañado. Necesito un reembolso urgente."),
            ("🇫🇷 French", "Mon compte a été bloqué. Je ne peux pas me connecter depuis 3 jours."),
            ("🇩🇪 German", "Das Paket ist nicht angekommen. Wo ist meine Bestellung?"),
        ]
        for i, (lang, text) in enumerate(examples):
            with cols[i]:
                st.caption(lang)
                if st.button(text[:35] + "...", key=f"eg_{i}"):
                    st.session_state.response_language = "same as query"
                    st.session_state.query_input = text
                    st.rerun()
    
    query = st.text_area(
        "Enter your customer query:",
        placeholder="e.g., My bill seems incorrect this month. I've been charged twice for the same item.",
        height=120,
        key="query_input"
    )
    
    col_btn = st.columns([1])
    with col_btn[0]:
        submit = st.button("🚀 Analyze Query", width='stretch')
    
    if submit:
        if not query or not query.strip():
            st.warning("⚠️ Please enter a query to analyze.")
            return
        
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            st.error(f"⚠️ {error_msg}")
            return
        
        redacted_query, redactions = redact_pii(query)
        if redactions:
            st.info("ℹ️ Personal information was automatically redacted before processing.", icon="ℹ️")
            query = redacted_query
        
        query = sanitize_query(query)
        
        with st.spinner("🔄 Analyzing your query with AI agents..."):
            try:
                result = run_async_in_thread(
                    run_analysis_streaming(query, st.session_state.thread_id, st.session_state.response_language)
                )
                
                entry = {
                    "query": query,
                    "category": result["category"],
                    "sentiment": result["sentiment"],
                    "priority": result["priority"],
                    "confidence_category": result["confidence_category"],
                    "confidence_sentiment": result["confidence_sentiment"],
                    "should_escalate": result["should_escalate"],
                    "escalation_reason": result["escalation_reason"],
                    "suggested_team": result["suggested_team"],
                    "language": result["language"],
                    "response": result["response"],
                    "time": f"{result['processing_time_ms']} ms"
                }
                
                record_analytics(entry)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                render_escalation_alert(
                    result["should_escalate"],
                    result["escalation_reason"],
                    result["suggested_team"]
                )
                
                st.markdown("### 📊 Analysis Results")
                render_result_cards(result)
                
                if result.get("rag_sources"):
                    with st.expander("📚 Knowledge Base Sources Used", expanded=False):
                        for i, title in enumerate(result["rag_sources"], 1):
                            st.markdown(f"**{i}.** {title}")
                
                format_reasoning_trace(result.get("reasoning_trace", []))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 💬 AI Response")
                render_response_card(result["response"], result["sentiment"])
                
                st.markdown(
                    f'<div style="text-align: right; color: #718096; font-size: 0.8rem;">'
                    f'⏱️ Processing time: {result["processing_time_ms"]} ms</div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"🚫 An error occurred: {str(e)}")
    
    if st.session_state.history:
        st.markdown("<hr style='border-color: #2d3561; margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("### 📜 Conversation History")
        
        for entry in st.session_state.history[:5]:
            with st.expander(
                f"**You:** {entry.get('query','')[:80]}{'...' if len(entry.get('query','')) > 80 else ''}",
                expanded=False
            ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    cat = entry.get("category", "")
                    color = CATEGORY_INFO.get(cat, {}).get("color", "#a0aec0")
                    emoji = CATEGORY_INFO.get(cat, {}).get("emoji", "💬")
                    st.markdown(
                        f'<span style="color: {color};">{emoji} {cat}</span>',
                        unsafe_allow_html=True
                    )
                with col2:
                    sent = entry.get("sentiment", "")
                    sent_color = SENTIMENT_COLORS.get(sent, "#a0aec0")
                    st.markdown(
                        f'<span style="color: {sent_color};">💬 {sent}</span>',
                        unsafe_allow_html=True
                    )
                
                priority = entry.get("priority", "Medium")
                st.markdown(f"**Priority:** {get_priority_emoji(priority)} {priority}")
                
                st.markdown("---")
                st.markdown(entry.get("response", ""))
                
                st.markdown(
                    f'<div style="text-align: right; color: #718096; font-size: 0.75rem;">'
                    f'⏱️ {entry.get("time","")}</div>',
                    unsafe_allow_html=True
                )
