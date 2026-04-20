"""AI-Based Customer Query Analyzer - Production-Grade Streamlit Application.

This application implements a multi-agent GenAI system using LangGraph and Google Gemini
for comprehensive customer query analysis with:
- Real LangGraph agent graph
- RAG-based response generation
- Streaming responses
- Cost tracking
- Evaluation harness
- Persistent memory
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.guardrails import validate_query, sanitize_query, redact_pii
from graph.runner import run_graph
from observability.costs import get_session_cost, format_cost, reset_session_cost
from evals.runner import run_evals, load_report, save_report

import concurrent.futures


def run_async_in_thread(coro):
    """Run an async coroutine safely from a sync Streamlit context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["LANGCHAIN_TRACING_V2"] = "false"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(
    page_title="AI Customer Query Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f1117;
    }
    
    .stSidebar {
        background-color: #1a1d2e;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d2e;
        padding: 10px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd6 0%, #6a4190 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1a1d2e;
        color: #e0e0e0;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    div[data-testid="stExpander"] {
        background-color: #1a1d2e;
        border-radius: 8px;
        border: 1px solid #2d3561;
    }
    
    .stProgress > div > div > div {
        background-color: #667eea;
    }
    
    .stMetric {
        background-color: #1a1d2e;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #2d3561;
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    div.js-plotly-plot {
        background-color: #1a1d2e !important;
    }
</style>
""", unsafe_allow_html=True)

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

PRIORITY_COLORS = {
    "Critical": "#e53935",
    "High": "#fb8c00",
    "Medium": "#fdd835",
    "Low": "#43a047"
}


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "session_cost" not in st.session_state:
        st.session_state.session_cost = 0.0
    if "response_language" not in st.session_state:
        st.session_state.response_language = "English"
    if "last_eval_report" not in st.session_state:
        st.session_state.last_eval_report = None
    logger.info("Session state initialized")


def get_priority_color(priority: str) -> str:
    """Get color for priority level."""
    return PRIORITY_COLORS.get(priority, "#a0aec0")


def get_priority_emoji(priority: str) -> str:
    """Get emoji for priority level."""
    emojis = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    return emojis.get(priority, "⚪")


async def run_analysis_streaming(query: str, preferred_language: str = "English") -> Dict[str, Any]:
    """Run analysis with streaming response.
    
    Args:
        query: The customer query to analyze.
    
    Returns:
        Dictionary containing all analysis results.
    """
    start_time_ms = int(time.time() * 1000)
    
    result = await run_graph(
        query,
        thread_id=st.session_state.thread_id,
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
        "reasoning_trace": result.get("reasoning_trace", [])
    }


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


def render_priority_badge(priority: str) -> None:
    """Render a priority badge with color coding."""
    color = get_priority_color(priority)
    emoji = get_priority_emoji(priority)
    st.markdown(
        f'<span style="background-color: {color}22; color: {color}; '
        f'border: 1px solid {color}; padding: 4px 12px; border-radius: 20px; '
        f'font-size: 0.85rem; font-weight: 600;">{emoji} {priority}</span>',
        unsafe_allow_html=True
    )


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


def render_response_card(response: str, sentiment: str) -> None:
    """Render the AI response card."""
    color = SENTIMENT_COLORS.get(sentiment, "#667eea")
    st.markdown(
        f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
        f'border-radius: 8px; padding: 20px; margin: 16px 0;">'
        f'<div style="color: #667eea; font-size: 0.85rem; font-weight: 600; margin-bottom: 12px;">'
        f'🤖 AI SUPPORT AGENT</div>'
        f'<div style="color: #e0e0e0; font-size: 1rem; line-height: 1.6;">{response}</div></div>',
        unsafe_allow_html=True
    )


def render_reasoning_expander(reasoning_trace: List[str]) -> None:
    """Render the reasoning trace in an expander."""
    if reasoning_trace:
        with st.expander("🧠 Reasoning Trace"):
            for i, reason in enumerate(reasoning_trace):
                st.markdown(f"**Step {i+1}:** {reason}")


def render_sidebar() -> None:
    """Render the sidebar with help, settings, and cost display."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: center; padding: 10px 0;">'
            '<h2 style="font-size: 1.8rem;">🎯 Query Analyzer</h2></div>',
            unsafe_allow_html=True
        )
        
        st.markdown("### 📋 Query Categories")
        for cat, info in CATEGORY_INFO.items():
            st.markdown(
                f'<span style="color: {info["color"]};">●</span> {info["emoji"]} {cat}',
                unsafe_allow_html=True
            )
        
        st.markdown("")
        st.markdown("### ⚙️ Settings")
        
        response_lang = st.selectbox(
            "Response Language",
            ["English", "Hindi", "Spanish", "French", "same as query"],
            index=["English", "Hindi", "Spanish", "French", "same as query"].index(
                st.session_state.response_language
            ) if st.session_state.response_language in ["English", "Hindi", "Spanish", "French", "same as query"] else 0
        )
        st.session_state.response_language = response_lang
        
        if st.button("🆕 New Thread", width='stretch'):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            reset_session_cost()
            st.session_state.session_cost = 0.0
            st.rerun()
        
        st.markdown("<hr style='border-color: #2d3561; margin-top: 20px;'>", unsafe_allow_html=True)
        
        st.markdown("### 💰 Session Cost")
        cost = get_session_cost()
        st.session_state.session_cost = cost
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-radius: 8px; padding: 16px; '
            f'text-align: center;">'
            f'<div style="color: #81c784; font-size: 1.5rem; font-weight: 700;">'
            f'{format_cost(cost)}</div></div>',
            unsafe_allow_html=True
        )
        
        tracing_status = "LangSmith ✓" if os.getenv("LANGCHAIN_API_KEY") else "local"
        st.markdown(f"**Tracing:** {tracing_status}")
        
        if st.button("🗑️ Clear History", width='stretch'):
            st.session_state.history = []
            reset_session_cost()
            st.session_state.session_cost = 0.0
            st.rerun()
        
        if st.button("📥 Export as CSV", width='stretch'):
            csv_data = export_history_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='content'
            )
        
        st.markdown("<hr style='border-color: #2d3561; margin-top: 20px;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; color: #718096; font-size: 0.75rem;">'
            'Built with LangGraph · Gemini 2.5 · Streamlit</div>',
            unsafe_allow_html=True
        )


def export_history_csv() -> str:
    """Export conversation history as CSV."""
    if not st.session_state.history:
        return "query,category,sentiment,priority,response,time\n"
    
    lines = ["query,category,sentiment,priority,response,time"]
    for entry in st.session_state.history:
        query = entry["query"].replace('"', '""') if "query" in entry else ""
        response = entry.get("response", "").replace('"', '""')
        priority = entry.get("priority", "Medium")
        line = f'"{query}",{entry.get("category","")},{entry.get("sentiment","")},{priority},"{response}",{entry.get("time","")}'
        lines.append(line)
    
    return "\n".join(lines)


def render_analyzer_tab() -> None:
    """Render the query analyzer tab."""
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
                    run_analysis_streaming(query, st.session_state.response_language)
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
                
                st.markdown("<br>", unsafe_allow_html=True)
                render_confidence_bars(
                    result["confidence_category"],
                    result["confidence_sentiment"]
                )
                
                render_reasoning_expander(result.get("reasoning_trace", []))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 💬 AI Response")
                render_response_card(result["response"], result["sentiment"])
                
                st.markdown(
                    f'<div style="text-align: right; color: #718096; font-size: 0.8rem;">'
                    f'⏱️ Processing time: {result["processing_time_ms"]} ms</div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                traceback.print_exc()
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


def render_evals_tab() -> None:
    """Render the evals dashboard tab."""
    st.markdown("### 📋 Evaluation Results")
    
    report = load_report("evals/last_report.json")
    
    if report:
        st.session_state.last_eval_report = report
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Category Accuracy", f"{report.category_accuracy:.1%}")
        with col2:
            st.metric("Sentiment Accuracy", f"{report.sentiment_accuracy:.1%}")
        with col3:
            st.metric("Priority Accuracy", f"{report.priority_accuracy:.1%}")
        with col4:
            st.metric("Avg Latency", f"{report.avg_latency_ms:.0f}ms")
        
        st.markdown("---")
        
        st.markdown("### 📊 Per-Case Results")
        
        case_data = []
        for result in report.case_results:
            case_data.append({
                "Query": result.query[:40] + "...",
                "Category": f"{'✓' if result.category_match else '✗'} {result.actual_category}",
                "Sentiment": f"{'✓' if result.sentiment_match else '✗'} {result.actual_sentiment}",
                "Priority": f"{'✓' if result.priority_match else '✗'} {result.actual_priority}",
                "Escalate": f"{'✓' if result.escalate_match else '✗'} {result.actual_escalate}"
            })
        
        if case_data:
            df = pd.DataFrame(case_data)
            st.dataframe(df, use_container_width=True)
        
        st.markdown(f"*Evaluated at: {report.timestamp}*")
    else:
        st.info("No evaluation results yet. Run evals to see metrics.")
    
    st.markdown("---")
    
    if st.button("▶ Run Evals Now", type="primary"):
        with st.spinner("Running evaluation on golden dataset..."):
            report = asyncio.run(run_evals(verbose=True))
            save_report(report)
            st.session_state.last_eval_report = report
            st.rerun()


def render_analytics_dashboard() -> None:
    """Render the analytics dashboard."""
    if not st.session_state.history:
        st.info("No analytics data yet. Analyze some queries to see statistics.")
        return
    
    total_queries = len(st.session_state.history)
    
    category_counts = {}
    sentiment_counts = {}
    priority_counts = {}
    
    for entry in st.session_state.history:
        cat = entry.get("category", "Unknown")
        sentiment_counts[cat] = category_counts.get(cat, 0) + 1
        
        sent = entry.get("sentiment", "Unknown")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        
        prio = entry.get("priority", "Unknown")
        priority_counts[prio] = priority_counts.get(prio, 0) + 1
    
    st.markdown("### 📊 Session Analytics")
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Total Queries", total_queries)
    with metric_cols[1]:
        avg_time = sum(int(e.get("time", "0 ms").replace(" ms", "")) for e in st.session_state.history) / total_queries if total_queries else 0
        st.metric("Avg Response Time", f"{avg_time:.0f} ms")
    with metric_cols[2]:
        escalation_rate = sum(1 for e in st.session_state.history if e.get("should_escalate", False)) / total_queries * 100 if total_queries else 0
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📂 Category Breakdown")
        if category_counts:
            df_cat = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
            fig_cat = px.bar(
                df_cat, x="Category", y="Count", color="Category",
                color_discrete_map={cat: CATEGORY_INFO.get(cat, {}).get("color", "#667eea") for cat in df_cat["Category"]}
            )
            fig_cat.update_layout(showlegend=False, paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig_cat, width='stretch')
    
    with col2:
        st.markdown("#### 💬 Sentiment Distribution")
        if sentiment_counts:
            df_sent = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
            fig_sent = px.pie(df_sent, names="Sentiment", values="Count", color="Sentiment", color_discrete_map=SENTIMENT_COLORS)
            fig_sent.update_layout(paper_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig_sent, width='stretch')


def render_main_content() -> None:
    """Render the main content area with tabs."""
    st.markdown(
        '<h1 style="text-align: center; font-size: 2.4rem; font-weight: 800; '
        'background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f64f59 100%); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
        'background-clip: text; margin-bottom: 0.5rem;">AI Customer Query Analyzer</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #a0aec0; margin-bottom: 30px;">'
        'Powered by LangGraph · Google Gemini 2.5 · Multi-Agent AI</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr style="border-color: #2d3561;">', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Analyzer", "📊 Analytics", "📋 Evals"])
    
    with tab1:
        render_analyzer_tab()
    
    with tab2:
        render_analytics_dashboard()
    
    with tab3:
        render_evals_tab()


def main() -> None:
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()