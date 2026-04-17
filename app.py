"""AI-Based Customer Query Analyzer - Production-Grade Streamlit Application.

This application implements a multi-agent GenAI system using LangChain and Google Gemini
for comprehensive customer query analysis with:
- Real LangChain ReAct agents with tool calling
- Parallel async execution
- Conversation memory
- Confidence scoring
- Priority and escalation logic
- Analytics dashboard
"""

import asyncio
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agents import (
    run_responder_agent,
    get_suggested_team_for_category
)
from utils.guardrails import validate_query, sanitize_query
from utils.analytics import QueryAnalytics, get_default_analytics
from utils.combined_analyzer import run_combined_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if "analytics" not in st.session_state:
        st.session_state.analytics = get_default_analytics()
    logger.info("Session state initialized")


def get_priority_color(priority: str) -> str:
    """Get color for priority level."""
    return PRIORITY_COLORS.get(priority, "#a0aec0")


def get_priority_emoji(priority: str) -> str:
    """Get emoji for priority level."""
    emojis = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    return emojis.get(priority, "⚪")


async def run_analysis(query: str) -> Dict[str, Any]:
    """Run analysis of a customer query using combined analyzer.
    
    Makes 2 LLM calls:
    1. Combined analysis (category, sentiment, priority, escalation, language)
    2. Response generation
    
    Args:
        query: The customer query to analyze.
    
    Returns:
        Dictionary containing all analysis results.
    """
    start_time_ms = int(time.time() * 1000)
    
    analysis_result = await asyncio.to_thread(run_combined_analysis, query)
    
    category = analysis_result["category"]
    sentiment = analysis_result["sentiment"]
    priority = analysis_result["priority"]
    confidence_category = analysis_result["category_confidence"]
    confidence_sentiment = analysis_result["sentiment_confidence"]
    should_escalate = analysis_result["should_escalate"]
    escalation_reason = analysis_result["escalation_reason"]
    suggested_team = analysis_result["suggested_team"]
    language = analysis_result["language"]
    
    if not suggested_team:
        suggested_team = get_suggested_team_for_category(category)
    
    language_code = language[:2].lower() if language else "en"
    
    response = await asyncio.to_thread(
        run_responder_agent,
        query,
        category,
        sentiment,
        priority,
        language
    )
    
    end_time_ms = int(time.time() * 1000)
    processing_time_ms = end_time_ms - start_time_ms
    
    return {
        "category": category,
        "sentiment": sentiment,
        "priority": priority,
        "confidence_category": confidence_category,
        "confidence_sentiment": confidence_sentiment,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "suggested_team": suggested_team,
        "language": language,
        "language_code": language_code,
        "response": response,
        "processing_time_ms": processing_time_ms
    }


def run_sync_analysis(query: str) -> Dict[str, Any]:
    """Synchronous wrapper for running query analysis.
    
    Args:
        query: The customer query to analyze.
    
    Returns:
        Dictionary containing all analysis results.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_analysis(query))
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(run_analysis(query))


def record_analytics(result: Dict[str, Any]) -> None:
    """Record query results in analytics.
    
    Args:
        result: Dictionary containing analysis results.
    """
    analytics: QueryAnalytics = st.session_state.analytics
    analytics.record_query(
        category=result["category"],
        sentiment=result["sentiment"],
        priority=result["priority"],
        response_time_ms=result["processing_time_ms"],
        should_escalate=result["should_escalate"],
        language=result["language"]
    )


def render_priority_badge(priority: str) -> None:
    """Render a priority badge with color coding.
    
    Args:
        priority: The priority level to display.
    """
    color = get_priority_color(priority)
    emoji = get_priority_emoji(priority)
    st.markdown(
        f'<span style="background-color: {color}22; color: {color}; '
        f'border: 1px solid {color}; padding: 4px 12px; border-radius: 20px; '
        f'font-size: 0.85rem; font-weight: 600;">{emoji} {priority}</span>',
        unsafe_allow_html=True
    )


def render_confidence_bars(confidence_cat: int, confidence_sent: int) -> None:
    """Render confidence score progress bars.
    
    Args:
        confidence_cat: Category confidence score (0-100).
        confidence_sent: Sentiment confidence score (0-100).
    """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category Confidence**")
        st.progress(confidence_cat / 100, text=f"{confidence_cat}%")
    with col2:
        st.markdown("**Sentiment Confidence**")
        st.progress(confidence_sent / 100, text=f"{confidence_sent}%")


def render_escalation_alert(should_escalate: bool, reason: str, team: str) -> None:
    """Render escalation alert if needed.
    
    Args:
        should_escalate: Whether escalation is recommended.
        reason: Reason for escalation.
        team: Suggested team for escalation.
    """
    if should_escalate:
        st.error(
            f"🚨 **This query requires human escalation**\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Suggested Team:** {team}",
            icon="🚨"
        )


def render_result_cards(result: Dict[str, Any]) -> None:
    """Render the analysis result cards.
    
    Args:
        result: Dictionary containing analysis results.
    """
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
        lang_code = result.get("language_code", "en")
        st.markdown(
            f'<div style="background-color: #1a1d2e; border-left: 4px solid #667eea; '
            f'border-radius: 8px; padding: 16px; text-align: center;">'
            f'<div style="color: #a0aec0; font-size: 0.85rem;">🌐 Language</div>'
            f'<div style="color: #667eea; font-size: 1.2rem; font-weight: 600;">'
            f'{lang} ({lang_code})</div></div>',
            unsafe_allow_html=True
        )


def render_response_card(response: str, sentiment: str) -> None:
    """Render the AI response card.
    
    Args:
        response: The generated response text.
        sentiment: The detected sentiment for color coding.
    """
    color = SENTIMENT_COLORS.get(sentiment, "#667eea")
    st.markdown(
        f'<div style="background-color: #1a1d2e; border-left: 4px solid {color}; '
        f'border-radius: 8px; padding: 20px; margin: 16px 0;">'
        f'<div style="color: #667eea; font-size: 0.85rem; font-weight: 600; margin-bottom: 12px;">'
        f'🤖 AI SUPPORT AGENT</div>'
        f'<div style="color: #e0e0e0; font-size: 1rem; line-height: 1.6;">{response}</div></div>',
        unsafe_allow_html=True
    )


def render_analytics_dashboard(analytics: QueryAnalytics) -> None:
    """Render the analytics dashboard with charts and metrics.
    
    Args:
        analytics: The QueryAnalytics instance.
    """
    summary = analytics.get_summary()
    
    st.markdown("### 📊 Session Summary")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Queries", summary["total_queries"])
    with metric_cols[1]:
        avg_time = summary["average_response_time_ms"]
        st.metric("Avg Response Time", f"{avg_time:.0f} ms")
    with metric_cols[2]:
        st.metric("Escalation Rate", f"{summary['escalation_rate']:.1f}%")
    with metric_cols[3]:
        most_common = summary["most_common_category"] or "N/A"
        st.metric("Top Category", most_common)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📂 Category Breakdown")
        category_data = summary["category_breakdown"]
        if category_data:
            df_cat = pd.DataFrame(
                list(category_data.items()),
                columns=["Category", "Count"]
            )
            fig_cat = px.bar(
                df_cat,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map={cat: CATEGORY_INFO.get(cat, {}).get("color", "#667eea") 
                                   for cat in df_cat["Category"]}
            )
            fig_cat.update_layout(
                showlegend=False,
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig_cat, width='stretch')
        else:
            st.info("No category data yet. Analyze some queries to see statistics.")
    
    with col2:
        st.markdown("#### 💬 Sentiment Distribution")
        sentiment_data = summary["sentiment_distribution"]
        if sentiment_data:
            df_sent = pd.DataFrame(
                list(sentiment_data.items()),
                columns=["Sentiment", "Count"]
            )
            fig_sent = px.pie(
                df_sent,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS
            )
            fig_sent.update_layout(
                paper_bgcolor="#1a1d2e",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig_sent, width='stretch')
        else:
            st.info("No sentiment data yet. Analyze some queries to see statistics.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### ⚡ Priority Distribution")
        priority_data = summary["priority_distribution"]
        if priority_data:
            df_prio = pd.DataFrame(
                list(priority_data.items()),
                columns=["Priority", "Count"]
            )
            fig_prio = px.bar(
                df_prio,
                x="Priority",
                y="Count",
                color="Priority",
                color_discrete_map=PRIORITY_COLORS,
                category_orders={"Priority": ["Critical", "High", "Medium", "Low"]}
            )
            fig_prio.update_layout(
                showlegend=False,
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig_prio, width='stretch')
        else:
            st.info("No priority data yet. Analyze some queries to see statistics.")
    
    with col4:
        st.markdown("#### 🌐 Language Distribution")
        lang_data = summary["language_distribution"]
        if lang_data:
            df_lang = pd.DataFrame(
                list(lang_data.items()),
                columns=["Language", "Count"]
            )
            fig_lang = px.bar(
                df_lang,
                x="Language",
                y="Count",
                color="Language",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_lang.update_layout(
                showlegend=False,
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig_lang, width='stretch')
        else:
            st.info("No language data yet. Analyze some queries to see statistics.")


def export_history_csv() -> str:
    """Export conversation history as CSV.
    
    Returns:
        CSV-formatted string for download.
    """
    if not st.session_state.history:
        return "query,category,sentiment,priority,response,time\n"
    
    lines = ["query,category,sentiment,priority,response,time"]
    for entry in st.session_state.history:
        query = entry["query"].replace('"', '""')
        response = entry["response"].replace('"', '""')
        priority = entry.get("priority", "Medium")
        line = f'"{query}",{entry["category"]},{entry["sentiment"]},{priority},"{response}",{entry["time"]}'
        lines.append(line)
    
    return "\n".join(lines)


def render_sidebar() -> None:
    """Render the sidebar with help and settings."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: center; padding: 10px 0;">'
            '<h2 style="font-size: 1.8rem;">🎯 Query Analyzer</h2></div>',
            unsafe_allow_html=True
        )
        st.markdown('<hr style="border-color: #2d3561;">', unsafe_allow_html=True)
        
        st.markdown("### 📋 Query Categories")
        for cat, info in CATEGORY_INFO.items():
            st.markdown(
                f'<span style="color: {info["color"]};">●</span> {info["emoji"]} {cat}',
                unsafe_allow_html=True
            )
        
        st.markdown("")
        st.markdown("### 💡 Example Queries")
        examples = [
            "My bill seems incorrect this month.",
            "I can't login to my account, password reset isn't working.",
            "Where is my order? It's been 2 weeks!",
            "I want to return a damaged product I received.",
            "How do I upgrade my subscription plan?",
            "Your service has been amazing, just wanted to say thanks!"
        ]
        for ex in examples:
            st.markdown(f'<div style="background-color: #1a1d2e; border-left: 3px solid #667eea; '
                       f'border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 0.85rem; '
                       f'color: #e0e0e0;">🔹 {ex}</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #2d3561; margin-top: 20px;'>", unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", width='stretch'):
            st.session_state.history = []
            if "analytics" in st.session_state:
                st.session_state.analytics.reset()
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
            'Built with LangChain · Gemini 2.5 · Streamlit</div>',
            unsafe_allow_html=True
        )


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
        'Powered by LangChain · Google Gemini 2.5 · Multi-Agent AI</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr style="border-color: #2d3561;">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 Analyzer", "📊 Analytics"])
    
    with tab1:
        render_analyzer_tab()
    
    with tab2:
        analytics: QueryAnalytics = st.session_state.get("analytics", get_default_analytics())
        render_analytics_dashboard(analytics)


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
        
        query = sanitize_query(query)
        
        with st.spinner("🔄 Analyzing your query with AI agents..."):
            start_time = time.time()
            
            try:
                result = run_sync_analysis(query)
                
                elapsed = time.time() - start_time
                
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
                
                st.session_state.history.insert(0, entry)
                record_analytics(result)
                
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
                f"**You:** {entry['query'][:80]}{'...' if len(entry['query']) > 80 else ''}",
                expanded=False
            ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    cat = entry["category"]
                    color = CATEGORY_INFO.get(cat, {}).get("color", "#a0aec0")
                    emoji = CATEGORY_INFO.get(cat, {}).get("emoji", "💬")
                    st.markdown(
                        f'<span style="color: {color};">{emoji} {cat}</span>',
                        unsafe_allow_html=True
                    )
                with col2:
                    sent = entry["sentiment"]
                    sent_color = SENTIMENT_COLORS.get(sent, "#a0aec0")
                    st.markdown(
                        f'<span style="color: {sent_color};">💬 {sent}</span>',
                        unsafe_allow_html=True
                    )
                
                priority = entry.get("priority", "Medium")
                st.markdown(f"**Priority:** {get_priority_emoji(priority)} {priority}")
                
                st.markdown("---")
                st.markdown(entry["response"])
                
                st.markdown(
                    f'<div style="text-align: right; color: #718096; font-size: 0.75rem;">'
                    f'⏱️ {entry["time"]}</div>',
                    unsafe_allow_html=True
                )


def main() -> None:
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
