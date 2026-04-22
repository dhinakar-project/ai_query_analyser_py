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

import logging
import os
import uuid
from datetime import datetime

import streamlit as st

from utils.guardrails import validate_query, sanitize_query, redact_pii
from observability.costs import get_session_cost, format_cost, reset_session_cost
from evals.runner import load_report
from ui.analyzer import render_analyzer_tab
from ui.analytics import render_analytics_dashboard
from ui.evals_tab import render_evals_tab
from ui.batch import render_batch_test_tab
from ui.voice import render_voice_agent_tab


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
        
        st.sidebar.divider()
        st.sidebar.markdown("### Session Summary")
        history = st.session_state.get("history", [])
        st.sidebar.metric("Queries analyzed", len(history))
        if history:
            escalated = sum(1 for h in history if h.get("should_escalate"))
            st.sidebar.metric("Escalations triggered", escalated)
            avg_ms = sum(int(h.get("time","0 ms").replace(" ms","")) for h in history) / len(history)
            st.sidebar.metric("Avg response time", f"{avg_ms:.0f} ms")
        
        st.markdown("<hr style='border-color: #2d3561; margin-top: 20px;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; color: #718096; font-size: 0.75rem;">'
            'Built with LangGraph · Gemini 2.5 · Streamlit</div>',
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
        'Powered by LangGraph · Google Gemini 2.5 · Multi-Agent AI</p>',
        unsafe_allow_html=True
    )
    
    eval_report = load_report("evals/last_report.json")
    if eval_report:
        st.markdown("### 📈 Last Benchmark Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Category Accuracy", f"{eval_report.category_accuracy:.1%}", delta="↑" if eval_report.category_accuracy > 0.8 else "↓")
        with col2:
            st.metric("Sentiment Accuracy", f"{eval_report.sentiment_accuracy:.1%}", delta="↑" if eval_report.sentiment_accuracy > 0.75 else "↓")
        with col3:
            st.metric("Priority Accuracy", f"{eval_report.priority_accuracy:.1%}", delta="↑" if eval_report.priority_accuracy > 0.8 else "↓")
        with col4:
            st.metric("Avg Latency", f"{eval_report.avg_latency_ms:.0f}ms", delta="-")
        
        st.caption(f"Last run: {eval_report.timestamp} · 30 test cases · Re-run from the Evals tab")
        st.markdown("---")
    else:
        st.info("💡 Run evals to see live accuracy scores.")
        st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analyzer", "📊 Analytics", "📋 Evals", "🧪 Batch Test", "🔊 TTS Support Agent"])

    with tab1:
        render_analyzer_tab()

    with tab2:
        render_analytics_dashboard()

    with tab3:
        render_evals_tab()

    with tab4:
        render_batch_test_tab()

    with tab5:
        render_voice_agent_tab()


def main() -> None:
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()