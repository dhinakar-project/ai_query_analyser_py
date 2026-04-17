"""AI-Based Customer Query Analyzer - Streamlit Application."""

import traceback
import time

import streamlit as st

from agents import run_classifier_agent, run_sentiment_agent, run_responder_agent

st.set_page_config(
    page_title="AI Customer Query Analyzer",
    page_icon="🎯",
    layout="centered",
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
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1d2e 0%, #16213e 100%);
        border-right: 1px solid #2d3561;
    }
    
    .gradient-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .result-card {
        background-color: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
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
    
    .metric-card {
        background-color: #1a1d2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .user-bubble {
        background-color: #1e2130;
        border-left: 3px solid #4a5568;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .response-card {
        background-color: #1a1d2e;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 4px;
    }
    
    .info-box {
        background-color: #1a1d2e;
        border-left: 3px solid #667eea;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #e0e0e0;
    }
    
    div[data-testid="stExpander"] {
        background-color: #1a1d2e;
        border-radius: 8px;
    }
    
    .category-list li {
        margin: 6px 0;
        color: #e0e0e0;
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


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment."""
    return SENTIMENT_COLORS.get(sentiment, "#a0aec0")


def get_category_color(category: str) -> str:
    """Get color for category."""
    return CATEGORY_INFO.get(category, {}).get("color", "#a0aec0")


def get_category_emoji(category: str) -> str:
    """Get emoji for category."""
    return CATEGORY_INFO.get(category, {}).get("emoji", "💬")


def render_metric_card(label: str, value: str, border_color: str):
    """Render a custom metric card."""
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: {border_color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_category_badge(category: str):
    """Render a category badge."""
    color = get_category_color(category)
    emoji = get_category_emoji(category)
    st.markdown(f'<span class="badge" style="background-color: {color}22; color: {color}; border: 1px solid {color};">{emoji} {category}</span>', unsafe_allow_html=True)


def render_sentiment_badge(sentiment: str):
    """Render a sentiment badge."""
    color = get_sentiment_color(sentiment)
    st.markdown(f'<span class="badge" style="background-color: {color}22; color: {color}; border: 1px solid {color};">{sentiment}</span>', unsafe_allow_html=True)


def render_user_bubble(query: str):
    """Render a user query bubble."""
    st.markdown(f"""
        <div class="user-bubble">
            <div style="color: #a0aec0; font-size: 0.75rem; font-weight: 600; margin-bottom: 6px;">YOU</div>
            <div style="color: #e0e0e0; font-size: 1rem;">{query}</div>
        </div>
    """, unsafe_allow_html=True)


def render_response_card(response: str, sentiment: str):
    """Render an AI response card."""
    color = get_sentiment_color(sentiment)
    st.markdown(f"""
        <div class="response-card" style="border-left-color: {color};">
            <div style="color: #667eea; font-size: 0.75rem; font-weight: 600; margin-bottom: 8px;">🤖 AI SUPPORT AGENT</div>
            <div style="color: #e0e0e0; font-size: 1rem; line-height: 1.6;">{response}</div>
        </div>
    """, unsafe_allow_html=True)


def render_history_entry(entry: dict):
    """Render a conversation history entry."""
    render_user_bubble(entry["query"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        render_category_badge(entry["category"])
    with col2:
        render_sentiment_badge(entry["sentiment"])
    
    st.markdown("")
    render_response_card(entry["response"], entry["sentiment"])
    
    st.markdown(f'<div style="text-align: right; color: #718096; font-size: 0.8rem;">⏱️ {entry["time"]}</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #2d3561; margin: 20px 0;'>", unsafe_allow_html=True)


def main():
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 10px 0;"><h2 style="font-size: 1.8rem;">🎯 Query Analyzer</h2></div>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color: #2d3561;">', unsafe_allow_html=True)
        
        st.markdown("### 📋 Query Categories")
        for cat, info in CATEGORY_INFO.items():
            st.markdown(f'<span style="color: {info["color"]};">●</span> {info["emoji"]} {cat}', unsafe_allow_html=True)
        
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
            st.markdown(f'<div class="info-box">🔹 {ex}</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #2d3561; margin-top: 30px;'>", unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; color: #718096; font-size: 0.75rem;">Built with LangChain · Gemini · Streamlit</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    st.markdown('<h1 class="gradient-title">AI-Based Customer Query Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a0aec0; margin-bottom: 30px;">Powered by LangChain · Google Gemini · Multi-Agent AI</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color: #2d3561;">', unsafe_allow_html=True)
    
    query = st.text_area(
        "Enter your customer query:",
        placeholder="e.g., My bill seems incorrect this month. I've been charged twice for the same item.",
        height=120,
        key="query_input"
    )
    
    col_btn = st.columns([1])
    with col_btn[0]:
        submit = st.button("🚀 Analyze Query")
    
    if submit:
        if not query or not query.strip():
            st.warning("⚠️ Please enter a query to analyze.")
            return
        
        with st.spinner("🔄 Analyzing your query with AI agents..."):
            start_time = time.time()
            
            try:
                category = run_classifier_agent(query)
                sentiment = run_sentiment_agent(query)
                response = run_responder_agent(query, category, sentiment)
                elapsed = time.time() - start_time
                
                entry = {
                    "query": query,
                    "category": category,
                    "sentiment": sentiment,
                    "response": response,
                    "time": f"{elapsed:.2f}s"
                }
                st.session_state.history.insert(0, entry)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Analysis Results", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    render_metric_card("📂 Category", category, get_category_color(category))
                with col2:
                    render_metric_card("💬 Sentiment", sentiment, get_sentiment_color(sentiment))
                with col3:
                    render_metric_card("⏱️ Time", f"{elapsed:.2f}s", "#667eea")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("### 💬 AI Response", unsafe_allow_html=True)
                render_response_card(response, sentiment)
                
            except EnvironmentError as e:
                st.error(f"🚫 Configuration Error: {str(e)}")
                st.info("💡 Please make sure your `.env` file contains a valid `GEMINI_API_KEY`.")
            except Exception as e:
                traceback.print_exc()
                st.error(f"🚫 An error occurred: {str(e)}")
    
    if st.session_state.history:
        st.markdown("<hr style='border-color: #2d3561; margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("### 📜 Conversation History", unsafe_allow_html=True)
        
        for entry in st.session_state.history:
            render_history_entry(entry)


if __name__ == "__main__":
    main()
