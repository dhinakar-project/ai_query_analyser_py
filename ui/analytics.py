import streamlit as st
import pandas as pd
import plotly.express as px


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
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
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
    
    st.markdown("---")
    
    st.markdown("#### ⚡ Priority Distribution")
    if priority_counts:
        df_prio = pd.DataFrame(list(priority_counts.items()), columns=["Priority", "Count"])
        df_prio["Priority"] = pd.Categorical(df_prio["Priority"], categories=["Critical", "High", "Medium", "Low"], ordered=True)
        df_prio = df_prio.sort_values("Priority")
        fig_prio = px.bar(df_prio, x="Priority", y="Count", color="Priority", color_discrete_map=PRIORITY_COLORS)
        fig_prio.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
        st.plotly_chart(fig_prio, width='stretch')
