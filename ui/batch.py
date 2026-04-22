import asyncio
import concurrent.futures
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px

from graph.runner import run_graph


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
        "reasoning_trace": result.get("reasoning_trace", [])
    }


def render_batch_test_tab() -> None:
    """Render the batch query tester tab."""
    st.markdown("### 🧪 Batch Query Tester")
    st.markdown("Test multiple queries at once and see results in a table.")
    
    default_queries = """My bill is incorrect this month
The app keeps crashing when I try to login
I want to return a damaged product
Where is my order? It's been 2 weeks
Can you help me reset my password?
This is the third time this issue happened!
Thank you for your excellent service"""
    
    queries_text = st.text_area(
        "Enter queries (one per line):",
        value=default_queries,
        height=150,
        key="batch_queries"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        run_batch = st.button("▶ Run Batch", type="primary")
    
    if run_batch and queries_text:
        queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
        total = len(queries)
        
        results = []
        progress_bar = st.progress(0)
        
        for i, query in enumerate(queries):
            try:
                result = run_async_in_thread(
                    run_analysis_streaming(query, f"batch_{i}", "English")
                )
                results.append({
                    "Query": query[:50] + "..." if len(query) > 50 else query,
                    "Category": result.get("category", ""),
                    "Sentiment": result.get("sentiment", ""),
                    "Priority": result.get("priority", ""),
                    "Escalate": "✓" if result.get("should_escalate") else "✗",
                    "Time(ms)": result.get("processing_time_ms", 0)
                })
            except Exception as e:
                results.append({
                    "Query": query[:50] + "..." if len(query) > 50 else query,
                    "Category": "Error",
                    "Sentiment": "Error",
                    "Priority": "Error",
                    "Escalate": "Error",
                    "Time(ms)": 0
                })
            progress_bar.progress((i + 1) / total)
        
        st.markdown("### 📊 Results")
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.markdown("### 📈 Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                cat_dist = df["Category"].value_counts()
                fig_cat = px.bar(cat_dist, x=cat_dist.index, y=cat_dist.values, title="Categories")
                fig_cat.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig_cat, use_container_width=True)
            with summary_col2:
                sent_dist = df["Sentiment"].value_counts()
                fig_sent = px.pie(sent_dist, names=sent_dist.index, values=sent_dist.values, title="Sentiments")
                fig_sent.update_layout(paper_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig_sent, use_container_width=True)
            with summary_col3:
                avg_time = df["Time(ms)"].mean()
                st.metric("Avg Time", f"{avg_time:.0f}ms")