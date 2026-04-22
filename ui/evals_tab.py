import asyncio

import streamlit as st
import pandas as pd

from evals.runner import run_evals, load_report, save_report


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