import asyncio

import streamlit as st
import pandas as pd

from evals.runner import run_evals, load_report, save_report


def render_evals_tab() -> None:
    """Render the evals dashboard tab."""
    st.markdown("### 📋 Model Evaluation")
    
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_evals_clicked = st.button("▶ Run Fresh Evals", use_container_width=True, type="primary")
    with col_info:
        st.caption("Runs 30 test cases from the golden dataset in parallel (Semaphore=5). "
                   "Takes ~2-3 minutes. Results vary ±3-5% due to LLM temperature.")
    
    if run_evals_clicked:
        progress = st.progress(0, text="Starting evaluation...")
        with st.spinner("Running 30 eval cases in parallel..."):
            report = asyncio.run(run_evals(verbose=False))
            save_report(report)
            st.session_state.last_eval_report = report
        progress.progress(100, text="Complete!")
        st.success(f"Eval complete! Category accuracy: {report.category_accuracy:.1%}")
        st.rerun()

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
        st.info("No evaluation results yet. Run evals to see metrics.")
    
    st.markdown("---")
    st.markdown("### 👤 Human Feedback Score")
    
    helpful_count = sum(
        1 for k, v in st.session_state.items()
        if k.startswith("fb_") and v == "helpful"
    )
    total_rated = sum(
        1 for k in st.session_state
        if k.startswith("fb_")
    )
    
    if total_rated > 0:
        feedback_rate = helpful_count / total_rated * 100
        st.metric("Human Feedback Score", f"{feedback_rate:.0f}%", 
                  delta=f"{total_rated} responses rated")
        st.caption("Collected from user 👍/👎 ratings on generated responses")
    else:
        st.info("No feedback collected yet. Rate some responses in the Analyzer tab.")