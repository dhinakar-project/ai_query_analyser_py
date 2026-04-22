import os
import asyncio
import uuid
import time

import streamlit as st

from voice.transcript_processor import process_voice_transcript
from voice.speaker import play_response


def render_voice_agent_tab() -> None:
    """Render the voice agent tab."""
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Add it to your .env file.")
        return

    st.session_state.setdefault("va_messages", [])
    st.session_state.setdefault("va_state", "idle")
    st.session_state.setdefault("va_response_index", 0)
    st.session_state.setdefault("va_call_count", 0)
    st.session_state.setdefault("va_satisfaction", 94)
    st.session_state.setdefault("va_tts_lang", "en")
    st.session_state.setdefault("va_muted", False)
    st.session_state.setdefault("va_last_result", None)

    st.markdown("### 🔊 Text-to-Speech Support Agent")
    st.caption("This agent analyzes your query through the full LangGraph pipeline and converts the AI response to speech using Google TTS.")
    st.divider()

    m1, m2, m3 = st.columns(3)
    call_count = st.session_state.va_call_count
    m1.metric("Queries Processed", call_count)
    
    # Calculate avg latency from last_result history
    if st.session_state.va_last_result:
        last_latency = st.session_state.va_last_result.get("processing_time_ms", 0)
        m2.metric("Last Response Time", f"{last_latency}ms")
    else:
        m2.metric("Last Response Time", "—")
    
    # Calculate real escalation rate from va_messages
    escalated = sum(
        1 for msg in st.session_state.get("va_messages", [])
        if msg.get("role") == "assistant"
    )
    non_escalated = call_count - sum(
        1 for msg in [st.session_state.va_last_result or {}]
        if msg.get("should_escalate")
    )
    resolution_rate = (non_escalated / call_count * 100) if call_count > 0 else 0
    m3.metric("Resolution Rate", f"{resolution_rate:.0f}%")

    st.divider()

    lang_col, mute_col = st.columns([3, 1])

    with lang_col:
        tts_lang = st.selectbox(
            "Agent speaks in",
            options=["en", "hi", "es", "fr", "de"],
            format_func=lambda x: {
                "en": "English",
                "hi": "Hindi",
                "es": "Spanish",
                "fr": "French",
                "de": "German"
            }[x],
            key="va_lang_select"
        )
        st.session_state.va_tts_lang = tts_lang

    with mute_col:
        st.session_state.va_muted = st.toggle("Mute", value=st.session_state.va_muted)

    st.divider()

    st.markdown("### 🔊 Text-to-Speech Support Agent")
    st.caption("Type a customer query below. The AI will analyze it and read the response aloud.")
    
    tts_query = st.text_input(
        "Enter customer query:",
        placeholder="e.g. My package hasn't arrived after 2 weeks",
        key="tts_query_input"
    )
    
    speak_clicked = st.button("🔊 Analyze & Speak Response", use_container_width=True)
    
    if speak_clicked and tts_query.strip():
        with st.spinner("Analyzing and generating audio response..."):
            result = asyncio.run(
                process_voice_transcript(
                    transcript=f"Customer: {tts_query}",
                    call_id=str(uuid.uuid4())
                )
            )
        ai_response = result.get("response", "I am here to help you.")
        st.session_state.va_messages.append({"role": "user", "content": tts_query})
        st.session_state.va_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.va_last_result = result
        st.session_state.va_call_count += 1
        play_response(ai_response, lang=st.session_state.va_tts_lang)

    st.divider()
    st.markdown("#### Conversation")

    if not st.session_state.va_messages:
        st.info("Click Start or type a query below to begin")
    else:
        for msg in st.session_state.va_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if st.session_state.va_last_result:
        result = st.session_state.va_last_result
        with st.expander("Last Call Analysis", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category", result.get("category", "—"))
            c2.metric("Sentiment", result.get("sentiment", "—"))
            c3.metric("Priority", result.get("priority", "—"))
            c4.metric("Escalate", "Yes" if result.get("should_escalate") else "No")

            st.info(result.get("response", ""))

            trace = result.get("reasoning_trace", [])
            if trace:
                with st.expander("Reasoning Trace"):
                    for step in trace:
                        st.caption(f"-> {step}")