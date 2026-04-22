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

    st.markdown("### AI Voice Support Agent")
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Calls Today", st.session_state.va_call_count)
    m2.metric("Avg Handle Time", "1m 42s")
    m3.metric("Satisfaction", f"{st.session_state.va_satisfaction}%")

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

    _, orb_col, _ = st.columns([1, 2, 1])
    with orb_col:
        state = st.session_state.va_state
        orb_map = {
            "idle": ("Agent ready", 0.0),
            "listening": ("Listening...", 0.35),
            "thinking": ("AI thinking...", 0.70),
            "speaking": ("Agent speaking...", 1.0),
        }
        label, progress_val = orb_map.get(state, orb_map["idle"])
        st.write(f"**{label}**")
        st.progress(progress_val, text=label)

    b1, b2, b3 = st.columns(3)

    demo_queries = [
        "My order #45621 hasn't arrived in 2 weeks, I am really frustrated",
        "I was charged twice for my subscription this month",
        "I cannot log into my account, password reset is not working",
        "I want to return a product I bought last week",
        "Can you upgrade my plan to Pro please",
    ]

    with b1:
        start_clicked = st.button("Start", use_container_width=True)

    with b2:
        end_clicked = st.button("End", use_container_width=True)

    with b3:
        new_clicked = st.button("New Session", use_container_width=True)

    if end_clicked:
        st.session_state.va_state = "idle"
        st.rerun()

    if new_clicked:
        st.session_state.va_messages = []
        st.session_state.va_state = "idle"
        st.session_state.va_call_count = 0
        st.session_state.va_last_result = None
        st.rerun()

    if start_clicked:
        st.session_state.va_state = "listening"
        st.session_state.va_call_count += 1

        query = demo_queries[
            st.session_state.va_response_index % len(demo_queries)
        ]

        with st.status("Processing voice input...", expanded=True) as status:
            st.write("Capturing audio input...")
            time.sleep(1.0)
            st.write(f"Transcribed: *{query}*")
            time.sleep(0.8)
            st.session_state.va_state = "thinking"
            st.write("Running LangGraph pipeline...")

            result = asyncio.run(
                process_voice_transcript(
                    transcript=f"Customer: {query}",
                    call_id=str(uuid.uuid4())
                )
            )

            st.write("Response ready")
            status.update(label="Complete", state="complete")

        st.session_state.va_messages.append(
            {"role": "user", "content": query}
        )

        ai_response = result.get("response", "I am here to help you.")
        st.session_state.va_messages.append(
            {"role": "assistant", "content": ai_response}
        )

        st.session_state.va_last_result = result
        st.session_state.va_response_index += 1
        st.session_state.va_state = "speaking"

        play_response(ai_response, lang=st.session_state.va_tts_lang)

        st.session_state.va_state = "idle"

    st.divider()
    user_input = st.chat_input("Or type a customer query here...")

    if user_input:
        st.session_state.va_messages.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("Agent thinking..."):
            result = asyncio.run(
                process_voice_transcript(
                    transcript=f"Customer: {user_input}",
                    call_id=str(uuid.uuid4())
                )
            )

        ai_response = result.get("response", "I am here to help you.")
        st.session_state.va_messages.append(
            {"role": "assistant", "content": ai_response}
        )

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