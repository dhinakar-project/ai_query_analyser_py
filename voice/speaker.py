import streamlit as st
from voice.tts import speak

def play_response(text: str, lang: str = "en") -> None:
    if not text or not text.strip():
        return
    
    if st.session_state.get("va_muted", False):
        return

    try:
        audio_bytes = speak(text=text, lang=lang)
        st.audio(audio_bytes, format="audio/mp3")
        st.caption("Click play to hear the agent response")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
