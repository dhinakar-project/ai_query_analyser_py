import io
from gtts import gTTS

def speak(text: str, lang: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
