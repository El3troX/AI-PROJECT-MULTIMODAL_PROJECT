import sys
import types

# Temporary patch for Python 3.13 where audioop was removed
if 'audioop' not in sys.modules:
    fake_audioop = types.ModuleType("audioop")
    def _stub(*args, **kwargs):
        raise NotImplementedError("audioop removed in Python 3.13; please downgrade to 3.12.")
    for fn in ["add", "mul", "bias", "adpcm2lin", "lin2adpcm", "avg", "avgpp", "cross", "findfactor",
               "findfit", "findmax", "getsample", "max", "maxpp", "minmax", "reverse", "rms", "tomono",
               "tostereo", "ulaw2lin", "lin2ulaw"]:
        setattr(fake_audioop, fn, _stub)
    sys.modules['audioop'] = fake_audioop

# Now safe to import pydub/audiorecorder
from audiorecorder import audiorecorder



import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import base64
import os
from dotenv import load_dotenv
from audiorecorder import audiorecorder   # 👈 NEW

# ---------------- SETUP ----------------
load_dotenv()  # loads .env file if exists

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="OmniGen", page_icon="🤖", layout="wide")
st.title("🤖 OmniGen – Gemini Multimodal AI Assistant")

# ---------------- GEMINI HELPER ----------------
def gemini_generate(prompt, file=None):
    try:
        model = genai.GenerativeModel(MODEL)
        parts = [{"text": prompt}]
        if file:
            file_bytes = file.read()
            mime_type = getattr(file, "type", "application/octet-stream")
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": encoded
                }
            })
        response = model.generate_content({"parts": parts})
        return response.text or "No response received."
    except Exception as e:
        st.error(f"❌ Gemini API error: {e}")
        return "Gemini service unavailable or request failed."

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📘 Ask My Notes", "🖼️ Image Story", "🎙️ Voice to English"])

# ---------------------------------------------------------------------------
# 📘 PDF Q&A
# ---------------------------------------------------------------------------
with tab1:
    st.header("📄 Ask Questions from a PDF (Gemini)")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    query = st.text_input("Enter your question")

    if st.button("Generate Answer", key="pdf_btn"):
        if pdf and query:
            reader = PdfReader(pdf)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            prompt = f"Context:\n{text}\n\nQuestion: {query}\n\nAnswer clearly and concisely."
            with st.spinner("Generating answer..."):
                answer = gemini_generate(prompt)
            st.success(answer)
        else:
            st.warning("Please upload a PDF and enter a question.")

# ---------------------------------------------------------------------------
# 🖼️ Image Caption / Story
# ---------------------------------------------------------------------------
with tab2:
    st.header("🖼️ Image Caption or Story Generator (Gemini)")
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption", key="img_btn"):
        if img:
            prompt = "Describe this image in a creative short caption or story."
            with st.spinner("Analyzing image..."):
                caption = gemini_generate(prompt, file=img)
            st.image(img, use_column_width=True)
            st.success(caption)
        else:
            st.warning("Please upload an image first.")

# ---------------------------------------------------------------------------
# 🎙️ Voice → English
# ---------------------------------------------------------------------------
with tab3:
    st.header("🎙️ Speech to English Translator (Gemini)")

    # 👇 NEW: Let user choose between record or upload
    mode = st.radio("Select Input Method:", ["🎙️ Record from Mic", "📁 Upload File"])

    audio_file = None

    if mode == "🎙️ Record from Mic":
        st.write("Click below to record audio:")
        audio = audiorecorder("🎤 Start Recording", "🔴 Recording... Click again to stop")

        if len(audio) > 0:
            st.audio(audio.tobytes(), format="audio/wav")
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.tobytes())
            audio_file = open("temp_audio.wav", "rb")
            st.success("✅ Audio recorded successfully!")

    else:
        uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
        if uploaded:
            audio_file = uploaded

    # Process the audio
    if st.button("Transcribe & Translate", key="audio_btn"):
        if audio_file:
            prompt = "Transcribe and translate this audio into clear English text."
            with st.spinner("Processing audio..."):
                transcript = gemini_generate(prompt, file=audio_file)
            st.success(transcript)
        else:
            st.warning("Please record or upload an audio file first.")

