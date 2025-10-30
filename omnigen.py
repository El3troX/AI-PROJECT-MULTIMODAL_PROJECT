import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import base64
import os
from dotenv import load_dotenv

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
            mime_type = file.type or "application/octet-stream"
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
    audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

    if st.button("Transcribe & Translate", key="audio_btn"):
        if audio:
            prompt = "Transcribe and translate this audio into clear English text."
            with st.spinner("Processing audio..."):
                transcript = gemini_generate(prompt, file=audio)
            st.success(transcript)
        else:
            st.warning("Please upload an audio file first.")
