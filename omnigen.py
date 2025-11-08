import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import base64
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from streamlit_drawable_canvas import st_canvas

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.0-flash"

# Streamlit UI setup
st.set_page_config(page_title="OmniGen", page_icon="🤖", layout="wide")
st.title("🤖 OmniGen – Gemini Multimodal AI Assistant")

# ---------------- Gemini API Wrappers ----------------
def gemini_generate(prompt, file=None):
    try:
        model = genai.GenerativeModel(MODEL)
        parts = [{"text": prompt}]
        if file:
            data = file.read()
            file.seek(0)
            encoded = base64.b64encode(data).decode("utf-8")
            mime = file.type or "application/octet-stream"
            parts.append({"inline_data": {"mime_type": mime, "data": encoded}})
        response = model.generate_content({"parts": parts})
        return response.text or ""
    except Exception:
        return "Gemini service unavailable."

def gemini_generate_chat(history):
    try:
        model = genai.GenerativeModel(MODEL)
        contents = [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in history]
        response = model.generate_content(contents)
        return response.text or ""
    except Exception:
        return "Gemini service unavailable."

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Ask My Notes",
    "🖼️ Image Prompt Mode",
    "🎙️ Audio Prompt Mode",
    "💬 Chat Mode",
    "🛠️ Image Editor"
])

# ---------------- Tab 1: PDF QA ----------------
with tab1:
    st.header("📄 Ask Questions from a PDF (Gemini)")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    query = st.text_input("Enter your question")

    if st.button("Generate Answer", key="pdf_btn"):
        if pdf and query:
            reader = PdfReader(pdf)
            pages = [page.extract_text() or "[Unextractable Page]" for page in reader.pages]

            chunks, buffer = [], ""
            limit = 8000
            for page_text in pages:
                if len(buffer) + len(page_text) < limit:
                    buffer += "\n" + page_text
                else:
                    chunks.append(buffer)
                    buffer = page_text
            chunks.append(buffer)

            full_output = ""
            for chunk in chunks:
                prompt = f"Context:\n{chunk}\n\nQuestion: {query}\nProvide a single clear answer."
                full_output += gemini_generate(prompt).strip() + "\n\n"

            st.success(full_output.strip())
        else:
            st.warning("Upload a PDF and enter a question.")

# ---------------- Tab 2: Image Prompt ----------------
with tab2:
    st.header("🖼️ Image Prompt Mode (Gemini)")
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    user_prompt = st.text_area("Enter your prompt for the image")
    if st.button("Generate", key="img_btn"):
        if img and user_prompt:
            with st.spinner("Processing..."):
                out = gemini_generate(user_prompt, file=img)
            st.image(img, use_column_width=True)
            st.success(out)
        else:
            st.warning("Upload image and enter prompt.")

# ---------------- Tab 3: Audio Prompt ----------------
with tab3:
    st.header("🎙️ Audio Prompt Mode (Gemini)")
    audio = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
    if audio:
        st.audio(audio)
    user_prompt = st.text_area("Enter your prompt for the audio")
    if st.button("Process Audio", key="audio_btn"):
        if audio and user_prompt:
            with st.spinner("Processing audio..."):
                st.success(gemini_generate(user_prompt, file=audio))
        else:
            st.warning("Upload audio and enter prompt.")

# ---------------- Tab 4: Chat ----------------
with tab4:
    st.header("💬 Gemini Chat (with History)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Enter your message")
    if st.button("Send", key="chat_send"):
        if user_input.strip():
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            reply = gemini_generate_chat(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "ai", "text": reply})
    for msg in st.session_state.chat_history:
        role = "**You:**" if msg["role"] == "user" else "**AI:**"
        st.markdown(f"{role} {msg['text']}")

# ---------------- Tab 5: Image Editor ----------------
with tab5:
    st.header("🛠️ Image Editor – OpenCV Inpainting")
    edit_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if edit_img:
        img = Image.open(BytesIO(edit_img.read())).convert("RGB")
        st.write("Draw over the object you want to remove:")
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.5)",
            stroke_width=30,
            stroke_color="#FFFFFF",
            background_image=img,
            height=img.height,
            width=img.width,
            drawing_mode="freedraw",
            key="canvas_editor",
        )
        if st.button("Generate Edited Image"):
            if canvas.image_data is not None:
                mask = (canvas.image_data[:, :, 3] > 0).astype(np.uint8) * 255
                result = cv2.inpaint(np.array(img), mask, 3, cv2.INPAINT_TELEA)
                st.image(result, caption="Edited Image", use_column_width=True)
