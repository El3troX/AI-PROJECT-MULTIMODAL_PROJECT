import os
import io
import base64
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from PyPDF2 import PdfReader
from PIL import Image
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------------------
# ⚙️ SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="OmniGen", page_icon="🤖", layout="wide")
st.title("🤖 OmniGen – Gemini Multimodal AI Assistant")

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not API_KEY:
    st.warning("⚠️ Please set GEMINI_API_KEY in .env or Streamlit secrets.")
else:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        st.error(f"Gemini initialization failed: {e}")

# ---------------------------------------------------------------------------
# 🔧 UTILITIES
# ---------------------------------------------------------------------------
def _uploaded_file_to_bytes(upl) -> bytes:
    return upl.getvalue() if upl else b""

def _to_inline_part(data_bytes: bytes, mime: str):
    return {
        "inline_data": {
            "mime_type": mime or "application/octet-stream",
            "data": base64.b64encode(data_bytes).decode("utf-8"),
        }
    }

def gemini_generate(prompt: str,
                    files: Optional[List[dict]] = None,
                    model_name: str = MODEL_NAME) -> str:
    """Send text + optional media to Gemini."""
    try:
        model = genai.GenerativeModel(model_name)
        parts = [{"text": prompt}]
        if files:
            for f in files:
                parts.append(_to_inline_part(f["data_bytes"], f["mime_type"]))
        resp = model.generate_content({"parts": parts})
        return (resp.text or "").strip() or "No response received."
    except Exception as e:
        return f"Gemini service unavailable or request failed.\n\nDetails: {e}"

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text)
    except Exception as e:
        return f"[PDF read error: {e}]"

def inpaint_with_mask(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """CPU inpainting using OpenCV's Telea algorithm."""
    img = np.array(image_pil.convert("RGB"))
    mask = np.array(mask_pil.convert("L"))
    mask = (mask > 0).astype(np.uint8) * 255
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

# ---------------------------------------------------------------------------
# 💾 SESSION STATE
# ---------------------------------------------------------------------------
st.session_state.setdefault("text_history", [])
st.session_state.setdefault("pdf_context", "")

# ---------------------------------------------------------------------------
# 🧭 TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Text & PDF Chat",
    "🖼️ Image Q&A Mode",
    "🛠️ Image Editor Mode",
    "🎙️ Audio Mode",
    "💬 Chat Mode"
])

# ---------------------------------------------------------------------------
# 📘 TEXT / PDF CHAT
# ---------------------------------------------------------------------------
with tab1:
    st.header("📘 Text and PDF Chat Mode")
    st.write("Upload a document and ask Gemini questions about its contents.")

    with st.expander("📎 Optional: Upload a PDF as context"):
        pdf_upl = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upl_tab1")
        if pdf_upl:
            pdf_bytes = _uploaded_file_to_bytes(pdf_upl)
            text = extract_pdf_text(pdf_bytes)
            st.session_state.pdf_context = text[:200000]
            st.success("✅ PDF loaded successfully.")
            st.text_area("Preview (first ~3k chars)", text[:3000], height=200)

    # Display chat history
    if st.session_state.text_history:
        st.markdown("### Conversation")
    for turn in st.session_state.text_history:
        st.chat_message(turn["role"]).write(turn["text"])

    user_msg = st.chat_input("Type your message for Gemini...")
    if user_msg:
        st.session_state.text_history.append({"role": "user", "text": user_msg})
        system_rules = (
            "You are a concise, context-aware assistant. "
            "Avoid prefacing answers with 'Part 1:' or headings unless requested."
        )
        pdf_context = st.session_state.pdf_context
        prompt = (
            f"{system_rules}\n\nPDF Context:\n{pdf_context}\n\nUser: {user_msg}"
            if pdf_context else f"{system_rules}\n\nUser: {user_msg}"
        )
        with st.spinner("Generating response..."):
            answer = gemini_generate(prompt)
        st.session_state.text_history.append({"role": "assistant", "text": answer})
        st.chat_message("assistant").write(answer)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear chat"):
            st.session_state.text_history.clear()
            st.rerun()
    with c2:
        if st.session_state.text_history:
            st.download_button(
                "⬇️ Download conversation",
                data="\n\n".join(f"{t['role'].upper()}: {t['text']}" for t in st.session_state.text_history),
                file_name="omnigen_chat.txt",
                mime="text/plain",
            )

# ---------------------------------------------------------------------------
# 🖼️ IMAGE Q&A / CAPTIONING MODE
# ---------------------------------------------------------------------------
with tab2:
    st.header("🖼️ Image Q&A and Captioning Mode")
    st.write("Upload an image and ask Gemini to describe, analyze, or extract information from it.")

    img_upl = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img_upl_tab2")
    user_prompt_img = st.text_area(
        "Prompt for Gemini:",
        placeholder="Examples: 'Describe this image in detail', 'What objects are present?', 'Extract text from the image'.",
        height=100,
    )

    if img_upl:
        st.image(img_upl, caption="Uploaded image", use_column_width=True)

    if st.button("🔎 Ask Gemini about the image", disabled=not img_upl):
        if not API_KEY:
            st.error("GEMINI_API_KEY missing.")
        elif not user_prompt_img.strip():
            st.warning("Please enter a prompt for the image.")
        else:
            with st.spinner("Analyzing image with Gemini..."):
                img_bytes = _uploaded_file_to_bytes(img_upl)
                answer = gemini_generate(
                    user_prompt_img,
                    files=[{"mime_type": img_upl.type or "image/png", "data_bytes": img_bytes}],
                )
            st.success(answer)

    st.caption("Note: Gemini analyzes the image but cannot modify or edit it.")

# ---------------------------------------------------------------------------
# 🛠️ IMAGE EDITOR MODE
# ---------------------------------------------------------------------------
with tab3:
    st.header("🛠️ Image Editor Mode (Object Removal via OpenCV)")
    st.write("Use your mouse or touchscreen to paint over unwanted objects. Inpainting runs locally without Gemini API.")

    img_upl_edit = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img_upl_tab3")

    if img_upl_edit:
        image = Image.open(io.BytesIO(_uploaded_file_to_bytes(img_upl_edit))).convert("RGB")
        orig_w, orig_h = image.size

        st.write("🎨 Draw over the regions to remove below:")
        max_canvas_w = 800
        scale = min(max_canvas_w / orig_w, 1.0)
        canvas_w = int(orig_w * scale)
        canvas_h = int(orig_h * scale)

        canvas = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=25,
            stroke_color="#FFFFFF",
            background_image=image.resize((canvas_w, canvas_h)),
            height=canvas_h,
            width=canvas_w,
            drawing_mode="freedraw",
            key="inpaint_canvas",
        )

        if st.button("🧽 Remove painted areas"):
            if canvas.image_data is not None:
                alpha = canvas.image_data[:, :, 3]
                painted = (alpha > 0).astype(np.uint8) * 255
                painted_resized = cv2.resize(painted, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask_img = Image.fromarray(painted_resized, mode="L")

                result = inpaint_with_mask(image, mask_img)
                st.image(result, caption="Inpainted Result", use_column_width=True)

                buf = io.BytesIO()
                result.save(buf, format="PNG")
                st.download_button("⬇️ Download result", buf.getvalue(), "inpainted.png", "image/png")
            else:
                st.warning("Please paint over the areas to remove first.")
    else:
        st.info("Upload an image to start editing.")

# ---------------------------------------------------------------------------
# 🎙️ AUDIO MODE
# ---------------------------------------------------------------------------
with tab4:
    st.header("🎙️ Audio Analysis and Transcription Mode")
    st.write("Upload an audio clip and ask Gemini to identify, transcribe, or summarize it.")

    audio_upl = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"], key="audio_upl_tab4")
    if audio_upl:
        st.audio(audio_upl, format=audio_upl.type or "audio/mpeg")

    prompt_audio = st.text_area(
        "Your prompt for the audio",
        placeholder="Examples: 'Transcribe this audio', 'Translate to English', 'Summarize the conversation'.",
        height=100,
    )

    if st.button("🎧 Ask Gemini about the audio", disabled=audio_upl is None):
        if not API_KEY:
            st.error("GEMINI_API_KEY missing.")
        elif not prompt_audio.strip():
            st.warning("Please enter a prompt for the audio.")
        else:
            with st.spinner("Processing audio with Gemini..."):
                audio_bytes = _uploaded_file_to_bytes(audio_upl)
                answer = gemini_generate(
                    prompt_audio,
                    files=[{"mime_type": audio_upl.type or "audio/mpeg", "data_bytes": audio_bytes}],
                )
            st.success(answer)

# ---------------------------------------------------------------------------
# 💬 STANDALONE CHAT MODE
# ---------------------------------------------------------------------------
with tab5:
    st.header("💬 Chat Mode (Standalone Conversation)")
    st.write("A general-purpose chat mode for open-ended dialogue without PDF or file context.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["text"])

    user_chat = st.chat_input("Say something to Gemini...")
    if user_chat:
        st.session_state.chat_history.append({"role": "user", "text": user_chat})
        with st.spinner("Gemini is thinking..."):
            answer = gemini_generate(user_chat)
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
        st.chat_message("assistant").write(answer)

# ---------------------------------------------------------------------------
# ℹ️ FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Notes: • Gemini analyzes text, images, and audio but does not modify them. "
    "• The Image Editor uses local CPU inpainting for privacy and speed. "
    "• Responses are refined to avoid redundant 'Part 1:' or headings."
)
