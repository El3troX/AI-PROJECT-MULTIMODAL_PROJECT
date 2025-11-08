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
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------------------
# ⚙️ SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="OmniGen", page_icon="🤖", layout="wide")
st.title("🤖 OmniGen – Gemini Multimodal AI Assistant")

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.warning("⚠️ Please set GEMINI_API_KEY in .env or Streamlit secrets.")


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
    try:
        model = genai.GenerativeModel(model_name)
        parts = [{"text": prompt}]
        if files:
            for f in files:
                parts.append(_to_inline_part(f["data_bytes"], f["mime_type"]))
        resp = model.generate_content({"parts": parts})
        return (resp.text or "").strip() or "No response received."
    except Exception as e:
        return f"Gemini service unavailable.\n\nDetails: {e}"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text)
    except Exception as e:
        return f"[PDF read error: {e}]"


# ---------------------------------------------------------------------------
# 🧠 PURE PYTHON IMAGE INPAINT (NO OPENCV)
# ---------------------------------------------------------------------------
def load_image(file) -> Image.Image:
    img = Image.open(file).convert("RGB")
    return img


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_for_speed(img: Image.Image, max_side=1024):
    w, h = img.size
    if max(w, h) <= max_side:
        return img, 1.0
    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS), scale


def upscale_back(img_small: Image.Image, scale: float, original_size):
    if scale == 1.0:
        return img_small
    return img_small.resize(original_size, Image.LANCZOS)


def canvas_mask_to_bool(mask_rgba: np.ndarray) -> np.ndarray:
    if mask_rgba is None:
        return None
    alpha = mask_rgba[..., 3]
    return alpha > 0


def diffuse_inpaint(image: np.ndarray, mask: np.ndarray, iters: int = 400) -> np.ndarray:
    """Simple NumPy diffusion-based inpainting."""
    img = image.astype(np.float32)
    m = mask.astype(bool)

    if not np.any(m):
        return image

    known = ~m
    if np.any(known):
        img[m] = img[known].mean(axis=0)

    for _ in range(iters):
        up    = np.roll(img, -1, axis=0)
        down  = np.roll(img,  1, axis=0)
        left  = np.roll(img, -1, axis=1)
        right = np.roll(img,  1, axis=1)
        avg = (up + down + left + right) / 4.0
        img[m] = avg[m]

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 💾 SESSION STATE
# ---------------------------------------------------------------------------
st.session_state.setdefault("text_history", [])
st.session_state.setdefault("pdf_context", "")
st.session_state.setdefault("chat_history", [])


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
# 📘 TAB 1 — TEXT + PDF CHAT **(chat_input moved outside tab)**
# ---------------------------------------------------------------------------
with tab1:
    st.header("📘 Text / PDF Chat Mode")
    st.write("Upload a PDF and chat with Gemini about its contents.")

    with st.expander("📎 Upload a PDF (optional)"):
        pdf_upl = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_tab1")
        if pdf_upl:
            pdf_bytes = _uploaded_file_to_bytes(pdf_upl)
            text = extract_pdf_text(pdf_bytes)
            st.session_state.pdf_context = text[:200000]   # trim for context safety
            st.success("✅ PDF loaded.")
            st.text_area("Preview", text[:3000], height=200)

    if st.session_state.text_history:
        st.markdown("### Conversation")
    for turn in st.session_state.text_history:
        st.chat_message(turn["role"]).write(turn["text"])


# ---------------------------------------------------------------------------
# 🖼️ TAB 2 — IMAGE Q&A
# ---------------------------------------------------------------------------
with tab2:
    st.header("🖼️ Image Q&A Mode")

    img_upl = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img_tab2")
    prompt_img = st.text_area("Prompt for Gemini:", height=100)

    if img_upl:
        st.image(img_upl, use_column_width=True)

    if st.button("🔍 Ask Gemini", disabled=not img_upl):
        if not prompt_img.strip():
            st.warning("Enter a prompt.")
        else:
            img_bytes = _uploaded_file_to_bytes(img_upl)
            answer = gemini_generate(
                prompt_img,
                files=[{"mime_type": img_upl.type, "data_bytes": img_bytes}]
            )
            st.success(answer)


# ---------------------------------------------------------------------------
# 🛠️ TAB 3 — PURE PYTHON IMAGE EDITOR
# ---------------------------------------------------------------------------
with tab3:
    st.header("🛠️ Image Editor Mode (Pure Python Diffusion)")

    uploaded = st.file_uploader("Upload image", ["png", "jpg", "jpeg"], key="img_tab3")
    col1, col2 = st.columns(2)
    with col1:
        iters = st.slider("Iterations", 100, 1500, 500)
        brush = st.slider("Brush size", 5, 80, 30)
    with col2:
        max_side = st.slider("Max size", 512, 2048, 1024)
        show_mask = st.checkbox("Preview mask")

    if uploaded:
        img = load_image(uploaded)
        orig_size = img.size
        small_img, scale = resize_for_speed(img, max_side)
        small_np = pil_to_np(small_img)

        st.subheader("Paint over areas to remove")

        canvas = st_canvas(
            fill_color="rgba(255,255,255,0.7)",
            stroke_width=brush,
            stroke_color="#FFFFFF",
            background_image=small_img,
            height=small_img.height,
            width=small_img.width,
            drawing_mode="freedraw",
            key="canvas_editor",
        )

        mask_bool = None
        if canvas.image_data is not None:
            mask_bool = canvas_mask_to_bool(canvas.image_data)

        if show_mask and mask_bool is not None:
            st.image((mask_bool * 255).astype(np.uint8), caption="Mask", use_column_width=True)

        if st.button("🪄 Inpaint"):
            if mask_bool is None or not np.any(mask_bool):
                st.warning("Draw a mask first.")
            else:
                result_small = diffuse_inpaint(small_np, mask_bool, iters)
                result_pil_small = np_to_pil(result_small)
                result_final = upscale_back(result_pil_small, scale, orig_size)
                st.image(result_final, caption="Inpainted result", use_column_width=True)


# ---------------------------------------------------------------------------
# 🎙️ TAB 4 — AUDIO MODE
# ---------------------------------------------------------------------------
with tab4:
    st.header("🎙️ Audio Analysis Mode")

    audio_upl = st.file_uploader("Upload audio", ["mp3", "wav", "m4a"], key="aud_tab4")
    if audio_upl:
        st.audio(audio_upl)

    prompt_audio = st.text_area("Prompt for audio:", height=100)

    if st.button("🎧 Ask Gemini", disabled=not audio_upl):
        if not prompt_audio.strip():
            st.warning("Enter a prompt.")
        else:
            audio_bytes = _uploaded_file_to_bytes(audio_upl)
            answer = gemini_generate(
                prompt_audio,
                files=[{"mime_type": audio_upl.type, "data_bytes": audio_bytes}]
            )
            st.success(answer)


# ---------------------------------------------------------------------------
# 💬 TAB 5 — STANDALONE CHAT
# ---------------------------------------------------------------------------
with tab5:
    st.header("💬 Chat Mode")
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["text"])


# ---------------------------------------------------------------------------
# ✅ GLOBAL CHAT INPUT (required for Streamlit 1.24)
# ---------------------------------------------------------------------------
user_msg_global = st.chat_input("Type your message for Gemini...")

if user_msg_global:
    # If user is inside PDF/Text tab → store in text_history
    active_tab = st.session_state.get("_selected_tab", "")

    # Always treat this prompt as "general text" chat
    st.session_state.text_history.append({"role": "user", "text": user_msg_global})

    pdf_context = st.session_state.pdf_context
    system = (
        "You are a concise assistant. "
        "Avoid headings unless requested."
    )

    prompt = (
        f"{system}\n\nPDF Context:\n{pdf_context}\n\nUser: {user_msg_global}"
        if pdf_context else f"{system}\n\nUser: {user_msg_global}"
    )

    with st.spinner("Gemini is thinking..."):
        answer = gemini_generate(prompt)

    st.session_state.text_history.append({"role": "assistant", "text": answer})
    st.chat_message("assistant").write(answer)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "• Gemini analyzes text, images, and audio.\n"
    "• Image Editor uses pure NumPy diffusion.\n"
    "• `st.chat_input` is placed outside tabs for Streamlit 1.24 compatibility."
)
