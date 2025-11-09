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

# ---------------------------------------------------------------------------
# 🧠 IMAGE EDITOR HELPERS (PURE PYTHON)
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
    """Pure NumPy diffusion-based inpainting."""
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

# ---------------------------------------------------------------------------
# 🖼️ IMAGE Q&A MODE
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

# ---------------------------------------------------------------------------
# 🛠️ PURE PYTHON IMAGE EDITOR MODE (FIXED BACKGROUND VISIBILITY)
# ---------------------------------------------------------------------------
with tab3:
    st.header("🛠️ Image Editor Mode (Pure Python Diffusion Inpainting)")
    st.write("Remove objects by painting over them — runs locally with NumPy (no GPU/OpenCV).")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key="img_upl_tab3")

    col1, col2 = st.columns(2)
    with col1:
        iters = st.slider("Diffusion iterations (more = smoother, slower)", 100, 1500, 500, 50)
        brush = st.slider("Brush size (px)", 5, 80, 30, 1)
    with col2:
        max_side = st.slider("Max working size (px)", 512, 2048, 1024, 128)
        eraser = st.checkbox("Eraser mode", False)
        show_mask = st.checkbox("Preview drawn mask", False)

    if uploaded:
        # Load image
        img = load_image(uploaded)
        orig_size = img.size
        small_img, scale = resize_for_speed(img, max_side=max_side)
        small_np = pil_to_np(small_img)

        # ✅ Convert to RGBA NumPy (uint8) for reliable display
        small_img = small_img.convert("RGBA")
        bg_array = np.asarray(small_img, dtype=np.uint8)

        # ✅ Wrapper class to bypass ambiguous truth-value check
        class SafeImage:
            def __init__(self, arr):
                self.arr = arr
            def __bool__(self):  # prevents ValueError
                return True
            def __array__(self):
                return self.arr
        safe_bg = SafeImage(bg_array)

        st.subheader("1) Paint the area to remove")
        st.caption("Use the brush to mark unwanted areas; toggle *Eraser mode* to correct mistakes.")

        height, width = bg_array.shape[:2]
        canvas_res = st_canvas(
            fill_color="rgba(255,255,255,0.7)",
            stroke_width=brush,
            stroke_color="#ffffff",
            background_image=safe_bg,  # ✅ Wrapped NumPy array
            height=height,
            width=width,
            drawing_mode="freedraw" if not eraser else "transform",
            update_streamlit=True,
            key="mask_canvas",
        )

        mask_bool = None
        if canvas_res and canvas_res.image_data is not None:
            mask_bool = canvas_mask_to_bool(canvas_res.image_data)

        if show_mask and mask_bool is not None:
            st.image((mask_bool * 255).astype(np.uint8),
                     caption="Current mask (white = masked)",
                     use_column_width=True)

        run = st.button("🪄 Inpaint (Pure Python)")
        if run:
            if mask_bool is None or not np.any(mask_bool):
                st.warning("Please draw a mask over the object you want to remove.")
            else:
                with st.spinner("Running diffusion inpainting..."):
                    result_small = diffuse_inpaint(small_np, mask_bool, iters=iters)
                    result_pil_small = np_to_pil(result_small)
                    result_pil = upscale_back(result_pil_small, scale, orig_size)

                st.success("✅ Done! Object removed successfully.")
                st.image(result_pil, caption="Inpainted result", use_column_width=True)

                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download PNG",
                    data=buf.getvalue(),
                    file_name="inpaint_result.png",
                    mime="image/png",
                )
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
    "• The Image Editor uses pure NumPy diffusion for privacy and lightweight CPU operation. "
    "• Responses are refined to avoid redundant 'Part 1:' or headings."
)



