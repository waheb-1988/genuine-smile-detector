# app.py
import os
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from smile.smile import AnalysisResult

# ---------- Page + CSS ----------
st.set_page_config(page_title="DeepFace Smile Authenticity", page_icon="😁", layout="centered")
st.title("DeepFace-Powered Smile Authenticity Rater")
st.caption("Camera or image upload → genuine-smile score with DeepFace + eye-crinkle proxy. Auto-saves image + JSON.")

st.markdown("""
<style>
.big-emoji { font-size: 80px; line-height: 1; display: inline-block; animation: bounce 1.2s infinite; }
.big-score { font-size: 36px; font-weight: 800; margin-left: 14px; display: inline-block; }
@keyframes bounce {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  30% { transform: translateY(-8px) rotate(6deg); }
  60% { transform: translateY(2px) rotate(-4deg); }
}
.small-note { opacity: 0.8; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Output dirs ----------
OUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
IMG_DIR = os.path.join(OUT_DIR, "images")
JSON_DIR = os.path.join(OUT_DIR, "json")
BASELINE_PATH = os.path.join(OUT_DIR, "baseline.json")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# ---------- Baseline persistence ----------
def load_baseline() -> list[float]:
    if not os.path.exists(BASELINE_PATH):
        return []
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [float(x) for x in data.get("neutral_samples", [])]
    except Exception:
        return []

def save_baseline(samples: list[float]) -> None:
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"neutral_samples": [float(x) for x in samples]}, f, ensure_ascii=False, indent=2)

# ---------- Analyzer ----------
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")
analyzer = SmileAnalyzer(output_dir=OUT_DIR, detector_backend=DETECTOR_BACKEND)

# hydrate analyzer baseline from disk
for s in load_baseline():
    analyzer.add_neutral_sample(s)

# ---------- State ----------
if "detector_backend" not in st.session_state:
    st.session_state["detector_backend"] = DETECTOR_BACKEND

def neutral_count() -> int:
    return analyzer.neutral_count()

# ---------- UI helpers ----------
def show_big_result(score: int, emoji: str, label: str) -> None:
    st.markdown(
        f'<span class="big-emoji">{emoji}</span>'
        f'<span class="big-score">{score}/100</span>',
        unsafe_allow_html=True,
    )
    st.subheader(label)

def add_neutral_from_image(pil_img: Image.Image) -> None:
    eye_ap, ok = analyzer.compute_eye_aperture(np.array(pil_img))
    if not ok:
        st.warning("No face detected. Try better lighting/framing.")
        return
    analyzer.add_neutral_sample(eye_ap)
    save_baseline(analyzer.export_neutral_samples())
    st.success(f"Neutral baseline sample captured ({neutral_count()} total).")

def analyze_and_auto_save(pil_img: Image.Image, origin_name: str, source_kind: str) -> None:
    try:
        res: AnalysisResult = analyzer.analyze_image(pil_img, origin_name=origin_name, source=source_kind)
    except RuntimeError as e:
        st.error(str(e)); return
    except Exception as e:
        st.error(f"Analysis error: {e}"); return

    # UI
    show_big_result(res.score, res.emoji, res.label)
    with st.expander("Details (JSON preview)"):
        st.json(res.__dict__)

    # Save (image + JSON)
    base = os.path.splitext(origin_name)[0] if origin_name else f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    res.image_filename = f"{base}.jpg"
    img_path = analyzer.save_image(pil_img, base)
    json_path = analyzer.save_json(res, base)
    st.success(f"Saved!\n\n📷 {img_path}\n📄 {json_path}")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    st.session_state["detector_backend"] = st.selectbox(
        "DeepFace detector backend",
        ["retinaface", "opencv", "mtcnn", "ssd"],
        index=["retinaface", "opencv", "mtcnn", "ssd"].index(st.session_state["detector_backend"])
    )
    analyzer.detector_backend = st.session_state["detector_backend"]
    st.markdown(
        f"<div class='small-note'>Baseline file: <code>{BASELINE_PATH}</code><br>"
        f"Samples: <b>{neutral_count()}</b></div>",
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear baseline"):
        analyzer.clear_neutral_samples()
        save_baseline([])
        st.info("Baseline cleared.")

# ---------- Instructions ----------
st.write("**How to use**")
st.markdown("""
1. Capture **2–3 neutral** images (no smile). Baseline auto-saves to disk.  
2. Then capture your **smile** (camera or upload).  
3. The app **auto-saves** the original image and a **JSON** with the same base name under `outputs/`.
""")

# ---------- Modes ----------
mode = st.radio("Choose input mode", ["📷 Camera", "🖼️ Image upload"], horizontal=True)

# Camera mode
if mode.startswith("📷"):
    img_input = st.camera_input("Camera (allow access, then take a snapshot)")
    col1, col2 = st.columns(2)
    with col1:
        btn_neutral = st.button("📷 Capture Neutral")
    with col2:
        btn_smile = st.button("😄 Analyze Smile")

    if img_input is not None:
        frame = Image.open(img_input).convert("RGB")
        fname = f"camera_{int(datetime.now().timestamp())}.jpg"
        if btn_neutral:
            add_neutral_from_image(frame)
        elif btn_smile:
            analyze_and_auto_save(frame, fname, "camera")

# Upload mode
else:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        pil = Image.open(file).convert("RGB")
        st.image(pil, caption="Uploaded image", use_column_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📷 Use as Neutral Baseline"):
                add_neutral_from_image(pil)
        with col2:
            if st.button("😄 Analyze Smile"):
                analyze_and_auto_save(pil, file.name, "upload")
