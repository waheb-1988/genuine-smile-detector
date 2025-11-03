import os
import io
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

# DeepFace for emotion
from deepface import DeepFace

# MediaPipe for landmarks
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

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
OUT_DIR = "outputs"
IMG_DIR = os.path.join(OUT_DIR, "images")
JSON_DIR = os.path.join(OUT_DIR, "json")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

BASELINE_PATH = os.path.join(OUT_DIR, "baseline.json")

# ---------- Utils ----------
def _dist(p1, p2):
    return math.dist(p1, p2)

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _timestamp_base(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def load_baseline():
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(map(float, data.get("neutral_samples", [])))
        except Exception:
            return []
    return []

def save_baseline(samples):
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"neutral_samples": list(map(float, samples))}, f, ensure_ascii=False, indent=2)

# ---------- Your original logic (with small hardening) ----------
def compute_eye_aperture_ratios(landmarks, w, h):
    # Left eye
    L_UP, L_LOW, L_IN, L_OUT = 386, 374, 133, 33
    # Right eye
    R_UP, R_LOW, R_IN, R_OUT = 159, 145, 362, 263

    def px(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    l_up, l_low, l_in, l_out = px(L_UP), px(L_LOW), px(L_IN), px(L_OUT)
    r_up, r_low, r_in, r_out = px(R_UP), px(R_LOW), px(R_IN), px(R_OUT)

    left_width = _dist(l_in, l_out) + 1e-6
    right_width = _dist(r_in, r_out) + 1e-6

    left_ap = _dist(l_up, l_low) / left_width
    right_ap = _dist(r_up, r_low) / right_width
    return left_ap, right_ap

def mediapipe_eye_aperture(image_rgb):
    """Return (eye_avg_aperture, ok) using MediaPipe FaceMesh on an RGB image."""
    h, w = image_rgb.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as fm:
        res = fm.process(image_rgb)
        if not res.multi_face_landmarks:
            return None, False
        lm = res.multi_face_landmarks[0].landmark
        l, r = compute_eye_aperture_ratios(lm, w, h)
        return float((l + r) / 2.0), True

def deepface_happy_prob(image_bgr):
    """
    Return happiness probability in [0,1], resilient to DeepFace version differences.
    """
    try:
        out = DeepFace.analyze(
            img_path=image_bgr,
            actions=["emotion"],
            detector_backend=st.session_state.get("detector_backend", "retinaface"),
            enforce_detection=True
        )
    except TypeError:
        # Older DeepFace version may not accept all params:
        out = DeepFace.analyze(img_path=image_bgr, actions=["emotion"])
    except Exception:
        # Last resort: try a different backend and allow missing face
        out = DeepFace.analyze(
            img_path=image_bgr, actions=["emotion"], detector_backend="opencv", enforce_detection=False
        )

    res = out[0] if isinstance(out, list) else out
    emo = res.get("emotion") or res.get("emotions") or {}
    val = float(emo.get("happy", 0.0))
    return val / 100.0 if val > 1.0 else val

def combine_score(happy_prob, eye_aperture, neutral_eye_baseline):
    """
    Blend DeepFace smile strength with AU6 proxy (eye narrowing relative to personal baseline).
    """
    intensity = float(np.clip(happy_prob, 0.0, 1.0))
    duchenne = 0.0
    if neutral_eye_baseline is not None:
        delta = float(np.clip(neutral_eye_baseline - eye_aperture, 0.0, 0.25))
        duchenne = delta / 0.25  # 0..1

    combined = 0.6 * intensity + 0.4 * duchenne
    score = int(np.clip(combined * 100, 0, 100))

    if score >= 85:
        return score, "Genuine smile! Legendary grin!", "🤩😁"
    elif score >= 70:
        return score, "Genuine smile likely", "😁"
    elif score >= 55:
        return score, "Nice smile, not sure it’s fully Duchenne", "😊🙂"
    elif score >= 45:
        return score, "Smile detected (uncertain authenticity)", "🙂🤔"
    else:
        return score, "Posed/weak smile likely", "🙃🫥"

def save_outputs(base_name, pil_img, payload):
    img_path = os.path.join(IMG_DIR, f"{base_name}.jpg")
    json_path = os.path.join(JSON_DIR, f"{base_name}.json")
    pil_img.save(img_path, format="JPEG", quality=95)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return img_path, json_path

# ---------- State ----------
if "detector_backend" not in st.session_state:
    st.session_state["detector_backend"] = "retinaface"

if "neutral_samples" not in st.session_state:
    st.session_state["neutral_samples"] = load_baseline()  # <-- persist/load
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_image" not in st.session_state:
    st.session_state["last_image"] = None

# ---------- UI helpers ----------
def show_big_result(score, emoji, label):
    st.markdown(
        f'<span class="big-emoji">{emoji}</span>'
        f'<span class="big-score">{score}/100</span>',
        unsafe_allow_html=True,
    )
    st.subheader(label)

def neutral_count():
    return len(st.session_state["neutral_samples"])

def add_neutral_from_image(pil_img):
    img_rgb = np.array(pil_img)
    eye_ap, ok = mediapipe_eye_aperture(img_rgb)
    if not ok:
        st.warning("No face detected. Try better lighting/framing.")
        return
    st.session_state["neutral_samples"].append(float(eye_ap))
    save_baseline(st.session_state["neutral_samples"])  # <-- persist immediately
    st.success(f"Neutral baseline sample captured ({neutral_count()} total).")

def analyze_and_auto_save(pil_img, origin_name, source_kind):
    """
    Analyze image, display big result, and ALWAYS save image+JSON automatically.
    """
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Eye aperture
    eye_ap, ok = mediapipe_eye_aperture(img_rgb)
    if not ok:
        st.error("No face detected. Try better lighting/framing.")
        return

    # DeepFace happiness
    happy = deepface_happy_prob(img_bgr)

    # Baseline (median of at least 2)
    baseline = None
    if neutral_count() >= 2:
        baseline = float(np.median(st.session_state["neutral_samples"]))

    # Score
    score, label, emoji = combine_score(happy, eye_ap, baseline)

    # Display
    show_big_result(score, emoji, label)

    payload = {
        "timestamp_utc": _now_iso(),
        "source": source_kind,
        "image_filename": origin_name,
        "deepface_happy_prob": round(float(happy), 4),
        "eye_aperture": round(float(eye_ap), 4),
        "neutral_eye_baseline": None if baseline is None else round(float(baseline), 4),
        "score": score,
        "label": label,
        "emoji": emoji,
    }

    # Autosave with stable basename
    base = os.path.splitext(origin_name)[0] if origin_name else _timestamp_base("capture")
    img_path, json_path = save_outputs(base, pil_img, payload)

    # Keep references for this run
    st.session_state["last_result"] = payload
    st.session_state["last_image"] = pil_img

    st.success(f"Saved!\n\n📷 {img_path}\n📄 {json_path}")

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Settings")
    st.session_state["detector_backend"] = st.selectbox(
        "DeepFace detector backend",
        ["retinaface", "opencv", "mtcnn", "ssd"],
        index=["retinaface", "opencv", "mtcnn", "ssd"].index(st.session_state["detector_backend"])
    )
    st.markdown(
        f"<div class='small-note'>Neutral samples stored at: <code>{BASELINE_PATH}</code><br>"
        f"Count: <b>{neutral_count()}</b></div>",
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear baseline"):
        st.session_state["neutral_samples"] = []
        save_baseline([])
        st.info("Baseline cleared.")

# ---------- How to use ----------
st.write("**How to use**")
st.markdown("""
1. Capture **2–3 neutral** images first (no smile). Your baseline is saved to disk automatically.  
2. Then capture your **smile** (camera or upload).  
3. The app **auto-saves** the original image and a **JSON** with the same base name under `outputs/`.
""")

# ---------- Mode switch ----------
mode = st.radio("Choose input mode", ["📷 Camera", "🖼️ Image upload"], horizontal=True)

# ---------- Camera mode ----------
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

# ---------- Upload mode ----------
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
