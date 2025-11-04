import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import math
import time
import yaml
from PIL import ImageFont, ImageDraw, Image
import inspect
import os
from datetime import datetime

# ---------- LOAD CONFIG ----------
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

IRIUN_URL = config.get("camera_url", "http://172.20.10.2:8080/video")
WINDOW_SIZE = tuple(config.get("window_size", [960, 540]))
EMOTION_MODEL = config.get("emotion_model", "Facenet512")
SAVE_INTERVAL = 15  # seconds between saved clips
SAVE_DIR = "saved_clips"

print(f"📸 Using camera stream: {IRIUN_URL}")
print(f"🧠 Emotion model: {EMOTION_MODEL}")
print(f"🪟 Window size: {WINDOW_SIZE}")

# ---------- PREPARE OUTPUT DIRECTORY ----------
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- FONT (Emoji Support) ----------
try:
    FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"  # Windows emoji font
    emoji_font = ImageFont.truetype(FONT_PATH, 36)
except Exception:
    emoji_font = ImageFont.truetype("arial.ttf", 36)

# ---------- FACE MESH ----------
mp_face_mesh = mp.solutions.face_mesh

def _dist(p1, p2):
    return math.dist(p1, p2)

def compute_eye_aperture_ratios(landmarks, w, h):
    L_UP, L_LOW, L_IN, L_OUT = 386, 374, 133, 33
    R_UP, R_LOW, R_IN, R_OUT = 159, 145, 362, 263

    def px(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    l_up, l_low, l_in, l_out = px(L_UP), px(L_LOW), px(L_IN), px(L_OUT)
    r_up, r_low, r_in, r_out = px(R_UP), px(R_LOW), px(R_IN), px(R_OUT)

    left_width = _dist(l_in, l_out) + 1e-6
    right_width = _dist(r_in, r_out) + 1e-6

    left_ap = _dist(l_up, l_low) / left_width
    right_ap = _dist(r_up, r_low) / right_width
    return (left_ap + right_ap) / 2.0

# ---------- DeepFace wrapper ----------
def _call_deepface_analyze(img_np, actions=["emotion"], enforce_detection=False, model_choice=None):
    func = DeepFace.analyze
    sig = inspect.signature(func)
    last_exc = None

    candidates = []
    if 'img_path' in sig.parameters:
        base_kwargs = {'img_path': img_np, 'actions': actions, 'enforce_detection': enforce_detection}
        if model_choice:
            if 'model_name' in sig.parameters:
                kw = dict(base_kwargs); kw['model_name'] = model_choice; candidates.append(kw)
            if 'model' in sig.parameters:
                kw = dict(base_kwargs); kw['model'] = model_choice; candidates.append(kw)
        candidates.append(base_kwargs)

    if True:
        if model_choice:
            if 'model_name' in sig.parameters:
                kw = {'actions': actions, 'enforce_detection': enforce_detection, 'model_name': model_choice}
                candidates.append(('pos', kw))
            if 'model' in sig.parameters:
                kw = {'actions': actions, 'enforce_detection': enforce_detection, 'model': model_choice}
                candidates.append(('pos', kw))
        candidates.append(('pos', {'actions': actions, 'enforce_detection': enforce_detection}))

    for cand in candidates:
        try:
            if isinstance(cand, tuple) and cand[0] == 'pos':
                return func(img_np, **cand[1])
            else:
                return func(**cand)
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            raise
    raise last_exc if last_exc else RuntimeError("DeepFace.analyze failed")

def deepface_happy_prob(image_bgr):
    try:
        res = _call_deepface_analyze(image_bgr, actions=['emotion'], enforce_detection=False, model_choice=EMOTION_MODEL)
        res_item = res[0] if isinstance(res, list) else res
        emo = res_item.get('emotion') or res_item.get('emotions') or {}
        val = float(emo.get('happy', 0.0))
        return val / 100.0 if val > 1.0 else val
    except Exception as e:
        print("DeepFace error (final):", e)
        return 0.0

def combine_score(happy_prob, eye_aperture, neutral_baseline=0.23):
    intensity = float(np.clip(happy_prob, 0.0, 1.0))
    delta = float(np.clip(neutral_baseline - eye_aperture, 0.0, 0.25))
    duchenne = delta / 0.25
    combined = 0.6 * intensity + 0.4 * duchenne
    return int(np.clip(combined * 100, 0, 100))

def draw_text_with_emoji(frame, text, position=(30, 50), color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=emoji_font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- MAIN ----------
cap = cv2.VideoCapture(IRIUN_URL)
if not cap.isOpened():
    print(f"❌ Cannot open stream at {IRIUN_URL}")
    exit()
else:
    print(f"✅ Connected to camera stream at {IRIUN_URL}")

# Initialize video writer variables
record_start_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None

def start_new_recording():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SAVE_DIR, f"smile_{now}.avi")
    print(f"💾 Starting new recording: {filepath}")
    return cv2.VideoWriter(filepath, fourcc, 20.0, WINDOW_SIZE)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    fps_time = time.time()
    video_writer = start_new_recording()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame received — check camera app.")
            break

        frame = cv2.resize(frame, WINDOW_SIZE)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        eye_ap = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            eye_ap = compute_eye_aperture_ratios(lm, w, h)

        happy = deepface_happy_prob(frame)
        score = combine_score(happy, eye_ap if eye_ap else 0.23)

        if score >= 85:
            label = f"Genuine smile 😁 ({score}/100)"; color = (0, 255, 0)
        elif score >= 70:
            label = f"Probably genuine 😄 ({score}/100)"; color = (0, 255, 255)
        elif score >= 55:
            label = f"Uncertain smile 🙂 ({score}/100)"; color = (0, 165, 255)
        else:
            label = f"Weak/fake smile 🙃 ({score}/100)"; color = (0, 0, 255)

        frame = draw_text_with_emoji(frame, label, (30, 40), color)

        # Write frame to video
        if video_writer:
            video_writer.write(frame)

        # Restart recording every SAVE_INTERVAL seconds
        if time.time() - record_start_time >= SAVE_INTERVAL:
            video_writer.release()
            video_writer = start_new_recording()
            record_start_time = time.time()

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("😊 Genuine Smile Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
