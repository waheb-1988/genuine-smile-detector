import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import math
import time
import yaml
from PIL import ImageFont, ImageDraw, Image
import inspect

# ---------- LOAD CONFIG (simple inline defaults if no YAML) ----------
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}
IRIUN_URL = config.get("camera_url", "http://172.20.10.2:8080/video")
WINDOW_SIZE = tuple(config.get("window_size", [960, 540]))
EMOTION_MODEL = config.get("emotion_model", "Facenet512")

print(f"📸 Using camera stream: {IRIUN_URL}")
print(f"🧠 Emotion model: {EMOTION_MODEL}")
print(f"🪟 Window size: {WINDOW_SIZE}")

# ---------- FONT (Emoji Support) ----------
try:
    FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"  # Windows emoji font
    emoji_font = ImageFont.truetype(FONT_PATH, 36)
except Exception:
    # fallback
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

# ---------- Robust DeepFace analyzer wrapper ----------
def _call_deepface_analyze(img_np, actions=["emotion"], enforce_detection=False, model_choice=None):
    """
    Try multiple calling patterns for DeepFace.analyze to handle different DeepFace versions:
      1) analyze(img_path=img_np, actions=..., enforce_detection=..., model_name=...)
      2) analyze(img_path=img_np, actions=..., enforce_detection=..., model=...)
      3) analyze(img_np, actions=..., enforce_detection=..., model_name=...)
      4) analyze(img_np, actions=..., enforce_detection=..., model=...)
      5) analyze(img_np, actions=..., enforce_detection=...)  (no model arg)
    Returns the analyze() result or raises the last exception.
    """
    func = DeepFace.analyze
    sig = inspect.signature(func)
    last_exc = None

    # Build possible kwargs variants depending on signature
    candidates = []

    # prefer named img_path if present
    if 'img_path' in sig.parameters:
        base_kwargs = {'img_path': img_np, 'actions': actions, 'enforce_detection': enforce_detection}
        if model_choice is not None:
            if 'model_name' in sig.parameters:
                kw = dict(base_kwargs); kw['model_name'] = model_choice; candidates.append(kw)
            if 'model' in sig.parameters:
                kw = dict(base_kwargs); kw['model'] = model_choice; candidates.append(kw)
        candidates.append(base_kwargs)
    # positional numpy array first argument
    if True:
        if model_choice is not None:
            if 'model_name' in sig.parameters:
                kw = {'actions': actions, 'enforce_detection': enforce_detection, 'model_name': model_choice}
                candidates.append(('pos', kw))
            if 'model' in sig.parameters:
                kw = {'actions': actions, 'enforce_detection': enforce_detection, 'model': model_choice}
                candidates.append(('pos', kw))
        candidates.append(('pos', {'actions': actions, 'enforce_detection': enforce_detection}))

    # Attempt candidates sequentially
    for cand in candidates:
        try:
            if isinstance(cand, tuple) and cand[0] == 'pos':
                # positional first arg
                kwargs = cand[1]
                return func(img_np, **kwargs)
            else:
                return func(**cand)
        except TypeError as e:
            last_exc = e
            # signature mismatch; try next
            continue
        except Exception as e:
            # other runtime error (model files missing etc.) — surface it
            raise

    # If reached here, nothing worked — raise the last TypeError for debugging
    raise last_exc if last_exc is not None else RuntimeError("DeepFace.analyze failed without exception")

def deepface_happy_prob(image_bgr):
    """
    Return happiness probability in [0,1] using robust DeepFace call attempts.
    """
    try:
        res = _call_deepface_analyze(image_bgr, actions=['emotion'], enforce_detection=False, model_choice=EMOTION_MODEL)
        # normalize result shape
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

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    fps_time = time.time()
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

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("😊 Genuine Smile Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
