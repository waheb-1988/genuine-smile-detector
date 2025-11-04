import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import math
import time

# Use built-in webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
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

def deepface_happy_prob(image_bgr):
    try:
        result = DeepFace.analyze(image_bgr, actions=['emotion'], enforce_detection=False)
        res = result[0] if isinstance(result, list) else result
        emo = res.get('emotion', res.get('emotions', {}))
        val = float(emo.get('happy', 0.0))
        return val / 100.0 if val > 1.0 else val
    except Exception as e:
        print("DeepFace error:", e)
        return 0.0

def combine_score(happy_prob, eye_aperture, neutral_baseline=0.23):
    intensity = float(np.clip(happy_prob, 0.0, 1.0))
    delta = float(np.clip(neutral_baseline - eye_aperture, 0.0, 0.25))
    duchenne = delta / 0.25
    combined = 0.6 * intensity + 0.4 * duchenne
    return int(np.clip(combined * 100, 0, 100))

if not cap.isOpened():
    print("❌ Cannot open laptop camera.")
    exit()
else:
    print("✅ Laptop camera connected.")

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
            print("⚠️ No frame from camera.")
            break

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
            label = "Genuine smile 😁"
        elif score >= 70:
            label = "Probably genuine 😄"
        elif score >= 55:
            label = "Uncertain smile 🙂"
        else:
            label = "Weak/fake smile 🙃"

        cv2.putText(frame, f"{label} ({score}/100)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("😊 Genuine Smile Detector (Laptop Camera)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
