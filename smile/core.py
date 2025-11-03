# smile/core.py
import os
import json
import math
import cv2
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from PIL import Image

from deepface import DeepFace
import mediapipe as mp


from smile.models import AnalysisResult

mp_face_mesh = mp.solutions.face_mesh

class SmileAnalyzer:
    """
    DeepFace + MediaPipe pipeline:
      - eye aperture (AU6 proxy)
      - happiness probability (DeepFace)
      - combined 0..100 score
      - persistent outputs/images + outputs/json
      - in-memory neutral baseline (persisted by caller)
    """

    # Landmark indices
    _L_UP, _L_LOW, _L_IN, _L_OUT = 386, 374, 133, 33
    _R_UP, _R_LOW, _R_IN, _R_OUT = 159, 145, 362, 263

    def __init__(self, output_dir: str = "outputs", detector_backend: str = "retinaface"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.json_dir = os.path.join(output_dir, "json")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        self.detector_backend = detector_backend
        self._neutral_samples: list[float] = []

    # ---------- utils ----------
    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.dist(p1, p2)

    # ---------- baseline ----------
    def add_neutral_sample(self, eye_aperture: float) -> None:
        self._neutral_samples.append(float(eye_aperture))

    def clear_neutral_samples(self) -> None:
        self._neutral_samples = []

    def export_neutral_samples(self) -> list[float]:
        return list(self._neutral_samples)

    def neutral_count(self) -> int:
        return len(self._neutral_samples)

    def get_neutral_baseline(self) -> Optional[float]:
        if self.neutral_count() >= 2:
            return float(np.median(self._neutral_samples))
        return None

    # ---------- measurements ----------
    def _compute_eye_aperture_ratios(self, landmarks, w: int, h: int) -> Tuple[float, float]:
        def px(idx): return (landmarks[idx].x * w, landmarks[idx].y * h)
        l_up, l_low, l_in, l_out = px(self._L_UP), px(self._L_LOW), px(self._L_IN), px(self._L_OUT)
        r_up, r_low, r_in, r_out = px(self._R_UP), px(self._R_LOW), px(self._R_IN), px(self._R_OUT)
        left_width = self._dist(l_in, l_out) + 1e-6
        right_width = self._dist(r_in, r_out) + 1e-6
        left_ap = self._dist(l_up, l_low) / left_width
        right_ap = self._dist(r_up, r_low) / right_width
        return left_ap, right_ap

    def compute_eye_aperture(self, image_rgb: np.ndarray) -> Tuple[Optional[float], bool]:
        """Return (avg_eye_aperture, ok)."""
        h, w = image_rgb.shape[:2]
        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5) as fm:
            res = fm.process(image_rgb)
            if not res.multi_face_landmarks:
                return None, False
            lm = res.multi_face_landmarks[0].landmark
            l, r = self._compute_eye_aperture_ratios(lm, w, h)
            return float((l + r) / 2.0), True

    # ---------- deepface ----------
    def deepface_happy_prob(self, image_bgr: np.ndarray) -> float:
        """Return happiness probability in [0,1], tolerant to API changes."""
        try:
            out = DeepFace.analyze(
                img_path=image_bgr,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
        except TypeError:
            out = DeepFace.analyze(img_path=image_bgr, actions=["emotion"])
        except Exception:
            out = DeepFace.analyze(img_path=image_bgr, actions=["emotion"], detector_backend="opencv", enforce_detection=False)

        res = out[0] if isinstance(out, list) else out
        emo = res.get("emotion") or res.get("emotions") or {}
        val = float(emo.get("happy", 0.0))
        return val/100.0 if val > 1.0 else val

    # ---------- scoring ----------
    @staticmethod
    def _combine_score(happy_prob: float, eye_aperture: float, neutral_eye_baseline: Optional[float]) -> tuple[int, str, str]:
        intensity = float(np.clip(happy_prob, 0.0, 1.0))
        duchenne = 0.0
        if neutral_eye_baseline is not None:
            delta = float(np.clip(neutral_eye_baseline - eye_aperture, 0.0, 0.25))
            duchenne = delta / 0.25
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

    # ---------- pipeline ----------
    def analyze_image(self, pil_img: Image.Image, origin_name: str, source: str) -> AnalysisResult:
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        eye_ap, ok = self.compute_eye_aperture(img_rgb)
        if not ok:
            raise RuntimeError("No face detected. Try better lighting/framing.")
        happy = self.deepface_happy_prob(img_bgr)
        baseline = self.get_neutral_baseline()
        score, label, emoji = self._combine_score(happy, eye_ap, baseline)

        return AnalysisResult(
            timestamp_utc=self._now_iso(),
            source=source,
            image_filename=origin_name,
            deepface_happy_prob=round(float(happy), 4),
            eye_aperture=round(float(eye_ap), 4),
            neutral_eye_baseline=None if baseline is None else round(float(baseline), 4),
            score=score,
            label=label,
            emoji=emoji
        )

    # ---------- saving ----------
    def save_image(self, pil_img: Image.Image, base_name: str) -> str:
        path = os.path.join(self.images_dir, f"{base_name}.jpg")
        pil_img.save(path, format="JPEG", quality=95)
        return path

    def save_json(self, result: AnalysisResult, base_name: str) -> str:
        path = os.path.join(self.json_dir, f"{base_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, indent=2)
        return path
