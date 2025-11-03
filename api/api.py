# api/api.py
import os
import io
import json
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smile.core import SmileAnalyzer
from smile.models import AnalysisResult, AnalyzeResponse

# ------------ Config & folders ------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline.json")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# ------------ App ------------
app = FastAPI(title="DeepFace Duchenne Rater API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

analyzer = SmileAnalyzer(output_dir=OUTPUT_DIR, detector_backend=DETECTOR_BACKEND)

# ------------ Baseline persistence ------------
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

# hydrate in-memory baseline
for s in load_baseline():
    analyzer.add_neutral_sample(s)

# ------------ Models ------------
class ListResponse(BaseModel):
    images: List[str]
    jsons: List[str]

class BaselineInfo(BaseModel):
    count: int
    samples: List[float]

# ------------ Helpers ------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _basename_from(filename: Optional[str]) -> str:
    if not filename:
        return f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.splitext(os.path.basename(filename))[0]

# ------------ Endpoints ------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": _now_iso(), "output_dir": OUTPUT_DIR, "backend": analyzer.detector_backend}

@app.get("/baseline", response_model=BaselineInfo)
async def get_baseline():
    samples = load_baseline()
    return BaselineInfo(count=len(samples), samples=samples)

@app.delete("/baseline", response_model=BaselineInfo)
async def clear_baseline():
    save_baseline([])
    analyzer.clear_neutral_samples()
    return BaselineInfo(count=0, samples=[])

@app.post("/baseline/add", response_model=BaselineInfo)
async def add_neutral_sample(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = np.array(pil)
    aperture, ok = analyzer.compute_eye_aperture(img_rgb)
    if not ok:
        raise HTTPException(status_code=422, detail="No face detected")

    analyzer.add_neutral_sample(float(aperture))
    save_baseline(analyzer.export_neutral_samples())

    samples = analyzer.export_neutral_samples()
    return BaselineInfo(count=len(samples), samples=samples)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), source: str = "api"):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    basename = _basename_from(file.filename)
    origin_name = f"{basename}.jpg"

    try:
        res: AnalysisResult = analyzer.analyze_image(pil, origin_name=origin_name, source=source)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    # align filename + save
    res.image_filename = origin_name
    analyzer.save_image(pil, basename)
    analyzer.save_json(res, basename)

    return AnalyzeResponse(**res.__dict__)

@app.get("/list", response_model=ListResponse)
async def list_outputs():
    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])
    jsns = sorted([f for f in os.listdir(JSON_DIR) if f.lower().endswith(".json")])
    return ListResponse(images=imgs, jsons=jsns)

@app.get("/results/{basename}")
async def get_result(basename: str):
    path = os.path.join(JSON_DIR, f"{basename}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path, media_type="application/json", filename=f"{basename}.json")

@app.get("/images/{basename}")
async def get_image(basename: str):
    path = os.path.join(IMG_DIR, f"{basename}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"{basename}.jpg")
