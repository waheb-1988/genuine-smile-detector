# DeepFace Duchenne Rater

Rates smile authenticity (genuine/Duchenne vs. posed) using:
- **DeepFace** for happiness probability (AU12 proxy),
- **MediaPipe** FaceMesh eye-aperture (AU6 proxy),
- A combined 0–100 score + a fun emoji verdict.

## Features
- Streamlit **app** with **camera** or **image upload**
- **Save** original image + **JSON** (same basename)
- **FastAPI** endpoint to analyze via HTTP
- Pinned requirements for Windows stability

## Quickstart

### 1) Create env (Python 3.10 recommended on Windows)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 2) Configure

###Create .env (or copy from this repo):
```bash
OUTPUT_DIR=outputs
DETECTOR_BACKEND=retinaface
```

### 3) Run the Streamlit app
```bash
streamlit run app/app.py
```
#### 4.1) Run the API
```bash
uvicorn api.main:app --reload


Open docs at: http://127.0.0.1:8000/docs

API endpoints

POST /analyze — multipart file (image). Saves image and JSON; returns analysis.

GET /list — lists saved images/ and json/.

GET /results/{basename} — download a JSON by base filename.
```
#### 4.2) Run the APIOutput format

    Saved under:
```bash
outputs/images/<name>.jpg

outputs/json/<name>.json

Example JSON:

{
  "timestamp_utc": "2025-11-03T08:45:00Z",
  "source": "upload",
  "image_filename": "mypic.jpg",
  "deepface_happy_prob": 0.83,
  "eye_aperture": 0.19,
  "neutral_eye_baseline": 0.22,
  "score": 78,
  "label": "Genuine smile likely",
  "emoji": "😁"
}

```

#### 5)How it works (methods & classes)
smile.core.SmileAnalyzer

Purpose: Orchestrates the full analysis and saving.

__init__(output_dir, detector_backend)
Sets output directories and DeepFace backend.

compute_eye_aperture(image_rgb) -> (aperture, ok)
Uses MediaPipe FaceMesh to measure normalized eyelid aperture (proxy for AU6).
Returns average of left/right eye and a boolean flag.

deepface_happy_prob(image_bgr) -> float
Runs DeepFace emotion analysis and extracts happy probability in [0,1].
Compatible with versions that return % or 0..1.

combine_score(happy_prob, eye_aperture, neutral_eye_baseline) -> (score, label, emoji)
Blends smile intensity (DeepFace) and eye narrowing vs. a personal baseline to infer genuineness.

add_neutral_sample(eye_aperture) / get_neutral_baseline()
Accumulates neutral captures (need ≥ 2) and returns median baseline.

analyze_image(pil_img, origin_name, source) -> AnalysisResult
End-to-end pipeline returning a structured result (not saved yet).

save_image(pil_img, base_name) / save_json(result, base_name)
Persists the original image and the analysis JSON under outputs/.

smile.models.AnalyzeResponse

Pydantic model mirroring the JSON schema returned by the API.

Tips & Troubleshooting

MediaPipe on Windows: prefer Python 3.10/3.11.

If DeepFace complains about prog_bar, this code doesn’t pass it.

For headless servers, use opencv-python-headless.

Good lighting and full face in frame improve stability.