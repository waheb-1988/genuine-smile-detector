# smile/models.py
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel

@dataclass
class AnalysisResult:
    timestamp_utc: str
    source: str
    image_filename: str
    deepface_happy_prob: float
    eye_aperture: float
    neutral_eye_baseline: Optional[float]
    score: int
    label: str
    emoji: str

class AnalyzeResponse(BaseModel):
    timestamp_utc: str
    source: str
    image_filename: str
    deepface_happy_prob: float
    eye_aperture: float
    neutral_eye_baseline: Optional[float]
    score: int
    label: str
    emoji: str
