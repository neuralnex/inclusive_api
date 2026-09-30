from typing import List, Optional

from pydantic import BaseModel


class PredictionItem(BaseModel):
    word: str
    confidence: float


class PredictResponse(BaseModel):
    top1_word: str
    top1_confidence: float
    low_confidence: bool
    predictions: List[PredictionItem]
    audio_base64: Optional[str] = None
    audio_format: str = "wav"


class VocabResponse(BaseModel):
    num_classes: int
    words: List[str]


class HealthResponse(BaseModel):
    status: str
    backbone: str
    num_classes: int
