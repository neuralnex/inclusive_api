"""
Inclusive API - main application.

Two ways a client can get a prediction:

1. POST /predict/video  -- upload a single video file (.mp4 etc), the
   "upload a clip" case. Matches exactly how Phase 4's evaluation script
   reads clips, so results are consistent with your reported accuracy.

2. POST /predict/frames -- upload several individual images, the "live
   scan" case. A browser frontend using getUserMedia() to access the
   webcam would capture frames periodically (e.g. every ~150-200ms via
   canvas.toBlob()) for a couple of seconds, then POST all of them here as
   a single multipart request. This endpoint handles ANY number of images
   -- fewer than the model's expected frame count get repeated
   (edge-hold), more get uniformly subsampled, exactly like a real video
   clip would be. There is no WebSocket/streaming complexity here on
   purpose: batching a few seconds of snapshots into one request is far
   simpler to implement on both ends and matches how the model was
   actually trained (on short, complete clips, not a continuous stream).

Every response includes synthesized speech for the top prediction as
base64-encoded WAV audio, so the client can decode and play it directly
without a second request.
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .inference_service import inference_service
from .schemas import HealthResponse, PredictionItem, PredictResponse, VocabResponse
from .tts_service import synthesize_to_wav_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    inference_service.load()  # loaded ONCE here, not per-request
    yield


app = FastAPI(title="Inclusive - ASL Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/test-client", StaticFiles(directory="static", html=True), name="test-client")


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/test-client")


def _build_predict_response(ranked) -> PredictResponse:
    """ranked: [(word, confidence), ...] sorted descending."""
    top1_word, top1_conf = ranked[0]
    low_conf = top1_conf < config.LOW_CONFIDENCE_THRESHOLD

    try:
        audio_bytes = synthesize_to_wav_bytes(top1_word)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as e:
        # TTS failing should never take down a prediction response -- the
        # recognized word is still useful on its own.
        print(f"[TTS] synthesis failed: {e}")
        audio_b64 = None

    return PredictResponse(
        top1_word=top1_word,
        top1_confidence=top1_conf,
        low_confidence=low_conf,
        predictions=[PredictionItem(word=w, confidence=c) for w, c in ranked],
        audio_base64=audio_b64,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        backbone=inference_service.backbone,
        num_classes=len(inference_service.vocabulary),
    )


@app.get("/vocab", response_model=VocabResponse)
def vocab():
    """The exact list of words the model can recognize -- there is no
    'unknown' class, so a client should only expect meaningful results for
    signs in this list."""
    words = inference_service.vocabulary
    return VocabResponse(num_classes=len(words), words=words)


@app.post("/predict/video", response_model=PredictResponse)
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB, max {config.MAX_UPLOAD_MB}MB)")

    config.ensure_dirs()
    tmp_path = config.TEMP_UPLOAD_DIR / f"upload_{file.filename}"
    tmp_path.write_bytes(contents)

    try:
        ranked = inference_service.predict_from_video_file(tmp_path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return _build_predict_response(ranked)


@app.post("/predict/frames", response_model=PredictResponse)
async def predict_frames(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(422, "No images provided.")

    images = []
    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, f"Could not decode image: {f.filename}")
        images.append(img)

    try:
        ranked = inference_service.predict_from_image_frames(images)
    except ValueError as e:
        raise HTTPException(422, str(e))

    return _build_predict_response(ranked)


@app.get("/speak")
@app.post("/speak")
def speak(word: str):
    """Standalone TTS endpoint -- synthesize any word/phrase directly,
    independent of a prediction (useful for testing the TTS path alone).

    Accepts both GET and POST so browser clients and quick smoke tests can
    request speech without getting a method mismatch.
    """
    if not word or not word.strip():
        raise HTTPException(422, "A non-empty 'word' is required.")

    try:
        audio_bytes = synthesize_to_wav_bytes(word)
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis failed: {e}")
    return {"audio_base64": base64.b64encode(audio_bytes).decode("ascii"), "audio_format": "wav"}
