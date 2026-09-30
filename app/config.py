"""
Inclusive API - configuration.

Points at a local models/ folder inside this project -- copy your trained
checkpoint (.pt) and its label_vocab.json into inclusive_api/models/
before starting the server. Everything here is overridable via environment
variables so you can deploy without editing code (e.g.
`INCLUSIVE_BACKBONE=videomae uvicorn app.main:app`).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Where to find the trained model -- LOCAL to this project, not Phase 4.
# Copy both files here:
#   models/{BACKBONE}_{TAG}_best.pt
#   models/{BACKBONE}_{TAG}_label_vocab.json
# e.g. models/mvit_checkpoints_gru_top30_baseline_label_vocab_best.pt
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("INCLUSIVE_MODELS_DIR", APP_ROOT / "models"))
CHECKPOINT_DIR = MODELS_DIR
CACHE_DIR = MODELS_DIR / "cache"  # pretrained-backbone download cache (torchvision/transformers)

BACKBONE = os.environ.get("INCLUSIVE_BACKBONE", "mvit")
TAG = os.environ.get("INCLUSIVE_TAG", "checkpoints_gru_top30_baseline_label_vocab")

CHECKPOINT_PATH = CHECKPOINT_DIR / f"{BACKBONE}_{TAG}_best.pt"
VOCAB_PATH = CHECKPOINT_DIR / f"{BACKBONE}_{TAG}_label_vocab.json"

# ---------------------------------------------------------------------------
# Video/frame preprocessing -- MUST match what the checkpoint was trained with
# ---------------------------------------------------------------------------
NUM_FRAMES = int(os.environ.get("INCLUSIVE_NUM_FRAMES", 16))
FRAME_SIZE = int(os.environ.get("INCLUSIVE_FRAME_SIZE", 224))

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("INCLUSIVE_DEVICE", "cpu")  # "cuda" if you have a GPU available
TOP_K = int(os.environ.get("INCLUSIVE_TOP_K", 5))

# Below this confidence, the API still returns a prediction but flags it as
# low-confidence -- the model always outputs SOME word from its vocabulary
# (there is no "I don't know" class), so the client/UI needs this signal to
# avoid presenting a low-confidence guess as a certain answer.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("INCLUSIVE_CONFIDENCE_THRESHOLD", 0.30))

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("INCLUSIVE_MAX_UPLOAD_MB", 50))
TEMP_UPLOAD_DIR = Path(os.environ.get("INCLUSIVE_TEMP_DIR", "./tmp_uploads"))

# ---------------------------------------------------------------------------
# Text-to-speech (pyttsx3 -- offline, per Phase 1 Section 10's offline-
# deployment recommendation; no network call, no per-request cost)
# ---------------------------------------------------------------------------
TTS_RATE = int(os.environ.get("INCLUSIVE_TTS_RATE", 150))   # words per minute
TTS_VOLUME = float(os.environ.get("INCLUSIVE_TTS_VOLUME", 1.0))

# ---------------------------------------------------------------------------
# CORS -- a browser-based frontend (React dev server, static HTML page,
# etc.) served from a different origin needs this to call the API at all
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.environ.get("INCLUSIVE_CORS_ORIGINS", "*").split(",")


def ensure_dirs():
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
