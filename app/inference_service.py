"""
Inference service. The model is loaded ONCE at process startup (see
main.py's lifespan handler) and reused across every request -- reloading a
video transformer per request would make every prediction take as long as
your Phase 4 training script's model-build step, which is not acceptable
for an API.

Two input paths:
  - predict_from_video_file: a single uploaded video clip (the "image/video
    upload" case). Reuses the exact same frame-sampling/normalization used
    in Phase 4 training and evaluation, so predictions here are consistent
    with your reported accuracy numbers.
  - predict_from_image_frames: a list of individual uploaded images (the
    "live scan" case) -- see the module docstring in main.py for how a
    frontend is expected to produce these (periodic webcam snapshots).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import config


def _resize_center_crop(frame_bgr, size):
    h, w = frame_bgr.shape[:2]
    scale = size / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h))
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    cropped = resized[top:top + size, left:left + size]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def _sample_frame_indices(num_available, num_frames):
    """Same logic as Phase 4's utils/video_frames.py -- uniform sampling,
    edge-hold repetition if fewer frames are available than needed."""
    if num_available <= 0:
        return []
    if num_available >= num_frames:
        return np.linspace(0, num_available - 1, num_frames).round().astype(int).tolist()
    reps = int(np.ceil(num_frames / num_available))
    extended = list(range(num_available)) * reps
    return extended[:num_frames]


def _normalize_for_model(frames_uint8, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    frames = frames_uint8.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    frames = (frames - mean) / std
    return np.transpose(frames, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)


class InferenceService:
    """Holds the loaded model + vocab; call .load() once at startup."""

    def __init__(self):
        self.model = None
        self.idx_to_gloss = None
        self.device = None
        self.backbone = config.BACKBONE

    def load(self):
        if not config.CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint at {config.CHECKPOINT_PATH}. Check INCLUSIVE_PHASE4_ROOT, "
                f"INCLUSIVE_BACKBONE, and INCLUSIVE_TAG environment variables."
            )
        if not config.VOCAB_PATH.exists():
            raise FileNotFoundError(f"No vocab file at {config.VOCAB_PATH}.")

        with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
            label_vocab = json.load(f)
        self.idx_to_gloss = {idx: gloss for gloss, idx in label_vocab.items()}
        num_classes = len(label_vocab)

        self.device = torch.device(config.DEVICE)

        # Imported lazily so this module doesn't hard-require torchvision/
        # transformers just to be imported (e.g. by a test that only checks
        # the frame-batching helpers above).
        from .model_loader import build_model
        self.model = build_model(self.backbone, num_classes, config.CACHE_DIR).to(self.device)
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"[InferenceService] loaded {self.backbone} checkpoint, "
              f"{num_classes} classes, device={self.device}")

    @property
    def vocabulary(self):
        return sorted(self.idx_to_gloss.values()) if self.idx_to_gloss else []

    def _forward(self, clip_uint8_frames):
        """clip_uint8_frames: (T, H, W, 3) uint8 RGB array, already sized to
        config.FRAME_SIZE. Returns ranked [(gloss, confidence), ...]."""
        normed = _normalize_for_model(clip_uint8_frames)
        clip_t = torch.from_numpy(normed).unsqueeze(0).to(self.device)  # (1,T,C,H,W)

        from .model_loader import forward_logits
        with torch.no_grad():
            logits = forward_logits(self.backbone, self.model, clip_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:config.TOP_K]
        return [(self.idx_to_gloss[int(i)], float(probs[i])) for i in top_idx]

    def predict_from_video_file(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError("Could not decode any frames from the uploaded video.")

        indices = _sample_frame_indices(len(frames), config.NUM_FRAMES)
        clip = np.zeros((config.NUM_FRAMES, config.FRAME_SIZE, config.FRAME_SIZE, 3), dtype=np.uint8)
        for i, idx in enumerate(indices):
            clip[i] = _resize_center_crop(frames[idx], config.FRAME_SIZE)

        return self._forward(clip)

    def predict_from_image_frames(self, image_bgr_list):
        """image_bgr_list: list of BGR uint8 arrays (as cv2.imdecode returns),
        e.g. individual webcam snapshots uploaded by a frontend. Handles any
        count -- fewer than NUM_FRAMES repeats via edge-hold, more than
        NUM_FRAMES is uniformly subsampled, same as a full video clip."""
        if not image_bgr_list:
            raise ValueError("No images provided.")

        indices = _sample_frame_indices(len(image_bgr_list), config.NUM_FRAMES)
        clip = np.zeros((config.NUM_FRAMES, config.FRAME_SIZE, config.FRAME_SIZE, 3), dtype=np.uint8)
        for i, idx in enumerate(indices):
            clip[i] = _resize_center_crop(image_bgr_list[idx], config.FRAME_SIZE)

        return self._forward(clip)


# Module-level singleton -- main.py's lifespan handler calls .load() once.
inference_service = InferenceService()
