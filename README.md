# Inclusive API

FastAPI service wrapping your trained Phase 4 model. Two ways to get a
prediction (video upload or live webcam scan), both returning the
recognized word plus synthesized speech, in one response.

## What was and wasn't verified before hand-off

No FastAPI or pyttsx3 installed in the environment that generated this
code (matches every prior phase's constraint).

- **Verified with real runtime tests:** the frame-batching logic in
  `inference_service.py` (`_sample_frame_indices`) — confirmed it correctly
  handles fewer-than-needed images (repeats via edge-hold), more-than-needed
  (uniform subsample), and exactly-matching counts, which is the mechanism
  that makes the live-scan endpoint work regardless of how many webcam
  snapshots a frontend sends.
- **Syntax-checked only:** everything touching FastAPI, torch, or pyttsx3.
  Run the smoke test below before trusting a real deployment.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Copy your two model files into `models/`** (this folder, local to the
API — not read from `inclusive_phase4/` anymore):

```
models/mvit_checkpoints_gru_top30_baseline_label_vocab_best.pt
models/mvit_checkpoints_gru_top30_baseline_label_vocab_label_vocab.json
```

Both files come straight from your Phase 4 `checkpoints/` folder — just
copy them over. If you trained a different backbone/tag combination, set
the matching environment variables instead of renaming files:

```bash
export INCLUSIVE_BACKBONE=mvit
export INCLUSIVE_TAG=checkpoints_gru_top30_baseline_label_vocab
```

`INCLUSIVE_MODELS_DIR` overrides where the API looks for these files
entirely (defaults to `./models` relative to the project root), if you'd
rather keep them somewhere else.

## Run it

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The model loads once at startup (watch the console for
`[InferenceService] loaded mvit checkpoint, 30 classes, device=cpu`) — if
that line doesn't print, nothing downstream will work, so check it first.

Interactive API docs: `http://localhost:8000/docs`
Test client (video upload + live webcam): `http://localhost:8000/test-client`

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Confirms the model loaded and which backbone/class count is active |
| `/vocab` | GET | The exact list of words the model can recognize — there is no "unknown" class, so anything outside this list will still get force-matched to the nearest wrong word |
| `/predict/video` | POST (multipart `file`) | Upload a single video clip, get a prediction + speech |
| `/predict/frames` | POST (multipart `files[]`, one or more images) | The "live scan" path — a burst of webcam snapshots, batched into one request |
| `/speak` | POST (query param `word`) | Standalone TTS, independent of prediction — useful for testing the audio path alone |

Every `/predict/*` response looks like:
```json
{
  "top1_word": "brother",
  "top1_confidence": 0.42,
  "low_confidence": false,
  "predictions": [{"word": "brother", "confidence": 0.42}, ...],
  "audio_base64": "UklGRi...",
  "audio_format": "wav"
}
```
Decode `audio_base64` client-side and play it directly — no second request
needed to fetch the audio.

## Quick test with curl

```bash
curl -X POST http://localhost:8000/predict/video \
  -F "file=@/path/to/some_clip.mp4" | python3 -m json.tool
```

## How "live scan" actually works here

There's no WebSocket or continuous streaming — the test client (and any
frontend you build) captures a burst of webcam snapshots over a few
seconds (the included `static/index.html` does ~20 frames over 3 seconds
via `canvas.toBlob()`) and sends them all as one multipart request to
`/predict/frames`. This is deliberate: it matches how the model was
trained (on short, complete clips), and it's far simpler to implement
correctly than frame-by-frame streaming inference, at the cost of a few
seconds of latency between "sign performed" and "result shown." For a
final-year demo, that tradeoff is the right one.

## Important limitations to state honestly in your writeup/demo

- **No "unknown" class.** The model always outputs its best guess from the
  30 trained words, even for signs it's never seen. `low_confidence` in the
  response (confidence below `INCLUSIVE_CONFIDENCE_THRESHOLD`, default 0.30)
  is the closest signal to "the model isn't sure," but it is not the same
  as "this isn't a sign the model knows" — a confidently wrong answer is
  still possible.
- **CPU inference latency.** Per your Phase 4 eval, this checkpoint runs
  at ~1.6s/clip on CPU — the API will feel that same latency per
  request. Fine for a demo; state it plainly rather than treating it as a
  production number, consistent with your Phase 1 Section 14.1 targets
  which assume GPU-class deployment hardware.
- **pyttsx3 is not built for concurrent requests.** A fresh TTS engine is
  created per call, which is fine at demo scale (one user at a time) but
  would need a queue/worker redesign for real multi-user traffic — noted
  directly in `tts_service.py`'s docstring.
