"""
Text-to-speech via pyttsx3 -- offline, no network call, no per-request cost,
matching Phase 1 Section 10's recommendation for offline TTS deployment.

pyttsx3 is not thread-safe and its engine object is meant to be used
synchronously (init -> speak/save -> runAndWait), so a fresh engine is
created per call rather than sharing one across concurrent requests. For a
demo/low-traffic deployment this is fine; if this API ever needs to serve
many concurrent users, swap this module for a queued/worker-based TTS
service or a cloud TTS API (Azure/Google, per Phase 1 Section 10's
recommendation for that scale) without changing the endpoint contract.
"""

import tempfile
from pathlib import Path

from . import config


def synthesize_to_wav_bytes(text: str) -> bytes:
    """Synthesizes `text` to speech and returns raw WAV file bytes."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", config.TTS_RATE)
    engine.setProperty("volume", config.TTS_VOLUME)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        engine.save_to_file(text, str(tmp_path))
        engine.runAndWait()
        audio_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
        try:
            engine.stop()
        except Exception:
            pass

    return audio_bytes
