#!/usr/bin/env python3
"""marmalade-tts kitten daemon — keeps the KittenTTS model loaded in RAM.

Request: {"text": "...", "voice": "Hugo", "speed": 1.0, "out": "/tmp/x.wav"}
"""

import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import serve, check_loaded

# Force CPU and offline mode — Pascal sm_61 isn't supported by current torch wheels,
# and we don't want re-downloads after the first cache fill.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_REPOS = {
    "nano":  "KittenML/kitten-tts-nano-0.8",
    "micro": "KittenML/kitten-tts-micro-0.8",
    "mini":  "KittenML/kitten-tts-mini-0.8",
}

_raw_model = os.environ.get("KITTEN_MODEL", "micro")  # micro = config-default.yaml default
MODEL_REPO = MODEL_REPOS.get(_raw_model, _raw_model)  # accept size name or full repo


# espeak-ng has no dictionary entry for "yeah" — its letter-to-sound
# fallback emits /jɛh/ (a literal aspirated H, audibly "yeh-h").
# KittenTTS phonemizes internally with espeak, so we correct the
# phoneme stream on its way to the model. The replacement is flat /jæ/
# ("ya"), not the lexically faithful /jɛə/: Kitten renders the ɛ→ə
# glide poorly on every model size, and /jæ/ won the 2026-07-27
# listening A/B bar none. Word-start match only; no right-hand boundary
# so "yeah's" → /jɛhz/ is caught too. Same fix as EnPhonemeFixups.kt
# (Model.KITTEN) in marmalade-tts-android.
_YEAH_RE = re.compile(r"(?<![^ ])j([ˈˌ]?)ɛh")


def fix_en_phonemes(phonemes: str) -> str:
    return _YEAH_RE.sub(r"j\1æ", phonemes)


# espeak keeps global state — concurrent phonemize calls are unsafe, so the
# patched wrapper serializes G2P while leaving ONNX inference free to run in
# parallel across requests (same split as upstream KittenTTS PR #147).
_phonemize_lock = threading.Lock()


def _patch_phonemizer(onnx_model):
    backend = getattr(onnx_model, "phonemizer", None)
    if backend is None:  # kittentts internals moved; skip rather than crash
        return
    orig = backend.phonemize

    def phonemize(texts, **kwargs):
        with _phonemize_lock:
            out = orig(texts, **kwargs)
        return [fix_en_phonemes(p) for p in out]

    backend.phonemize = phonemize


def _materialize_voices(onnx_model):
    """Replace the lazy NpzFile voice store with a plain dict of arrays.

    numpy's NpzFile reads entries from one shared zip handle, which is not
    thread-safe — concurrent requests raced it into "Overlapped entries ...
    (possible zip bomb)" errors. Plain ndarrays are read-only-safe."""
    voices = getattr(onnx_model, "voices", None)
    files = getattr(voices, "files", None)
    if files is None:  # already a dict, or kittentts internals moved
        return
    onnx_model.voices = {name: voices[name] for name in files}


def load_model():
    from kittentts import KittenTTS
    model = KittenTTS(MODEL_REPO)
    _patch_phonemizer(model.model)
    _materialize_voices(model.model)
    return model


def synth(model, req):
    want = req.get("model")
    check_loaded("kitten", MODEL_REPOS.get(want, want), MODEL_REPO)
    model.generate_to_file(
        req["text"],
        req["out"],
        voice=req.get("voice", "Kiki"),
        speed=float(req.get("speed", 1.0)),
    )


if __name__ == "__main__":
    # Kitten's handler is thread-safe (phonemize locked above; ORT session.run
    # is reentrant; each request writes its own out file), so let chunked
    # requests overlap. Capped at 4: each inference already uses ORT intra-op
    # threads, so more concurrent runs just oversubscribe the CPU.
    serve("kitten", load_model, synth,
          max_concurrency=min(4, os.cpu_count() or 1))
