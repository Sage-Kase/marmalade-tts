#!/usr/bin/env python3
"""marmalade-tts kitten daemon — keeps the KittenTTS model loaded in RAM.

Request: {"text": "...", "voice": "Hugo", "speed": 1.0, "out": "/tmp/x.wav"}
"""

import os
import re
import sys

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


def _patch_phonemizer(onnx_model):
    backend = getattr(onnx_model, "phonemizer", None)
    if backend is None:  # kittentts internals moved; skip rather than crash
        return
    orig = backend.phonemize

    def phonemize(texts, **kwargs):
        return [fix_en_phonemes(p) for p in orig(texts, **kwargs)]

    backend.phonemize = phonemize


def load_model():
    from kittentts import KittenTTS
    model = KittenTTS(MODEL_REPO)
    _patch_phonemizer(model.model)
    return model


def synth(model, req):
    want = req.get("model")
    check_loaded("kitten", MODEL_REPOS.get(want, want), MODEL_REPO)
    model.generate_to_file(
        req["text"],
        req["out"],
        voice=req.get("voice", "Hugo"),
        speed=float(req.get("speed", 1.0)),
    )


if __name__ == "__main__":
    serve("kitten", load_model, synth)
