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

_raw_model = os.environ.get("KITTEN_MODEL", "nano")  # nano = config-default.yaml default
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


# ── Duration-aware seam surgery ─────────────────────────────────────────────
# The model's ONNX graph emits a second output the kittentts wrapper throws
# away: per-input-token durations, at exactly FRAME samples per frame
# (sum(duration)*FRAME == len(waveform); verified 2026-07-28, probe scripts
# in ~/coding/scratch/probe_kitten*.py). That gives sample-exact boundaries
# for free, which we use to:
#   * trim the ~460ms BOS lead pad and the tail pad every chunk carries —
#     the ~700-800ms of dead air at every chunk-stream seam (and fix the
#     wrapper's blind [:-5000] trim, which clips ~200ms of real speech on
#     punctuation-less chunks);
#   * cut a rendered continuation prefix ("context") at the space token
#     nearest its end, so a chunk can be conditioned on the previous
#     chunk's tail words yet emit only its own audio. Cutting at a space
#     token means an off-by-one lands in an inter-word gap, not in speech.
# We capture (input_ids, waveform, duration) by proxying session.run rather
# than re-implementing tokenization/speed-priors — generate() still does
# all of that; we just rebuild the audio from the raw outputs.

SAMPLE_RATE = 24000
FRAME = 600           # samples per duration frame (model constant)
HEAD_KEEP = 2         # frames of lead pad kept (~50ms) — onset safety
TAIL_KEEP = 3         # frames of tail pad kept (~75ms) — natural gap
RUN_GAP_MS = 150      # silence between kittentts's internal sentence runs

# Token vocab (kittentts/StyleTTS2 symbol set). Only the space id is used.
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = ("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
                "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
VOCAB = {s: i for i, s in enumerate(
    [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa))}
SPACE_ID = VOCAB[" "]


class _CaptureSession:
    """Proxy around the ORT session that records raw run outputs."""

    def __init__(self, session):
        self._session = session
        self._local = threading.local()

    def begin(self):
        self._local.runs = []

    def taken(self):
        return getattr(self._local, "runs", [])

    def run(self, output_names, feed, *a, **kw):
        res = self._session.run(output_names, feed, *a, **kw)
        runs = getattr(self._local, "runs", None)
        if runs is not None and len(res) > 1:
            ids = feed.get("input_ids")
            runs.append((None if ids is None else list(map(int, ids[0])),
                         res[0], res[1]))
        return res

    def __getattr__(self, name):
        return getattr(self._session, name)


def _cum_samples(dur) -> list:
    out, acc = [0], 0
    for d in dur:
        acc += int(d) * FRAME
        out.append(acc)
    return out


def _context_cut(ids, dur, n_context_phonemes: int) -> int:
    """Sample offset where the context prefix ends: the space token nearest
    to (BOS + n_context_phonemes), cut at that token's end."""
    target = 1 + n_context_phonemes
    spaces = [i for i, t in enumerate(ids) if t == SPACE_ID]
    if not spaces:
        return 0
    idx = min(spaces, key=lambda i: abs(i - target))
    return _cum_samples(dur)[idx + 1]


def _tail_silence_frames(ids, dur) -> int:
    """Frames of trailing non-speech: the EOS pad plus any punctuation/
    space tokens before it (ensure_punctuation appends a comma whose pause
    frames sit in front of the pad — trim the whole group)."""
    frames = 0
    for tid, d in zip(reversed(ids), reversed(list(dur))):
        if tid <= SPACE_ID:  # pad (0), punctuation, or space
            frames += int(d)
        else:
            break
    return frames


def _trim_run(ids, wav, dur, n_context_phonemes: int = 0):
    """Return the speech-bearing slice of one raw (flat) run: context
    prefix (if any) cut at a word gap, lead/tail non-speech reduced to
    small margins. Pure sequence ops — unit-tested without the model."""
    total = _cum_samples(dur)
    if len(wav) != total[-1]:  # duration contract broken; don't touch it
        return wav
    if n_context_phonemes > 0:
        start = _context_cut(ids, dur, n_context_phonemes)
    else:
        start = max(0, (int(dur[0]) - HEAD_KEEP)) * FRAME
    tail_pad = max(0, _tail_silence_frames(ids, dur) - TAIL_KEEP) * FRAME
    end = len(wav) - tail_pad
    return wav[start:end] if start < end else wav


def _synth_direct(model, om, req) -> bool:
    """Duration-trimmed synthesis via output capture. False → caller falls
    back to plain generate_to_file (kittentts internals moved)."""
    import numpy as np
    import soundfile as sf

    cap = getattr(om, "session", None)
    if not isinstance(cap, _CaptureSession):
        return False
    text = req["text"]
    context = (req.get("context") or "").strip()
    full = (context + " " + text) if context else text

    n_ctx = 0
    if context:
        backend = getattr(om, "phonemizer", None)
        if backend is None:
            return False
        ph = backend.phonemize([context])[0]
        n_ctx = sum(1 for c in ph if c in VOCAB)
        if n_ctx == 0:
            context, full = "", text

    cap.begin()
    try:
        if hasattr(model, "generate"):
            model.generate(full, voice=req.get("voice", "Kiki"),
                           speed=float(req.get("speed", 1.0)))
        else:
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                model.generate_to_file(full, tmp,
                                       voice=req.get("voice", "Kiki"),
                                       speed=float(req.get("speed", 1.0)))
            finally:
                os.unlink(tmp)
    finally:
        runs = cap.taken()
    if not runs or any(r[0] is None for r in runs):
        return False

    gap = np.zeros(int(SAMPLE_RATE * RUN_GAP_MS / 1000), dtype=np.float32)
    pieces = []
    for i, (ids, wav, dur) in enumerate(runs):
        if pieces:
            pieces.append(gap)
        # Only the first run contains the context prefix.
        pieces.append(_trim_run(ids, np.asarray(wav).reshape(-1),
                                np.asarray(dur).ravel(),
                                n_ctx if i == 0 else 0))
    sf.write(req["out"], np.concatenate(pieces), SAMPLE_RATE)
    return True


def load_model():
    from kittentts import KittenTTS
    model = KittenTTS(MODEL_REPO)
    _patch_phonemizer(model.model)
    _materialize_voices(model.model)
    if hasattr(model.model, "session"):
        model.model.session = _CaptureSession(model.model.session)
    return model


def synth(model, req):
    want = req.get("model")
    check_loaded("kitten", MODEL_REPOS.get(want, want), MODEL_REPO)
    try:
        if _synth_direct(model, model.model, req):
            return
    except Exception as exc:  # never let seam surgery break synthesis
        import logging
        logging.getLogger("kitten-daemon").warning(
            "direct path failed (%s); falling back to generate_to_file", exc)
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
