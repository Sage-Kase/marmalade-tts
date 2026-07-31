#!/usr/bin/env python3
"""marmalade-tts kokoro daemon — keeps the Kokoro pipeline loaded in RAM.

Text request:  {"text": "...", "voice": "af_heart", "speed": 1.0, "lang": "a",
                "out": "/tmp/x.wav"}
Phonemize:     {"op": "phonemize", "text": "...", "lang": "a", "out": ...}
               — writes the misaki phoneme string (marks retained) to `out`.
Phoneme synth: {"ph_text": "...", "ph_context": ..., "ph_lookahead": ...,
                "style_ref": int, "pad_marks": {":": 150}, "voice": ...,
                "speed": ..., "lang": ..., "out": ...}

The phoneme path mirrors the kitten daemon's, with one deliberate
difference: every cut is ACOUSTIC, not alignment-index. Kokoro's pred_dur
is a usable approximation but not a boundary — measured 2026-07-30
(~/coding/scratch/probe_kokoro1.py): the BOS token claims 325–450ms of
frames while true lead silence is ~228ms, i.e. the alignment lands ~100ms
inside speech, and a conditioning cut taken straight from cum[...] falls
on voiced audio (probe_kokoro2.py: rms 0.035 → 0.073 across the index).
So pred_dur locates the neighbourhood and a quiet-span search places the
knife. Frame size and trim margins match kitten (600 samples/frame at
24kHz — same constant, verified).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import serve, check_loaded

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

DEFAULT_LANG = os.environ.get("KOKORO_LANG", "a")

SAMPLE_RATE = 24000
FRAME = 600           # samples per pred_dur frame (measured, same as kitten)
HEAD_KEEP = 2         # frames of lead silence kept (~50ms) — onset safety
TAIL_KEEP = 3         # frames of tail silence kept (~75ms) — natural gap
_SIL_LEVEL = 0.006    # |sample| below this is silence
MIN_PAUSE_SAMPLES = 240   # 10ms — below this a mark has no rendered pause
_MIN_GAP_SAMPLES = 60     # 2.5ms — smallest quiet run worth snapping a cut to

# Search window around an alignment-derived cut position. Biased backward:
# the measured misalignment is the boundary landing late (inside speech).
_CUT_SEARCH_BACK = 5 * FRAME
_CUT_SEARCH_FWD = 3 * FRAME

# misaki keeps punctuation in its phoneme output; these (plus whitespace)
# are the non-speech characters for group-walking and cut placement.
_PUNCT_SET = set(';:,.!?¡¿—…"«»“” ')


def _is_speech(c: str) -> bool:
    return c not in _PUNCT_SET and not c.isspace()


def _first_loud(wav) -> int:
    for i in range(len(wav)):
        if abs(float(wav[i])) >= _SIL_LEVEL:
            return i
    return len(wav)


def _last_loud_end(wav) -> int:
    for i in range(len(wav) - 1, -1, -1):
        if abs(float(wav[i])) >= _SIL_LEVEL:
            return i + 1
    return 0


def _quiet_span(wav, lo: int, hi: int):
    """(start, end) of the longest near-silent run inside [lo, hi)."""
    best_s = best_e = lo
    i, n = max(0, lo), min(hi, len(wav))
    while i < n:
        if abs(float(wav[i])) < _SIL_LEVEL:
            j = i
            while j < n and abs(float(wav[j])) < _SIL_LEVEL:
                j += 1
            if j - i > best_e - best_s:
                best_s, best_e = i, j
            i = j
        else:
            i += 1
    return best_s, best_e


def _extend_quiet(wav, s: int, e: int):
    """Grow [s, e) outward across contiguous silence — the search window
    locates a pause; the pause itself may run past it, and cuts must be
    measured against the whole thing."""
    while s > 0 and abs(float(wav[s - 1])) < _SIL_LEVEL:
        s -= 1
    while e < len(wav) and abs(float(wav[e])) < _SIL_LEVEL:
        e += 1
    return s, e


def _context_cut(wav, approx: int) -> int:
    """Acoustic start cut for a context prefix: snap the alignment-derived
    position to the word gap near it, backed off one frame into the gap so
    the first kept word keeps its attack (onsets bleed into the space
    frames before them). No credible gap nearby → the approximation is the
    best available."""
    s, e = _quiet_span(wav, approx - _CUT_SEARCH_BACK, approx + _CUT_SEARCH_FWD)
    if e - s < _MIN_GAP_SAMPLES:
        return max(0, approx)
    s, e = _extend_quiet(wav, s, e)
    return max(s, e - FRAME)


def _lookahead_cut(wav, approx: int) -> int:
    """Acoustic end cut before a lookahead suffix: snap to the word gap
    near the alignment position and keep at most TAIL_KEEP frames of it —
    the chunk keeps its own rendered pause, not the lookahead's onset
    bleed (and never a runaway model-chosen silence)."""
    s, e = _quiet_span(wav, approx - _CUT_SEARCH_BACK, approx + _CUT_SEARCH_FWD)
    if e - s < _MIN_GAP_SAMPLES:
        return max(0, approx)
    s, e = _extend_quiet(wav, s, e)
    return min(e, s + TAIL_KEEP * FRAME)


def _pause_inserts(wav, chars, bounds, targets: dict) -> list:
    """[(sample_position, zeros_to_insert)] so each marked punctuation joint
    reaches its target total silence. Same policy as the kitten daemon: the
    pause goes inside the actual near-silent run within the frames the
    model assigned to the mark (the alignment locates it, the waveform
    places it), measured over the whole contiguous silence; a mark with no
    rendered pause is left alone rather than cut through. ``bounds[i]`` is
    the sample where phoneme char ``i`` starts; ``bounds[i+1]`` its end."""
    if not targets:
        return []
    inserts = []
    for i, c in enumerate(chars):
        ms = targets.get(c)
        if ms is None:
            continue
        j = i
        while j + 1 < len(chars) and not _is_speech(chars[j + 1]):
            j += 1
        if j + 1 >= len(chars):
            continue  # trailing group — the tail trim owns that boundary
        start, end = _quiet_span(wav, bounds[i] - FRAME, bounds[j + 1] + FRAME)
        if end - start < MIN_PAUSE_SAMPLES:
            continue
        start, end = _extend_quiet(wav, start, end)
        extra = int(SAMPLE_RATE * float(ms) / 1000) - (end - start)
        if extra > 0:
            inserts.append(((start + end) // 2, extra))
    return inserts


def _splice(wav, inserts: list, start: int, end: int):
    """wav[start:end] with the in-range inserts realized as silence."""
    import numpy as np
    pieces, pos = [], start
    for at, n in inserts:
        if not (start < at < end):
            continue
        pieces.append(wav[pos:at])
        pieces.append(np.zeros(n, dtype=wav.dtype))
        pos = at
    pieces.append(wav[pos:end])
    return np.concatenate(pieces)


def _synth_phonemes(pipeline, req):
    """Conditioned synthesis from pre-phonemized input, acoustic cuts."""
    import numpy as np
    import soundfile as sf

    ctx = (req.get("ph_context") or "").strip()
    text = req["ph_text"].strip()
    la = (req.get("ph_lookahead") or "").strip()
    ps = " ".join(p for p in (ctx, text, la) if p)

    voice = req.get("voice", "af_heart")
    pack = pipeline.load_voice(voice)
    rows = pack.shape[0]
    # Default row is kokoro's own rule applied to the chunk's text:
    # KPipeline.infer indexes pack[len(ps)-1] over the whole segment.
    ref_id = req.get("style_ref")
    if ref_id is None:
        ref_id = len(text) - 1
    ref_id = max(0, min(int(ref_id), rows - 1))

    out = pipeline.model(ps, pack[ref_id], float(req.get("speed", 1.0)),
                         return_output=True)
    wav = out.audio.numpy()
    dur = None if out.pred_dur is None else out.pred_dur.numpy()

    vocab = pipeline.model.vocab
    chars = [c for c in ps if c in vocab]
    if (dur is None or len(dur) != len(chars) + 2
            or int(dur.sum()) * FRAME != len(wav)):
        raise RuntimeError("duration contract broken on kokoro phoneme path")
    # bounds[i] = sample where phoneme char i's token starts (BOS excluded).
    cum = np.concatenate([[0], np.cumsum(dur)]) * FRAME
    bounds = cum[1:]

    n_pre = sum(1 for c in ctx + " " if c in vocab) if ctx else 0
    n_text = sum(1 for c in text if c in vocab)

    if ctx:
        start = _context_cut(wav, int(bounds[n_pre]))
    else:
        start = max(0, _first_loud(wav) - HEAD_KEEP * FRAME)
    if la:
        # The alignment end is the start of the joining space before the
        # lookahead — the chunk's own trailing punctuation stays inside.
        end = _lookahead_cut(wav, int(bounds[n_pre + n_text]))
    else:
        end = min(len(wav), _last_loud_end(wav) + TAIL_KEEP * FRAME)
    if start >= end:
        start, end = 0, len(wav)

    targets = req.get("pad_marks") or {}
    out_wav = _splice(wav, _pause_inserts(wav, chars, bounds, targets),
                      start, end)
    sf.write(req["out"], out_wav, SAMPLE_RATE)


def load_model():
    from kokoro import KPipeline
    import soundfile as sf  # noqa: F401  — fail fast if missing
    pipe = KPipeline(lang_code=DEFAULT_LANG, device="cpu")
    # A second, model-free pipeline serves op=phonemize: calling the full
    # pipeline would run inference; this one stops at misaki G2P.
    g2p = KPipeline(lang_code=DEFAULT_LANG, model=False)
    return pipe, g2p


def synth(bundle, req):
    import numpy as np
    import soundfile as sf

    pipeline, g2p = bundle

    # The pipeline's G2P language is fixed at load; a request for a
    # different lang (e.g. a British voice through an 'a' daemon) would
    # silently mispronounce, so refuse it instead.
    check_loaded("kokoro", req.get("lang"), DEFAULT_LANG, what="lang")

    if req.get("op") == "phonemize":
        ph = " ".join(r.phonemes for r in g2p(req["text"]) if r.phonemes)
        with open(req["out"], "w", encoding="utf-8") as f:
            f.write(ph)
        return

    if "ph_text" in req:
        _synth_phonemes(pipeline, req)  # no text fallback: exact or error
        return

    audio_chunks = []
    for result in pipeline(req["text"],
                           voice=req.get("voice", "af_heart"),
                           speed=float(req.get("speed", 1.0))):
        if result.audio is not None:
            audio_chunks.append(result.audio.numpy())

    if not audio_chunks:
        raise RuntimeError("No audio generated")

    audio = np.concatenate(audio_chunks)
    sf.write(req["out"], audio, 24000)


if __name__ == "__main__":
    serve("kokoro", load_model, synth)
