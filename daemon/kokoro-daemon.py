#!/usr/bin/env python3
"""marmalade-tts kokoro daemon — keeps the Kokoro pipeline loaded in RAM.

Any supported `lang` is served: KOKORO_LANG only picks the one preloaded
at startup (warm first call), and other languages get their front end
built on demand around the same weights — see _Pipelines.

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
from _common import serve

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

# A found gap is only credible if it abuts the alignment boundary. Two
# fused words ("ðə mˈɑɹkət" — K1-4b P8 probe, 2026-08-01) render with NO
# gap at their join; the window search then finds the PREVIOUS word's
# pause 3+ frames away, and snapping to it replays the tail of the
# context (a duplicated "the" at the seam) or truncates kept words on
# the lookahead side. Farther than this → treat as no gap and cut at
# the alignment boundary instead.
_GAP_NEAR = 3 * FRAME

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


def _frication(wav, lo: int, hi: int) -> bool:
    """Does [lo, hi) look like frication? High zero-crossing rate at low-
    to-mid energy — an /s ʃ f/ hovers under the silence threshold's energy
    neighbourhood, which is exactly why the quiet-span machinery misjudges
    it."""
    n = hi - lo
    if n <= 1:
        return False
    r2 = 0.0
    crossings = 0
    prev = float(wav[lo])
    for i in range(lo, hi):
        v = float(wav[i])
        r2 += v * v
        if (v < 0) != (prev < 0):
            crossings += 1
        prev = v
    return crossings / n > 0.2 and (r2 / n) ** 0.5 > 0.002


def _fricative_backoff(wav, cut: int, max_frames: int = 8) -> int:
    """Walk a start cut backward across contiguous frication so an onset
    fricative survives the cut. Measured failure (P10, 2026-07-31): on
    the tight join "bˈæk, slˈOli" the model renders /s/ BEFORE a 7ms
    closure gap, and the alignment boundary lands after it — the
    quiet-span cut beheaded the /s/ and whisper heard "lowly". Walking
    back over fricative-looking frames keeps it; at worst this duplicates
    a sliver of the context word's release, and duplication beats loss
    (round-3 precedent)."""
    lo = max(0, cut - max_frames * FRAME)
    t = cut
    while t - FRAME >= lo and _frication(wav, t - FRAME, t):
        t -= FRAME
    return t


def _context_cut(wav, approx: int) -> int:
    """Acoustic start cut for a context prefix: snap the alignment-derived
    position to the word gap near it, backed off two frames into the gap
    (onsets bleed into the space frames before them; a soft onset can
    start under the silence threshold), then walked back across any
    frication butting the cut (see _fricative_backoff). No credible gap
    nearby → the approximation is the best available."""
    s, e = _quiet_span(wav, approx - _CUT_SEARCH_BACK, approx + _CUT_SEARCH_FWD)
    if e - s < _MIN_GAP_SAMPLES:
        return _fricative_backoff(wav, max(0, approx))
    s, e = _extend_quiet(wav, s, e)
    if approx - e > _GAP_NEAR or s - approx > _GAP_NEAR:
        return _fricative_backoff(wav, max(0, approx))
    return _fricative_backoff(wav, max(s, e - 2 * FRAME))


def _lookahead_cut(wav, approx: int) -> int:
    """Acoustic end cut before a lookahead suffix: snap to the word gap
    near the alignment position and keep at most TAIL_KEEP frames of it —
    the chunk keeps its own rendered pause, not the lookahead's onset
    bleed (and never a runaway model-chosen silence)."""
    s, e = _quiet_span(wav, approx - _CUT_SEARCH_BACK, approx + _CUT_SEARCH_FWD)
    if e - s < _MIN_GAP_SAMPLES:
        return max(0, approx)
    s, e = _extend_quiet(wav, s, e)
    if approx - e > _GAP_NEAR or s - approx > _GAP_NEAR:
        return max(0, approx)
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


class _Pipelines:
    """Per-language pipelines over ONE loaded KModel.

    A KPipeline is a misaki G2P front end bolted to a model; the voice
    embedding and the G2P language are orthogonal, so a lang the daemon
    didn't preload costs only a front end, not a second copy of the
    weights (passing `model=` reuses the loaded one). Both maps are
    filled lazily and kept — requests are serialized by the daemon's
    concurrency gate, so plain dicts are enough.
    """

    def __init__(self, lang: str):
        from kokoro import KPipeline
        import soundfile as sf  # noqa: F401  — fail fast if missing
        self._KPipeline = KPipeline
        pipe = KPipeline(lang_code=lang, device="cpu")
        self.model = pipe.model
        self._full = {lang: pipe}
        # Model-free pipelines serve op=phonemize: calling the full
        # pipeline would run inference; these stop at misaki G2P.
        self._g2p = {lang: KPipeline(lang_code=lang, model=False)}

    def full(self, lang: str):
        if lang not in self._full:
            self._full[lang] = self._KPipeline(lang_code=lang,
                                               model=self.model)
        return self._full[lang]

    def g2p(self, lang: str):
        if lang not in self._g2p:
            self._g2p[lang] = self._KPipeline(lang_code=lang, model=False)
        return self._g2p[lang]


def load_model():
    return _Pipelines(DEFAULT_LANG)


def synth(pipelines, req):
    import numpy as np
    import soundfile as sf

    # KOKORO_LANG is the preloaded (warm) language, not a restriction:
    # any other lang builds its front end on first use.
    lang = req.get("lang") or DEFAULT_LANG

    if req.get("op") == "phonemize":
        g2p = pipelines.g2p(lang)
        ph = " ".join(r.phonemes for r in g2p(req["text"]) if r.phonemes)
        with open(req["out"], "w", encoding="utf-8") as f:
            f.write(ph)
        return

    pipeline = pipelines.full(lang)

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
