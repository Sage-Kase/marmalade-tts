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
#   * cut a rendered continuation prefix ("context") sample-exactly at the
#     word gap that closes it, so a chunk can be conditioned on the previous
#     chunk's tail words yet emit only its own audio;
#   * cut a rendered continuation suffix ("lookahead") the same way from the
#     other end, so a chunk can be conditioned on the NEXT chunk's opening
#     words — its last word then carries natural coarticulation into a real
#     rendered pause instead of end-of-utterance decay. Cut positions are
#     found by counting the snippet's PHONEMES and snapping to the nearest
#     word onset: pure word counting broke on espeak's function-word fusion
#     ("from the" → "fɹʌmðə", one IPA word in context but two standalone —
#     heard as skipped words at a P6 seam), and pure phoneme counting
#     without onset snapping could land one gap off and clip a short word
#     ("the horn" → "horn"). Head cuts back off one frame into the gap and
#     tail cuts drop the trailing space run, because a word's acoustic
#     onset bleeds into the space frames before it.
#     NOTE: kittentts splits its input on [.!?] into independent renders, so
#     conditioning cannot cross a sentence boundary — a lookahead after a
#     sentence end comes back as its own run, contributes nothing, and is
#     dropped. Clause seams (';' ':' and the dialogue comma) get the full
#     effect.
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
_PUNCT_SET = set(_punctuation)


def _speech_phoneme_count(ph: str) -> int:
    """Tokens a phonemized string contributes to the model, excluding
    spaces/punctuation. Words can't be counted instead: espeak fuses
    function-word pairs with their neighbours ("from the" → "fɹʌmðə"),
    so a snippet's standalone word count disagrees with the full render's
    — but its phoneme count barely moves."""
    return sum(1 for c in ph
               if c in VOCAB and VOCAB[c] > SPACE_ID)




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


def _speech_onsets(ids) -> list:
    """(token_index, speech_tokens_before) for each speech-group onset. A
    group is a run of speech tokens; punctuation neither opens nor splits
    one, so trailing puncts and their pauses ride with the word they
    follow."""
    onsets, in_group, seen = [], False, 0
    for i, t in enumerate(ids):
        if t > SPACE_ID:
            if not in_group:
                onsets.append((i, seen))
            in_group = True
            seen += 1
        elif t == 0 or t == SPACE_ID:
            in_group = False
    return onsets


def _speech_total(ids) -> int:
    return sum(1 for t in ids if t > SPACE_ID)


def _context_cut(ids, dur, n_context_phonemes: int) -> int:
    """Sample offset where the context prefix ends: the word onset nearest
    the context's phoneme count (snapping absorbs espeak's context-
    dependent realizations), backed off one frame into the preceding gap
    so the first kept word keeps its attack — word onsets bleed a little
    into the space frames before them. Ties snap earlier: a duplicated
    sliver of context beats a clipped word."""
    onsets = _speech_onsets(ids)
    if not onsets:
        return 0
    idx, _ = min(onsets, key=lambda o: (abs(o[1] - n_context_phonemes), o[0]))
    cut = _cum_samples(dur)[idx]
    if idx > 0 and 0 < ids[idx - 1] <= SPACE_ID:
        cut = max(0, cut - FRAME)
    return cut


def _lookahead_cut(ids, dur, n_lookahead_phonemes: int):
    """Sample offset where the kept audio ends: locate the lookahead's
    first word onset (phoneme count + onset snap, ties later — keeping a
    sliver of lookahead beats clipping the last word), then walk back off
    the space run before it. The chunk keeps its own rendered punctuation
    pause but not the word gap, whose frames carry the next word's onset
    bleed. None → the run has no text of its own (caller falls back to
    the tail-pad trim)."""
    onsets = _speech_onsets(ids)
    target = _speech_total(ids) - n_lookahead_phonemes
    if target <= 0 or not onsets:
        return None
    idx, before = min(onsets,
                      key=lambda o: (abs(o[1] - target), -o[0]))
    if before == 0:
        return None  # cut would discard the whole run
    while idx > 1 and ids[idx - 1] == SPACE_ID:
        idx -= 1
    return _cum_samples(dur)[idx]


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


def _trim_run(ids, wav, dur, n_context_phonemes: int = 0,
              n_lookahead_phonemes: int = 0):
    """Return the speech-bearing slice of one raw (flat) run: context
    prefix / lookahead suffix (if any) cut at their word gaps, lead/tail
    non-speech reduced to small margins. Pure sequence ops — unit-tested
    without the model."""
    total = _cum_samples(dur)
    if len(wav) != total[-1]:  # duration contract broken; don't touch it
        return wav
    if n_context_phonemes > 0:
        start = _context_cut(ids, dur, n_context_phonemes)
    else:
        start = max(0, (int(dur[0]) - HEAD_KEEP)) * FRAME
    end = None
    if n_lookahead_phonemes > 0:
        end = _lookahead_cut(ids, dur, n_lookahead_phonemes)
    if end is None:
        tail_pad = max(0, _tail_silence_frames(ids, dur) - TAIL_KEEP) * FRAME
        end = len(wav) - tail_pad
    return wav[start:end] if start < end else wav


# ── Phoneme-direct path (Max's two-phase idea, 2026-07-29) ──────────────────
# The client phonemizes the WHOLE utterance once (op=phonemize below, a few
# ms even for paragraphs) and cuts it in phoneme space, so context/text/
# lookahead arrive as exact substrings of one consistent espeak output —
# fusion, sandhi and standalone-vs-context drift can't shift a cut, and the
# boundaries are exact token counts rather than counted-and-snapped guesses.
# Bypassing kittentts's generate() also skips its [.!?] splitter: windows
# may cross sentence boundaries (conditioning finally works at sentence-end
# seams) and the model sees real sentence-final punctuation for the first
# time (the wrapper strips .!? and substitutes commas).

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _ph_tokens(ph: str) -> list:
    """The wrapper's exact phoneme→token pipeline: split words/punctuation,
    rejoin with single spaces, map chars through the vocab (unknowns
    dropped). Mirrors basic_english_tokenize + TextCleaner."""
    return [VOCAB[c] for c in " ".join(_TOKEN_RE.findall(ph)) if c in VOCAB]


def _ph_request_ids(ctx: str, text: str, la: str):
    """(ids, i_text, i_la): full token stream BOS..EOS plus the exact token
    index where the text begins and where the lookahead begins (None
    without lookahead). Parts are joined by single space tokens, exactly
    as one flat phonemization would."""
    ids, i_text, i_la = [0], 1, None
    if ctx:
        ids += _ph_tokens(ctx)
        ids.append(SPACE_ID)
        i_text = len(ids)
    ids += _ph_tokens(text)
    if la:
        ids.append(SPACE_ID)
        i_la = len(ids)
        ids += _ph_tokens(la)
    ids += [10, 0]  # wrapper quirk: ellipsis token, then EOS pad
    return ids, i_text, i_la


def _synth_phonemes(om, req):
    """Exact-cut synthesis from pre-phonemized input. Same boundary policy
    as the text path — head cuts back off one frame into the gap, tail
    cuts stop before the boundary space — but at known indices."""
    import numpy as np
    import soundfile as sf

    ids, i_text, i_la = _ph_request_ids(
        (req.get("ph_context") or "").strip(),
        req["ph_text"].strip(),
        (req.get("ph_lookahead") or "").strip())

    voice = req.get("voice", "Kiki")
    voice = om.voice_aliases.get(voice, voice)
    speed = float(req.get("speed", 1.0)) * om.speed_priors.get(voice, 1.0)
    ref_id = min(len(req["ph_text"]), om.voices[voice].shape[0] - 1)

    out = om.session.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "style": om.voices[voice][ref_id:ref_id + 1],
        "speed": np.array([speed], dtype=np.float32),
    })
    wav = np.asarray(out[0]).reshape(-1)
    dur = np.asarray(out[1]).ravel()
    cum = _cum_samples(dur)
    if len(wav) != cum[-1]:
        raise RuntimeError("duration contract broken on phoneme path")

    if i_text > 1:
        start = max(0, cum[i_text] - FRAME)
    else:
        start = max(0, (int(dur[0]) - HEAD_KEEP)) * FRAME
    if i_la is not None:
        end = cum[i_la - 1]  # before the boundary space
    else:
        end = len(wav) - max(0, _tail_silence_frames(ids, dur)
                             - TAIL_KEEP) * FRAME
    sf.write(req["out"], wav[start:end] if start < end else wav, SAMPLE_RATE)


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
    lookahead = (req.get("lookahead") or "").strip()

    n_ctx = n_la = 0
    if context or lookahead:
        backend = getattr(om, "phonemizer", None)
        if backend is None:
            return False
        # Word counts come from the phonemized snippet, not the raw text,
        # so espeak expansions ("42" → two words) stay consistent with the
        # full render's token stream.
        if context:
            n_ctx = _speech_phoneme_count(backend.phonemize([context])[0])
            if n_ctx == 0:
                context = ""
        if lookahead:
            n_la = _speech_phoneme_count(backend.phonemize([lookahead])[0])
            if n_la == 0:
                lookahead = ""
    full = " ".join(s for s in (context, text, lookahead) if s)

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

    # kittentts splits on [.!?], so conditioning never crosses a sentence
    # boundary: a context ending in one comes back as its own leading run
    # (or several) and a lookahead past one as its own trailing run — they
    # conditioned nothing, so drop them whole instead of rendering them.
    # The +2 margin absorbs espeak's context-dependent realizations.
    while n_ctx and len(runs) > 1:
        run_ph = _speech_total(runs[0][0])
        if run_ph > n_ctx + 2:
            break
        runs = runs[1:]
        n_ctx = max(0, n_ctx - run_ph)
    if n_la and len(runs) > 1 and _speech_total(runs[-1][0]) <= n_la + 2:
        runs = runs[:-1]
        n_la = 0

    gap = np.zeros(int(SAMPLE_RATE * RUN_GAP_MS / 1000), dtype=np.float32)
    pieces = []
    for i, (ids, wav, dur) in enumerate(runs):
        if pieces:
            pieces.append(gap)
        # Only the first run contains the context prefix; only the last
        # contains the lookahead suffix.
        pieces.append(_trim_run(ids, np.asarray(wav).reshape(-1),
                                np.asarray(dur).ravel(),
                                n_ctx if i == 0 else 0,
                                n_la if i == len(runs) - 1 else 0))
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
    if req.get("op") == "phonemize":
        # Phase 1 of the phoneme-direct path: espeak the whole utterance
        # once (~2ms/paragraph) with the same preprocessor + fixups the
        # text path uses; result written as a text file to req["out"].
        om = model.model
        text = req["text"]
        pre = getattr(om, "preprocessor", None)
        if pre is not None:
            text = pre(text)
        ph = om.phonemizer.phonemize([text])[0]
        with open(req["out"], "w", encoding="utf-8") as f:
            f.write(ph)
        return
    if "ph_text" in req:
        _synth_phonemes(model.model, req)  # no text fallback: exact or error
        return
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
