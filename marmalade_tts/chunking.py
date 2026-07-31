"""Text chunking and WAV concatenation.

Each engine declares a soft character limit (``MAX_CHARS`` on the engine
class, or ``engines.<name>.max_chars`` in config). When a single user
input exceeds the limit, the CLI silently splits it on sentence
boundaries, synthesizes each chunk, and concatenates the resulting WAVs
into the requested output. The user sees one WAV out for one input in
— that's the contract.

This is **not** batch mode. Batch mode (``--batch``) is the per-line one-
WAV-per-line behavior. Chunking is transparent: it preserves the
input → output count.
"""

from __future__ import annotations

import os
import re
import shutil
import wave

# Sentence boundary: punctuation followed by whitespace. Lookbehind keeps
# the punctuation attached to the preceding chunk.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Paragraph boundary: one or more blank lines.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into pieces no longer than ``max_chars`` characters.

    Cascade of split strategies, each finer than the last:
      1. Whole text already fits → return ``[text]``.
      2. Paragraph splits (``\\n\\n``); each paragraph chunked recursively.
      3. Sentence splits; sentences greedily packed up to the limit.
      4. Single overlong sentence → word splits.

    Never returns empty strings. Strips whitespace from each piece.
    """
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text] if text.strip() else []

    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    if len(paragraphs) > 1:
        out: list[str] = []
        for p in paragraphs:
            out.extend(chunk_text(p, max_chars))
        return out

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if len(sentences) > 1:
        return _pack_sentences(sentences, max_chars)

    # One long sentence — fall back to word splits.
    return _split_by_words(text, max_chars)


def _pack_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Greedy bin-pack of sentences up to ``max_chars`` per bin."""
    out: list[str] = []
    cur = ""
    for s in sentences:
        candidate = (cur + " " + s).strip() if cur else s
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            out.append(cur)
        if len(s) <= max_chars:
            cur = s
        else:
            # A single sentence is itself overlong — drop down to words.
            out.extend(_split_by_words(s, max_chars))
            cur = ""
    if cur:
        out.append(cur)
    return out


def _split_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip() if cur else w
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = w  # if a single word is longer than max_chars, it stays as-is
    if cur:
        out.append(cur)
    return out


# Clause boundary for streaming chunks. Mid-sentence cuts sound like the
# TTS was cut off and restarted (2026-07-28 listening lab), so streaming
# only ever breaks at: sentence ends (incl. inside closing quotes),
# semicolons, colons, and newlines. Commas are NOT boundaries — including
# the dialogue-intro comma before a quote, which was one until Max's
# 2026-07-29 lab round flagged that seam as the one audible break.
_CLAUSE_BOUNDARY = re.compile(
    r'(?<=[.!?;:])[)"”\']?\s+'   # clause-final punct (+ closing quote)
    r'|\s*\n+\s*'                # explicit line breaks
)

# Streaming chunk-size ramp: early chunks stay small so the playback gate
# can open after little buffered audio without a later chunk missing its
# deadline; later chunks grow to amortize per-chunk overhead. Values are
# character targets — actual chunks end on clause boundaries.
_STREAM_RAMP = (60, 100, 160, 250, 400)


def _clause_units(text: str) -> list[str]:
    """Split at clause boundaries, keeping all punctuation (closing quotes,
    the dialogue comma) attached to the preceding unit — the separator match
    ends exactly where the next clause begins."""
    units: list[str] = []
    pos = 0
    for m in _CLAUSE_BOUNDARY.finditer(text):
        u = text[pos:m.end()].strip()
        if u:
            units.append(u)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        units.append(tail)
    return units


def chunk_for_streaming(text: str, max_chars: int) -> list[str]:
    """Chunk ``text`` for streamed playback: clause-boundary cuts only,
    with ramped sizes (small first for time-to-first-audio, growing after).

    A clause unit longer than ``max_chars`` falls back to ``chunk_text``
    word-splitting for that unit — unavoidable, and rare in real prose.
    """
    units: list[str] = []
    for u in _clause_units(text):
        if len(u) > max_chars:
            units.extend(chunk_text(u, max_chars))
        else:
            units.append(u)
    if len(units) <= 1:
        return units

    out: list[str] = []
    cur = ""
    for u in units:
        target = _STREAM_RAMP[min(len(out), len(_STREAM_RAMP) - 1)]
        target = min(target, max_chars)
        if cur and len(cur) + 1 + len(u) > target:
            out.append(cur)
            cur = u
        else:
            cur = (cur + " " + u) if cur else u
    if cur:
        out.append(cur)
    return out


# ── Phoneme-space streaming plan ────────────────────────────────────────────
# The kitten daemon can phonemize a whole utterance in one espeak call
# (~2ms for a paragraph) and synthesize from phonemes directly. Planning the
# stream in phoneme space instead of text space is strictly better, and the
# 2026-07-29/30 listening rounds decided every part of it:
#
#   * Cuts are exact substrings of ONE espeak output, so espeak's
#     context-dependent fusion ("from the" → "fɹʌmðə") can't shift a
#     boundary — the text path had to count phonemes and snap to word
#     onsets, and still only approximated it.
#   * Sentence runs split on the mark itself, so closing quotes stay with
#     their sentence. The text path split inside quotations ('"Where did
#     you put the keys,' + '" she asked,') and destroyed the question —
#     this is why the phoneme pipeline beat the wrapper outright on
#     dialogue (P9) and instructions (P10).
#   * Terminal marks render for real (round 15: the wrapper's comma swap
#     lost or tied on every mark once tested in isolation). The swap
#     machinery stays parameterized (``keep_marks``) for A/B work.
#   * Runs are joined by a uniform 150ms gap. Mark-proportional gaps
#     (J2g) gave no audible win.
#   * Conditioning (context/lookahead) applies ONLY inside a sentence run.
#     Across a run gap it gives continuation prosody that contradicts the
#     inserted silence — Max's "really bad" verdict on J2c.

_PH_SENTENCE = re.compile(r"(?<=[.!?])(?=\s)")
_PH_TERMINAL_MARK = re.compile(r"([.!?])[\"”']*$")
_PH_TERMINAL = re.compile(r"[.!?]+(?=[\"”']*$)")

RUN_GAP_MS = 150      # silence between sentence runs

# Conditioning depth, counted in ESPEAK'S OWN ATOMIC UNITS — the
# whitespace-separated tokens of the phoneme string — not in English
# words. espeak fuses function words with their neighbours ("from the" →
# "fɹʌmðə", one unit), so a unit is what the model actually treats as a
# word, and slicing the phoneme string is exact by construction. On the
# legacy text path these are English words and only approximately this.
#
# Depth was swept on 2026-07-30 (lab round 13, P3 + P6 at 60-char chunks).
# Max: 2+2 is indistinguishable from the original 4+2 on both passages, so
# 2 it is — 4 was never more than the first number tried, and the two units
# saved are ~9% of render time. Below 2 the seams degrade, for two reasons
# worth keeping straight: a 1-unit LOOKAHEAD made the model treat the chunk
# as utterance-final and render a runaway pause (an implementation bug,
# fixed by the daemon's tail cap), while a 1-unit CONTEXT leaves the model
# still in utterance-initial prosody where the chunk's own audio begins —
# that one is inherent, not a bug. 1+1 is still open pending a re-listen
# now that the pause bug is gone.
CONTEXT_UNITS = 2     # conditioning prefix, cut away after rendering
LOOKAHEAD_UNITS = 2   # conditioning suffix (i5)

# A chunk may end early on a clause mark once it is at least this fraction
# of the size target — clause ends are the nicest seams, but never at the
# cost of a chunk far below target (which would cost render overhead).
_CLAUSE_CLOSE_AT = 0.75


# ── Chunk-size bands ────────────────────────────────────────────────────────
# Chunk size is a performance knob (round-7 verdict: mid-sentence cuts sound
# fine once trimmed and conditioned), so it should follow the device rather
# than be a constant. Measured cost on kitten nano, 2026-07-30 (probe:
# ~/coding/scratch/chunk-lab/rtf_bands.py, 5 reps per size):
#
#     bare render_s = 0.021 + 0.092 × audio_s      (fit over 20–400 ph chars)
#     conditioning  = ~2.0s of audio rendered and thrown away per chunk at
#                     the 4+2 units measured, independent of the chunk's
#                     own size (roughly proportionally less at today's 2+2)
#
# So per-chunk overhead is dominated by conditioning, not by the request:
# at 20 ph chars the chunk pays ×1.58 its own render; at 250 chars, ×1.09.
# That is exactly Max's footgun — shrinking chunks makes measured RTF
# worse, so a naive closed loop spirals. Two things stop it here:
#
#   1. Sizing reads the MARGINAL rtf (perfstats.estimate_marginal), which
#      divides by the audio actually rendered including the discarded
#      conditioning, and is therefore size-independent.
#   2. Bands are stepped with a hysteresis margin, so a device sitting on
#      an edge doesn't alternate.
#
# Band choice, from the cost model with a 0.6s time-to-first-audio target
# and a ceiling of ~0.7 on the SUSTAINED effective RTF (above that the
# playback gate stops opening early and streaming buys nothing):
#
#   fast (≤0.15)     conditioning is nearly free — keep the full ramp.
#   moderate (≤0.35) conditioning still fits under the ceiling, but the
#                    first chunk must shrink to hold TTFA under ~0.8s.
#   slow (>0.35)     chunks grow to amortize the conditioning, and the
#                    playback gate buffers more. Conditioning itself is
#                    NOT dropped: it is the quality of the seams, and
#                    trading it away to make a number fit was not a
#                    decision anyone signed off (Max, 2026-07-30). The
#                    real lever is how deep conditioning has to be —
#                    see CONTEXT_UNITS.
_STREAM_BANDS = (
    # (name, upper marginal-RTF bound, ramp, context units, lookahead units)
    ("fast", 0.15, (60, 100, 160, 250, 400), CONTEXT_UNITS, LOOKAHEAD_UNITS),
    ("moderate", 0.35, (30, 60, 120, 220, 400), CONTEXT_UNITS, LOOKAHEAD_UNITS),
    ("slow", float("inf"), (90, 180, 320, 400), CONTEXT_UNITS, LOOKAHEAD_UNITS),
)

# A band is only left once the estimate is this far past its edge.
BAND_HYSTERESIS = 0.15


class StreamBand:
    __slots__ = ("name", "ramp", "context_units", "lookahead_units")

    def __init__(self, name, ramp, context_units, lookahead_units):
        self.name = name
        self.ramp = ramp
        self.context_units = context_units
        self.lookahead_units = lookahead_units

    def __repr__(self):
        return f"StreamBand({self.name!r}, {self.ramp!r})"


def _band(entry) -> StreamBand:
    return StreamBand(entry[0], entry[2], entry[3], entry[4])


def band_for_rtf(mrtf: "float | None",
                 current: "str | None" = None) -> StreamBand:
    """Pick the chunk-size band for a measured marginal RTF.

    ``current`` is the band last used on this device; staying in it wins
    ties within ``BAND_HYSTERESIS`` of the edge. With no measurement yet,
    the fast band is the optimistic default — the first render is the
    first measurement, and a wrong guess costs a gap, never a failure.
    """
    if mrtf is None:
        return _band(_STREAM_BANDS[0])
    for i, entry in enumerate(_STREAM_BANDS):
        if mrtf <= entry[1]:
            chosen = i
            break
    else:  # pragma: no cover — the last band is unbounded
        chosen = len(_STREAM_BANDS) - 1
    if current is None or current == _STREAM_BANDS[chosen][0]:
        return _band(_STREAM_BANDS[chosen])
    names = [e[0] for e in _STREAM_BANDS]
    if current not in names:
        return _band(_STREAM_BANDS[chosen])
    cur = names.index(current)
    # Leaving a band needs the estimate to clear the relevant edge by the
    # hysteresis margin; otherwise stay put.
    if chosen > cur:
        edge = _STREAM_BANDS[cur][1]
        return _band(_STREAM_BANDS[chosen if mrtf > edge * (1 + BAND_HYSTERESIS)
                                   else cur])
    edge = _STREAM_BANDS[cur - 1][1]
    return _band(_STREAM_BANDS[chosen if mrtf < edge * (1 - BAND_HYSTERESIS)
                               else cur])


class PhPiece:
    """One rendered unit of a phoneme-space stream plan.

    ``style_ref`` is the style-pack row for this piece — the character
    count of the sentence (run) the piece belongs to, so every sub-chunk
    of a sentence shares its sentence's register and rows can't drift
    mid-sentence (P11). None means the caller picks (legacy pinned row).
    The engine clamps to its pack's row count."""

    __slots__ = ("text", "context", "lookahead", "gap_after_ms", "style_ref")

    def __init__(self, text, context=None, lookahead=None, gap_after_ms=0,
                 style_ref=None):
        self.text = text
        self.context = context
        self.lookahead = lookahead
        self.gap_after_ms = gap_after_ms
        self.style_ref = style_ref

    def __eq__(self, other):
        return (isinstance(other, PhPiece)
                and (self.text, self.context, self.lookahead,
                     self.gap_after_ms, self.style_ref)
                == (other.text, other.context, other.lookahead,
                    other.gap_after_ms, other.style_ref))

    def __repr__(self):
        return (f"PhPiece({self.text!r}, {self.context!r}, "
                f"{self.lookahead!r}, {self.gap_after_ms}, "
                f"{self.style_ref})")


def ph_sentence_runs(ph: str, keep_marks: str = "") -> list[str]:
    """Split a phonemized utterance into sentence runs, swapping each run's
    terminal .!? for a comma. Marks listed in ``keep_marks`` are left alone
    (the model does render them — measured pauses period 225ms · ! 146 ·
    ? 114 — it is the wrapper that substitutes commas)."""
    runs: list[str] = []
    for run in _PH_SENTENCE.split(ph):
        run = run.strip()
        if not run:
            continue
        m = _PH_TERMINAL_MARK.search(run)
        if not (m and m.group(1) in keep_marks):
            run = _PH_TERMINAL.sub(",", run)
        runs.append(run)
    return runs


def _ends_clause(word: str, next_word: str | None) -> bool:
    """True if ``word`` closes a clause we may cut after. A clause mark
    followed by an opening quote is not a boundary — the dialogue comma
    before a quotation was the one seam Max singled out as audible."""
    if not word or word[-1] not in ";:,":
        return False
    return not (next_word and next_word[0] in '"“\'')


def ph_pack(run: str, max_chars: int, step: int = 0,
            ramp: "tuple[int, ...]" = _STREAM_RAMP) -> list[str]:
    """Pack one sentence run into ramp-sized sub-chunks at word gaps,
    closing on a clause mark when one lands near the target.

    Word-gap cuts inside a sentence were rejected in round 1 and accepted
    in round 7 once seams were trimmed and conditioned both ways (C2/D2) —
    chunk size is a performance knob now, not a quality one.
    """
    words = run.split()
    out: list[str] = []
    cur = ""
    for i, w in enumerate(words):
        target = min(ramp[min(step + len(out), len(ramp) - 1)], max_chars)
        cand = (cur + " " + w) if cur else w
        if cur and len(cand) > target:
            out.append(cur)
            cur = w
            continue
        cur = cand
        if (len(cur) >= _CLAUSE_CLOSE_AT * target
                and _ends_clause(w, words[i + 1] if i + 1 < len(words) else None)):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out or [run]


def _run_style_refs(ph: str, ph_runs: list[str], text: str) -> list[int]:
    """Per-run style rows, stock-faithful: the upstream wrapper indexes the
    pack by each sentence's TEXT character count (verified in
    ``_prepare_inputs``: ``min(len(text), rows-1)`` on the text chunk —
    phoneme chars run ~1.1× text chars, and neighbouring rows are audible,
    P11). Its ``[.!?]+`` split drops the mark and appends a comma, so our
    sentence-with-its-mark length equals stock's counted length.

    The text is split with the same rule as the phoneme runs; when the
    counts disagree (espeak erases abbreviation dots the text split trips
    over), fall back to scaling each run's phoneme length by the
    utterance-wide text/phoneme ratio — register-faithful, never crashes.
    """
    text_runs = [r.strip() for r in _PH_SENTENCE.split(text) if r.strip()]
    if len(text_runs) == len(ph_runs):
        return [len(r) for r in text_runs]
    ratio = len(text) / len(ph) if ph else 1.0
    return [round(len(r) * ratio) for r in ph_runs]


def ph_stream_plan(ph: str, max_chars: int, keep_marks: str = "",
                   gap_ms: int = RUN_GAP_MS,
                   band: "StreamBand | None" = None,
                   text: "str | None" = None) -> list[PhPiece]:
    """The full streaming plan for one phonemized utterance.

    ``text`` is the pre-phonemization utterance; when given, each run's
    pieces carry the run's style row (see ``_run_style_refs``). Without it
    ``style_ref`` stays None and the caller picks a row."""
    if band is None:
        band = band_for_rtf(None)
    runs = ph_sentence_runs(ph, keep_marks)
    rows = _run_style_refs(ph, runs, text) if text else [None] * len(runs)
    pieces: list[PhPiece] = []
    for ri, run in enumerate(runs):
        subs = ph_pack(run, max_chars, step=len(pieces), ramp=band.ramp)
        last_run = ri == len(runs) - 1
        for i, s in enumerate(subs):
            last_sub = i == len(subs) - 1
            ctx = (" ".join(subs[i - 1].split()[-band.context_units:])
                   if i and band.context_units else None)
            la = (" ".join(subs[i + 1].split()[:band.lookahead_units])
                  if not last_sub and band.lookahead_units else None)
            pieces.append(PhPiece(
                s, ctx, la, gap_ms if last_sub and not last_run else 0,
                style_ref=rows[ri]))
    return pieces


def pad_wav_end(path: str, ms: int) -> None:
    """Append ``ms`` of silence to a WAV in place (the inter-run gap)."""
    if ms <= 0:
        return
    with wave.open(path, "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    n = int(params.framerate * ms / 1000)
    with wave.open(path, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
        dst.writeframes(b"\0" * (n * params.sampwidth * params.nchannels))


def concat_wavs(in_paths: list[str], out_path: str) -> None:
    """Concatenate WAVs end-to-end into ``out_path``.

    All inputs must have identical sample rate, channels, and sample width
    — which is always true for chunks produced by the same engine in one
    synthesis call. Empty input list raises ``ValueError``.
    """
    if not in_paths:
        raise ValueError("concat_wavs: no input files")

    if len(in_paths) == 1:
        if os.path.abspath(in_paths[0]) != os.path.abspath(out_path):
            shutil.copyfile(in_paths[0], out_path)
        return

    with wave.open(in_paths[0], "rb") as src:
        nchannels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()

    with wave.open(out_path, "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        for p in in_paths:
            with wave.open(p, "rb") as src:
                if (src.getnchannels(), src.getsampwidth(),
                        src.getframerate()) != (nchannels, sampwidth, framerate):
                    raise ValueError(
                        f"concat_wavs: format mismatch — {p} has "
                        f"{src.getnchannels()}ch/{src.getsampwidth()*8}-bit/"
                        f"{src.getframerate()}Hz; expected {nchannels}ch/"
                        f"{sampwidth*8}-bit/{framerate}Hz."
                    )
                dst.writeframes(src.readframes(src.getnframes()))


def resolve_max_chars(engine, eng_cfg: dict) -> int | None:
    """Effective per-engine character limit.

    Config (``engines.<name>.max_chars``) overrides the engine's class
    attribute. ``None``, ``0``, or any non-positive-int value disables
    chunking. Robust to mocked engines (where attribute access can return
    arbitrary objects).
    """
    if "max_chars" in eng_cfg:
        v = eng_cfg["max_chars"]
    else:
        v = getattr(engine, "MAX_CHARS", None)
    # bool is a subclass of int in Python — exclude it explicitly so
    # `MAX_CHARS = True` is treated as "not configured".
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return None
    return v
