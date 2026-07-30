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
#   * Terminal .!? are swapped for a comma AFTER espeak has seen the real
#     marks, so the model gets sentence-final phonology with the wrapper's
#     pausing (which Max preferred to continuous real-mark rendering).
#   * Runs are joined by a uniform 150ms gap. Mark-proportional gaps
#     (J2g) gave no audible win.
#   * Conditioning (context/lookahead) applies ONLY inside a sentence run.
#     Across a run gap it gives continuation prosody that contradicts the
#     inserted silence — Max's "really bad" verdict on J2c.

_PH_SENTENCE = re.compile(r"(?<=[.!?])(?=\s)")
_PH_TERMINAL_MARK = re.compile(r"([.!?])[\"”']*$")
_PH_TERMINAL = re.compile(r"[.!?]+(?=[\"”']*$)")

RUN_GAP_MS = 150      # silence between sentence runs
CONTEXT_WORDS = 4     # conditioning prefix, cut away after rendering
LOOKAHEAD_WORDS = 2   # conditioning suffix (i5)

# A chunk may end early on a clause mark once it is at least this fraction
# of the size target — clause ends are the nicest seams, but never at the
# cost of a chunk far below target (which would cost render overhead).
_CLAUSE_CLOSE_AT = 0.75


class PhPiece:
    """One rendered unit of a phoneme-space stream plan."""

    __slots__ = ("text", "context", "lookahead", "gap_after_ms")

    def __init__(self, text, context=None, lookahead=None, gap_after_ms=0):
        self.text = text
        self.context = context
        self.lookahead = lookahead
        self.gap_after_ms = gap_after_ms

    def __eq__(self, other):
        return (isinstance(other, PhPiece)
                and (self.text, self.context, self.lookahead,
                     self.gap_after_ms)
                == (other.text, other.context, other.lookahead,
                    other.gap_after_ms))

    def __repr__(self):
        return (f"PhPiece({self.text!r}, {self.context!r}, "
                f"{self.lookahead!r}, {self.gap_after_ms})")


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


def ph_stream_plan(ph: str, max_chars: int, keep_marks: str = "",
                   gap_ms: int = RUN_GAP_MS) -> list[PhPiece]:
    """The full streaming plan for one phonemized utterance."""
    runs = ph_sentence_runs(ph, keep_marks)
    pieces: list[PhPiece] = []
    for ri, run in enumerate(runs):
        subs = ph_pack(run, max_chars, step=len(pieces))
        last_run = ri == len(runs) - 1
        for i, s in enumerate(subs):
            last_sub = i == len(subs) - 1
            pieces.append(PhPiece(
                s,
                " ".join(subs[i - 1].split()[-CONTEXT_WORDS:]) if i else None,
                (" ".join(subs[i + 1].split()[:LOOKAHEAD_WORDS])
                 if not last_sub else None),
                gap_ms if last_sub and not last_run else 0,
            ))
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
