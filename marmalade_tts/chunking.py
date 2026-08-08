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
# As above, but a sentence mark inside closing quotes also ends a run
# (ph_sentence_runs(split_quote_ends=True)).
_PH_SENTENCE_Q = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’']))(?=\s)")
_PH_TERMINAL_MARK = re.compile(r"([.!?])[\"”']*$")
_PH_TERMINAL = re.compile(r"[.!?]+(?=[\"”']*$)")

RUN_GAP_MS = 150      # silence between sentence runs (uniform-gap engines)

# Graded gaps for engines with CLAUSE_GAPS (the F rules — Max's 2026-08-07
# kitten clause-split ear-lab, mirrored from marmalade-tts-android). Kitten
# gives mid-render `,;:` only ~50 ms of pause vs ~390 ms for `.`, so every
# clause mark a listener should hear must be a real render boundary. With
# the renders' trimmed edges keeping ~135 ms of pause, effective silence ≈
# gap + 135 ms: clause ≈ 215 ms (between comma and period), sentence ≈
# 395 ms (matching the model's natural period — the uniform 150 ms made
# chunked periods pause LESS than unchunked ones).
CLAUSE_GAP_MS = 80        # after `;` `:` and the dialogue comma before a quote
CLAUSE_SENT_GAP_MS = 260  # after a sentence end

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
# saved are ~9% of render time.
#
# Below 2 was re-listened after the tail-cap fix (R14-1, decided
# 2026-07-31): 0 context is unacceptable, and 1+2 / 1+1 / 2+1 all share a
# similar audible seam — **2+2 is the settled default**. Max's mechanism,
# which the data supports: the model articulates a render's opening and
# closing words differently from mid-sentence words, and the cut lands at
# a conditioning word's onset — so a depth of 1 on either side leaves the
# chunk's KEPT edge word adjacent to the render boundary, colored as an
# opening/closing word. The second unit's job is to push the kept word a
# full word away from the boundary so it renders as mid-sentence. 2+1
# (closing-word coloring only; the tail cap tames the worst of it) is the
# most tolerable degraded form — sanctioned ONLY for a device that can't
# hold streaming or sub-1s TTFA otherwise, i.e. the slow band.
CONTEXT_UNITS = 2     # conditioning prefix, cut away after rendering
LOOKAHEAD_UNITS = 2   # conditioning suffix (i5)
SLOW_LOOKAHEAD_UNITS = 1  # R14-1: the sanctioned degraded depth (2+1)

# A chunk may end early on a clause mark once it is at least this fraction
# of the size target — clause ends are the nicest seams, but never at the
# cost of a chunk far below target (which would cost render overhead).
# 0.75 skipped P3's semicolons — the one boundary kokoro pauses at — and
# forced a word-gap cut through "turned hard" instead (Max's K1-4b flag);
# 0.5 puts the seams on marks. 0.35 was tried and REVERTED: many small
# clause chunks each drain the playback gate's budget (P3 TTFA 2.28s).
_CLAUSE_CLOSE_AT = 0.5

# Placement rules from the K1-4b/K1-5 listening rounds (2026-08-01):
_PH_STRESS = ("ˈ", "ˌ")  # misaki/espeak stress marks — a unit with neither
                         # is an unstressed function word
# A chunk may overshoot its target by this factor to reach a clause-mark
# seam (or, in ph_stream_plan, to keep a whole sentence in one chunk).
WHOLE_TOL = 1.35
_MARK_REACH = 3   # how many words ahead to look for that mark
_RUNT_TAIL = 16   # a final piece under this many chars merges back


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
#   slow (>0.35)     chunks grow to amortize the conditioning, the
#                    playback gate buffers more, and lookahead drops to
#                    1 unit — the one depth reduction Max sanctioned for
#                    a struggling device (R14-1, 2026-07-31: 2+1 is "the
#                    most tolerable" degraded form; context stays at 2
#                    because opening-word coloring is the worse seam).
_STREAM_BANDS = (
    # (name, upper marginal-RTF bound, ramp, context units, lookahead units)
    ("fast", 0.15, (60, 100, 160, 250, 400), CONTEXT_UNITS, LOOKAHEAD_UNITS),
    ("moderate", 0.35, (30, 60, 120, 220, 400), CONTEXT_UNITS, LOOKAHEAD_UNITS),
    ("slow", float("inf"), (90, 180, 320, 400), CONTEXT_UNITS,
     SLOW_LOOKAHEAD_UNITS),
)

# A band is only left once the estimate is this far past its edge.
BAND_HYSTERESIS = 0.15


class StreamBand:
    """``long_start`` is an alternate ramp for packing the utterance's
    FIRST run (optional 6th band-table column): a mid-sentence chunk 0
    banks speech-only audio — no trailing pause, no inter-run gap — so
    the playback gate needs a bigger opening buffer than a sentence-
    final chunk 0 provides for free (K1-4b: 44 left later deadlines
    short by a hair; 56 opened the gate)."""

    __slots__ = ("name", "ramp", "context_units", "lookahead_units",
                 "long_start")

    def __init__(self, name, ramp, context_units, lookahead_units,
                 long_start=None):
        self.name = name
        self.ramp = ramp
        self.context_units = context_units
        self.lookahead_units = lookahead_units
        self.long_start = long_start

    def __repr__(self):
        return f"StreamBand({self.name!r}, {self.ramp!r})"


def _band(entry) -> StreamBand:
    return StreamBand(entry[0], entry[2], entry[3], entry[4],
                      entry[5] if len(entry) > 5 else None)


def band_for_rtf(mrtf: "float | None", current: "str | None" = None,
                 bands: "tuple | None" = None) -> StreamBand:
    """Pick the chunk-size band for a measured marginal RTF.

    ``current`` is the band last used on this device; staying in it wins
    ties within ``BAND_HYSTERESIS`` of the edge. With no measurement yet,
    the first band is the optimistic default — the first render is the
    first measurement, and a wrong guess costs a gap, never a failure.

    ``bands`` substitutes an engine-declared table (same row format as
    ``_STREAM_BANDS``): the shared table encodes KITTEN's measured cost
    model, and an engine with a very different RTF/cps profile needs its
    own edges and ramps (see ``KokoroEngine.STREAM_BANDS``).
    """
    if bands is None:
        bands = _STREAM_BANDS
    if mrtf is None:
        return _band(bands[0])
    for i, entry in enumerate(bands):
        if mrtf <= entry[1]:
            chosen = i
            break
    else:  # pragma: no cover — the last band is unbounded
        chosen = len(bands) - 1
    if current is None or current == bands[chosen][0]:
        return _band(bands[chosen])
    names = [e[0] for e in bands]
    if current not in names:
        return _band(bands[chosen])
    cur = names.index(current)
    # Leaving a band needs the estimate to clear the relevant edge by the
    # hysteresis margin; otherwise stay put.
    if chosen > cur:
        edge = bands[cur][1]
        return _band(bands[chosen if mrtf > edge * (1 + BAND_HYSTERESIS)
                           else cur])
    edge = bands[cur - 1][1]
    return _band(bands[chosen if mrtf < edge * (1 - BAND_HYSTERESIS)
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


def ph_sentence_runs(ph: str, keep_marks: str = "",
                     split_quote_ends: bool = False) -> list[str]:
    """Split a phonemized utterance into sentence runs, swapping each run's
    terminal .!? for a comma. Marks listed in ``keep_marks`` are left alone
    (the model does render them — measured pauses period 225ms · ! 146 ·
    ? 114 — it is the wrapper that substitutes commas).

    ``split_quote_ends`` also ends a run at a sentence mark inside closing
    quotes ('!"' + space) — a real sentence end the plain split misses, so
    without it a dialogue-final sentence merges with the narration after
    it and gets cut mid-flow instead of at the pause (K1-5). Off by
    default: turning it on changes run boundaries (rows, gaps), so each
    engine's sound owner opts in (kokoro does; kitten's approved renders
    keep the merge)."""
    splitter = _PH_SENTENCE_Q if split_quote_ends else _PH_SENTENCE
    runs: list[str] = []
    for run in splitter.split(ph):
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
    chunk size is a performance knob now, not a quality one. Placement
    still matters (K1-4b/K1-5, Max's listens): three rules below decide
    WHERE the legal cuts land.

    * Reach for the mark: on overflow, if a clause-mark word lies within
      WHOLE_TOL of the target, run on to the mark instead of cutting at
      a word gap (a mark seam sits in a rendered pause; a word-gap seam
      may land in coarticulated voiced audio — "turned | hard").
    * Never end a chunk on unstressed units: function words cliticize
      onto the NEXT word, so cutting after them splits a spoken unit
      ("as the | market" → "cobblestones | as the market").
    * Runt tails merge back: a final piece under _RUNT_TAIL chars is not
      worth a seam if its predecessor can absorb it within tolerance.
    """
    words = run.split()
    out: list[str] = []
    cur: list[str] = []

    def _stressed(w):
        return any(m in w for m in _PH_STRESS)

    def close():
        # Back the cut off before any trailing unstressed run — provided
        # a stressed unit remains and the tail doesn't end a clause.
        k = len(cur)
        while (k > 1 and not _stressed(cur[k - 1])
                and cur[k - 1][-1] not in ";:,"):
            k -= 1
        if any(_stressed(w) for w in cur[:k]):
            out.append(" ".join(cur[:k]))
            rest = cur[k:]
        else:
            out.append(" ".join(cur))
            rest = []
        cur.clear()
        cur.extend(rest)

    def mark_within_reach(i, target):
        cap = min(target * WHOLE_TOL, max_chars)
        total = len(" ".join(cur + [words[i]]))
        for j in range(i, min(i + _MARK_REACH, len(words))):
            if j > i:
                total += 1 + len(words[j])
            if total > cap:
                return False
            core = words[j].rstrip('"”’\'')
            nxt = words[j + 1] if j + 1 < len(words) else None
            if (core and core[-1] in ";:,.!?"
                    and not (nxt and nxt[0] in '"“\'')):
                return True
        return False

    for i, w in enumerate(words):
        target = min(ramp[min(step + len(out), len(ramp) - 1)], max_chars)
        cand = len(" ".join(cur + [w]))
        if cur and cand > target and not mark_within_reach(i, target):
            close()
        cur.append(w)
        if (len(" ".join(cur)) >= _CLAUSE_CLOSE_AT * target
                and _ends_clause(w, words[i + 1] if i + 1 < len(words)
                                 else None)):
            out.append(" ".join(cur))
            cur.clear()
    if cur:
        close()
        if cur:
            out.append(" ".join(cur))
    subs = [s for s in out if s] or [run]
    if len(subs) >= 2 and len(subs[-1]) < _RUNT_TAIL:
        t_prev = min(ramp[min(step + len(subs) - 2, len(ramp) - 1)],
                     max_chars)
        merged = subs[-2] + " " + subs[-1]
        if len(merged) <= WHOLE_TOL * t_prev:
            subs[-2:] = [merged]
    return subs


# ── Eager chunk 0 (K1-5, Max's verdict 2026-08-01) ──────────────────────────
# A tiny pause-rich chunk 0 is what makes 0.35s TTFA possible (P10's
# "Stop!"). When the opening sentence is long, the same condition can be
# manufactured by cutting chunk 0 at a mark the model already pauses on:
# an em dash (probe: ~110ms natural pause) or the comma before a quote
# (389ms — the biggest pause in the sentence). The cut behaves like a run
# boundary — no context on what follows — and a gap top-up restores the
# mark's beat while banking playback-gate buffer. A dash head keeps ONE
# lookahead unit: rendered solo, "Wait —" gets utterance-final prosody
# (F0 falls 329→207Hz + 67% final lengthening, probe_kokoro6.py); la1
# flattens the fall at no measured render cost (la2 restores the full
# rise for ~0.11s more — Max picked la1). A quote head renders solo: Max
# approved that seam as-is.
_EAGER_MIN = 5        # ph chars a head must reach to be worth cutting
_EAGER_HEAD_MAX = 60  # absolute cap so streamed and saved plans agree
                      # (the saved path plans with a flat max-size ramp)
_EAGER_DASH_GAP_MS = 300   # natural ~110ms — Max expects a real beat
_EAGER_QUOTE_GAP_MS = 350  # natural 389ms; the solo render keeps ~110
_EAGER_DASH_LA = 1


def eager_head_cut(run: str, target: int):
    """Cut the utterance's first chunk at a strong pause mark — an em
    dash, or a ``;:,``-final word right before an opening quote. Returns
    ``(head, rest, gap_ms, lookahead_units)`` or None."""
    cap = min(target * WHOLE_TOL, _EAGER_HEAD_MAX)
    words = run.split()
    acc: list[str] = []
    for i, w in enumerate(words[:-1]):
        acc.append(w)
        pref = " ".join(acc)
        if len(pref) > cap:
            return None
        if len(pref) < _EAGER_MIN or not any(
                m in u for u in acc for m in _PH_STRESS):
            continue
        nxt = words[i + 1]
        if w.endswith("—"):
            return (pref, " ".join(words[i + 1:]),
                    _EAGER_DASH_GAP_MS, _EAGER_DASH_LA)
        if w[-1] in ";:," and nxt[0] in '"“\'':
            return pref, " ".join(words[i + 1:]), _EAGER_QUOTE_GAP_MS, 0
    return None


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


class ClauseChunk:
    """One chunk of the F plan — EXACT mirror of Android's
    ``TextChunker.ClauseChunk`` (marmalade-tts-android, c33838e). The two
    implementations must stay behavior-identical, fixtures and all: Max's
    parity rule (2026-08-07) is that a chunking bug found on one platform
    is thereby found on both.

    ``row_text`` is the PRE-SPLIT sentence — style rows must index by its
    length, not the fragment's, so the register never shifts mid-sentence.
    ``sentence_end`` True → the engine's sentence gap follows; False →
    clause gap (`;` `:`, dialogue intro, newline ending in a comma).
    Always False on the last chunk."""

    __slots__ = ("text", "row_text", "sentence_end")

    def __init__(self, text, row_text, sentence_end):
        self.text = text
        self.row_text = row_text
        self.sentence_end = sentence_end

    def __eq__(self, other):
        return (isinstance(other, ClauseChunk)
                and (self.text, self.row_text, self.sentence_end)
                == (other.text, other.row_text, other.sentence_end))

    def __repr__(self):
        return (f"ClauseChunk({self.text!r}, {self.row_text!r}, "
                f"{self.sentence_end})")


_TERMINAL_CLOSERS = "\"'”’)]"
_SENTENCE_OPENERS = "\"“‘'"
_DIALOGUE_INTRO = re.compile(r'(?<=[,:])\s+(?=["“])')
_CLAUSE_MARK = re.compile(r"(?<=[:;])\s+")


def clause_chunks(text: str) -> list[ClauseChunk]:
    """The F chunking rules — port of Android ``TextChunker.clauseChunks``:
    quote-aware sentence ends, newline = sentence boundary, dialogue-intro
    cut at ``said, "``, clause cuts at ``;`` ``:``, NO merging, never
    word-split. See the Android original for the full rationale."""
    out: list[ClauseChunk] = []
    sentences = _sentences_quote_aware(text.strip())
    for si, s in enumerate(sentences):
        frags = [f.strip()
                 for part in _DIALOGUE_INTRO.split(s)
                 for f in _CLAUSE_MARK.split(part)
                 if f.strip()]
        for fi, f in enumerate(frags):
            last_of_sentence = fi == len(frags) - 1
            last_of_text = si == len(sentences) - 1 and last_of_sentence
            out.append(ClauseChunk(
                text=f,
                row_text=s,
                # A "sentence" ending in a comma is a newline-split
                # continuation (list item) — comma pause, not period.
                sentence_end=(last_of_sentence and not last_of_text
                              and not s.endswith(",")),
            ))
    return out


def _sentences_quote_aware(text: str) -> list[str]:
    """Port of Android ``sentencesQuoteAware``: ``.!?`` + optional closing
    quotes/brackets ends a sentence; with closers present the next word
    must start uppercase/digit/opening-quote (lowercase = attribution,
    stays attached). Plain ``.!?`` + whitespace always cuts. Newlines and
    CJK enders (。！？) are sentence boundaries too."""
    out: list[str] = []
    start = 0
    i = 0

    def emit(end_exclusive):
        t = text[start:end_exclusive].strip()
        if t:
            out.append(t)

    while i < len(text):
        c = text[i]
        if c == "\n":
            emit(i)
            while i < len(text) and text[i].isspace():
                i += 1
            start = i
        elif c in "。！？":
            emit(i + 1)
            i += 1
            start = i
        elif c in ".!?":
            j = i + 1
            while j < len(text) and text[j] in _TERMINAL_CLOSERS:
                j += 1
            k = j
            while k < len(text) and text[k].isspace():
                k += 1
            if k > j and k < len(text):
                nxt = text[k]
                plain = j == i + 1
                if (plain or nxt.isupper() or nxt.isdigit()
                        or nxt in _SENTENCE_OPENERS):
                    emit(j)
                    start = k
                    i = k
                    continue
            i = j
        else:
            i += 1
    emit(len(text))
    return out


def ph_stream_plan(ph: str, max_chars: int, keep_marks: str = "",
                   gap_ms: int = RUN_GAP_MS,
                   band: "StreamBand | None" = None,
                   text: "str | None" = None,
                   ph_rows: bool = False,
                   split_quote_ends: bool = False,
                   eager_head: bool = False) -> list[PhPiece]:
    """The full streaming plan for one phonemized utterance.

    ``text`` is the pre-phonemization utterance; when given, each run's
    pieces carry the run's style row (see ``_run_style_refs``).
    ``ph_rows`` instead derives each run's row from its own PHONEME length
    (kokoro indexes its pack by phoneme count — ``pack[len(ps)-1]``).
    With neither, ``style_ref`` stays None and the caller picks a row.
    ``split_quote_ends`` is ph_sentence_runs' flag, ``eager_head`` is
    ``eager_head_cut`` (both engine opt-ins). Engines with
    ``TEXT_CLAUSE_PLAN`` (kitten) never reach this — they plan in text
    space via ``clause_chunks`` for exact Android parity."""
    if band is None:
        band = band_for_rtf(None)
    runs = ph_sentence_runs(ph, keep_marks, split_quote_ends)
    if text:
        rows = _run_style_refs(ph, runs, text)
    elif ph_rows:
        rows = [max(0, len(r) - 1) for r in runs]
    else:
        rows = [None] * len(runs)
    pieces: list[PhPiece] = []
    for ri, run in enumerate(runs):
        t = band.ramp[min(len(pieces), len(band.ramp) - 1)]
        if eager_head and not pieces:
            head = eager_head_cut(run, t)
            if head:
                pref, run, gap, la_units = head
                la = (" ".join(run.split()[:la_units]) or None) \
                    if la_units else None
                pieces.append(PhPiece(pref, None, la, gap,
                                      style_ref=rows[ri]))
                t = band.ramp[min(len(pieces), len(band.ramp) - 1)]
        # A sentence within WHOLE_TOL of its ramp target stays whole
        # (K1-4b): a sentence end is a free unconditioned boundary with
        # the approved seam sound, so a modest overshoot beats cutting
        # into the sentence. The 35 floor keeps short-sentence openers
        # whole even at small early targets.
        if len(run) <= min(max(t * WHOLE_TOL, 35), max_chars):
            subs = [run]
        else:
            ramp = (band.long_start
                    if band.long_start and not pieces else band.ramp)
            subs = ph_pack(run, max_chars, step=len(pieces), ramp=ramp)
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
