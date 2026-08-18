"""Per-fragment stream planning — plan ONE arriving text fragment at a time.

``stream_play`` plans a whole utterance up front: it has the entire text
before the first render. A live stream doesn't — the fragments arrive one
at a time while the model is still writing (the voice loop releases a
sentence at a time), and the plan for fragment N must be made before
fragment N+1 exists. This module is that planner, kept pure so it can be
tested without a model, a socket, or a clock.

Data flow
---------
    stream daemon: op:text "<fragment>"
      → plan_fragment(engine, fragment, step=<ramp offset>, sentence=k)
          ├─ PHONEME_STREAM engine ──→ engine.phonemize(fragment)
          │                            → chunking.ph_stream_plan(ph, band=…)
          ├─ TEXT_CLAUSE_PLAN engine → chunking.clause_chunks(fragment)
          │                            → engine.phonemize(chunk) per chunk
          └─ anything else ──────────→ chunking.chunk_for_streaming(fragment)
          → list[RenderItem] + a graded trailing gap for the seam to the
            NEXT fragment
      → FragmentPlan(items, next_step)
      → daemon renders each item (``RenderItem.synthesize``) → chunk WAV

What carries across a fragment boundary, and what must not
----------------------------------------------------------
* **The ramp step offset carries.** ``chunking``'s chunk-size ramp starts
  small (time to first audio) and grows (per-chunk overhead). A stream
  that re-planned each fragment from step 0 would keep paying the small
  early sizes forever, so ``plan_fragment`` takes a ``step`` and returns
  the next one. On the phoneme path the offset is applied by *shifting
  the band's ramp tuple*, which is exactly equivalent to indexing it from
  ``step`` (``ph_stream_plan`` indexes by its own piece count).
* **The run gap carries**, graded by whether the fragment ended a
  sentence — see ``fragment_gap_ms``. The final fragment gets one too:
  the daemon cannot know a fragment is last until ``op:end`` arrives, and
  a trailing pause on the last chunk is inaudible (ratified spec).
* **Conditioning must NOT carry.** Context/lookahead across a run gap
  gives continuation prosody that contradicts the inserted silence —
  Max's "really bad" verdict on J2c, already encoded inside
  ``ph_stream_plan`` for runs within one plan. Fragments ARE sentence
  runs, so the same prohibition applies to the seam between them: the
  first item of a fragment never has context, the last never has
  lookahead. Conditioning INSIDE a fragment is untouched.
* **Style rows are per fragment, not per utterance.** kokoro's stock rule
  (``STYLE_ROWS="ph-utterance"``) indexes the pack by the whole
  utterance's phoneme length — unavailable mid-stream, and wrong to
  guess. Mid-stream the fragment IS the sentence, so rows come from
  per-run phoneme length ("ph-sentence" shape). Engines that index rows
  by text length keep doing that, per fragment.

An engine with no ``PHONEME_STREAM`` falls back to text-space chunking of
each fragment — same ramp-offset and gap rules, no phonemes involved.
"""

from __future__ import annotations

from . import chunking, perfstats
from .stream_play import (CONTEXT_WORDS, KEEP_TERMINAL_MARKS, LOOKAHEAD_WORDS,
                          PAD_MARKS)

# A fragment ends a sentence when its last non-closing character is a
# terminal mark (CJK enders included, mirroring chunking's sentence split).
_SENTENCE_ENDERS = ".!?。！？"
_TRAILING_CLOSERS = "\"'”’)]»"


class RenderItem:
    """One render → one chunk WAV → one ``ev:chunk`` on the wire.

    ``text`` is what the engine is asked to render — a phoneme string when
    ``phonemes`` is set, otherwise plain text. ``sentence``/
    ``sentence_text`` are wire bookkeeping: the index of the fragment this
    item came from within its utterance, and that fragment's original
    text (the client files whole fragments as spoken, so it needs the
    fragment's own words, not a sub-chunk's).
    """

    __slots__ = ("text", "phonemes", "context", "lookahead", "style_ref",
                 "gap_after_ms", "pad_marks", "sentence", "sentence_text")

    def __init__(self, text, *, phonemes=False, context=None, lookahead=None,
                 style_ref=None, gap_after_ms=0, pad_marks=None,
                 sentence=0, sentence_text=""):
        self.text = text
        self.phonemes = phonemes
        self.context = context
        self.lookahead = lookahead
        self.style_ref = style_ref
        self.gap_after_ms = gap_after_ms
        self.pad_marks = pad_marks
        self.sentence = sentence
        self.sentence_text = sentence_text

    # ── gate accounting ──────────────────────────────────────────────────
    @property
    def cond_chars(self) -> int:
        """Characters rendered and then cut away (conditioning)."""
        return len(self.context or "") + len(self.lookahead or "")

    def gate_item(self):
        """``(kept, rendered, gap_s)`` for ``stream_play.should_start``."""
        kept = len(self.text)
        return (kept, kept + self.cond_chars, self.gap_after_ms / 1000.0)

    # ── rendering ────────────────────────────────────────────────────────
    def synthesize(self, engine, out_path: str, **kwargs) -> None:
        """Render this item to ``out_path`` (no gap padding — the caller
        pads, so it can record perfstats on rendered audio only)."""
        if self.phonemes:
            engine.synthesize_phonemes(
                self.text, out_path, context=self.context,
                lookahead=self.lookahead, style_ref=self.style_ref,
                pad_marks=self.pad_marks, **kwargs)
            return
        kw = dict(kwargs)
        if self.context:
            kw["context"] = self.context
        if self.lookahead:
            kw["lookahead"] = self.lookahead
        engine.synthesize(self.text, out_path, **kw)

    def __eq__(self, other):
        return (isinstance(other, RenderItem)
                and all(getattr(self, s) == getattr(other, s)
                        for s in RenderItem.__slots__))

    def __repr__(self):
        return (f"RenderItem({self.text!r}, phonemes={self.phonemes}, "
                f"context={self.context!r}, lookahead={self.lookahead!r}, "
                f"style_ref={self.style_ref}, gap={self.gap_after_ms}, "
                f"sentence={self.sentence})")


class FragmentPlan:
    """``items`` for one fragment, plus the ramp ``next_step`` to hand the
    next fragment of the same utterance."""

    __slots__ = ("items", "next_step")

    def __init__(self, items, next_step):
        self.items = items
        self.next_step = next_step

    def __repr__(self):
        return f"FragmentPlan({self.items!r}, next_step={self.next_step})"


def ends_sentence(text: str) -> bool:
    """Does this fragment close a sentence? (Trailing quotes/brackets and
    whitespace don't hide the mark.)"""
    t = text.rstrip()
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] in _SENTENCE_ENDERS


def fragment_gap_ms(engine, text: str) -> int:
    """Silence appended after a fragment, before the next one starts.

    A fragment boundary is a render boundary the listener hears, so it
    gets the same treatment an in-plan boundary would: engines that grade
    their gaps (kitten's clause plan) get the sentence gap or the smaller
    clause gap; engines on the uniform-gap phoneme plan get
    ``RUN_GAP_MS`` at a sentence end. A fragment that does NOT end a
    sentence (a mid-sentence release) still gets the small clause gap —
    conditioning across the seam is prohibited, so the seam exists either
    way and a short beat reads better than a hard butt-join.
    """
    graded = (getattr(engine, "TEXT_CLAUSE_PLAN", None) is True
              or getattr(engine, "CLAUSE_GAPS", None) is True)
    if ends_sentence(text):
        return chunking.CLAUSE_SENT_GAP_MS if graded else chunking.RUN_GAP_MS
    return chunking.CLAUSE_GAP_MS


def _shift_band(band: chunking.StreamBand, step: int) -> chunking.StreamBand:
    """The same band with its ramp advanced by ``step`` chunks.

    ``ph_stream_plan``/``ph_pack`` index the ramp by how many pieces this
    plan has produced, so a mid-utterance fragment must not start at the
    ramp's small opening sizes again. Dropping ``step`` entries off the
    front is exactly that offset (the last entry repeats forever, as
    before). ``long_start`` — the alternate ramp for packing an
    utterance's FIRST run — is dropped: past step 0 there is no first run.
    """
    if step <= 0:
        return band
    ramp = tuple(band.ramp[min(step, len(band.ramp) - 1):])
    return chunking.StreamBand(band.name, ramp, band.context_units,
                               band.lookahead_units, None)


def _resolve_band(engine, engine_name: str, ph_key, band):
    """(band, chosen_here) — the caller's band, or the device's measured
    one (recorded back only when we picked it, as ``stream_play`` does)."""
    if band is not None:
        return band, False
    eng_bands = getattr(engine, "STREAM_BANDS", None)
    return chunking.band_for_rtf(
        perfstats.estimate_marginal(engine_name, ph_key),
        perfstats.band(engine_name, ph_key),
        bands=eng_bands if isinstance(eng_bands, tuple) else None), True


def _style_mode(engine) -> str:
    """Row rule for this engine, mid-stream.

    ``ph-utterance`` (kokoro's stock rule) is unavailable here — there is
    no whole utterance yet — so it degrades to the per-run phoneme-length
    rule over the fragment. Ratified spec: "kokoro style rows mid-stream:
    per-fragment phoneme length (ph-sentence shape)".
    """
    mode = getattr(engine, "STYLE_ROWS", None)
    if mode == "ph-utterance":
        return "ph-sentence"
    if mode == "ph-sentence":
        return mode
    return "text-sentence"


def _phoneme_items(engine, text, max_chars, synth_kwargs, engine_name,
                   ph_key, band, step):
    """Phoneme-space pieces for one fragment, or None when this engine
    can't do it (no PHONEME_STREAM, phonemization failed, empty result) —
    the caller then plans in text space. Never fatal: an optimization over
    a path that already works."""
    if not getattr(engine, "PHONEME_STREAM", False):
        return None
    if getattr(engine, "TEXT_CLAUSE_PLAN", None) is True:
        return _clause_items(engine, text, synth_kwargs)
    try:
        ph = engine.phonemize(text, **synth_kwargs)
    except Exception:  # daemon down, engine mid-refactor — use the text path
        return None
    if not ph or not ph.strip():
        return None

    band, chosen_here = _resolve_band(engine, engine_name, ph_key, band)
    mode = _style_mode(engine)
    pieces = chunking.ph_stream_plan(
        ph, max_chars, keep_marks=KEEP_TERMINAL_MARKS,
        band=_shift_band(band, step),
        text=text if mode == "text-sentence" else None,
        ph_rows=(mode == "ph-sentence"),
        # Strict identity checks: mocked engines return truthy attributes.
        split_quote_ends=getattr(engine, "QUOTE_END_RUNS", None) is True,
        # The eager first-chunk cut is a time-to-first-audio trick for the
        # START of a stream; mid-stream there is already audio buffered.
        eager_head=(step == 0
                    and getattr(engine, "EAGER_HEAD", None) is True))
    if not pieces:
        return None
    if chosen_here:
        perfstats.set_band(engine_name, ph_key, band.name)
    fallback_row = max(0, len(ph) - 1)
    return [(p, fallback_row) for p in pieces]


def _clause_items(engine, text, synth_kwargs):
    """TEXT_CLAUSE_PLAN engines (kitten): one piece per
    ``chunking.clause_chunks`` chunk, phonemized independently, no seam
    conditioning, gap graded by boundary type — the exact mirror of
    Android's pipeline that ``stream_play`` also uses. Rows index by the
    PRE-SPLIT sentence's text length, so they can't drift mid-sentence."""
    chunks = chunking.clause_chunks(text)
    if not chunks:
        return None
    out = []
    for i, c in enumerate(chunks):
        try:
            ph = engine.phonemize(c.text, **synth_kwargs)
        except Exception:  # daemon down — caller falls back to text path
            return None
        if not ph or not ph.strip():
            continue
        last = i == len(chunks) - 1
        gap = (0 if last
               else chunking.CLAUSE_SENT_GAP_MS if c.sentence_end
               else chunking.CLAUSE_GAP_MS)
        out.append((chunking.PhPiece(ph, None, None, gap,
                                     style_ref=len(c.row_text)),
                    len(text)))
    return out or None


def _text_items(engine, text, max_chars, step):
    """Text-space chunks for one fragment (engines with no phoneme path).

    Cross-chunk conditioning is applied only for engines that advertise
    it, and only INSIDE the fragment — never on the first/last chunk,
    which sit on fragment seams.
    """
    if max_chars is None:
        chunks = [text] if text.strip() else []
    else:
        chunks = chunking.chunk_for_streaming(text, max_chars, step=step)
    if not chunks:
        return []
    use_context = bool(getattr(engine, "STREAM_CONTEXT", False))
    use_lookahead = bool(getattr(engine, "STREAM_LOOKAHEAD", False))
    items = []
    for i, c in enumerate(chunks):
        ctx = (" ".join(chunks[i - 1].split()[-CONTEXT_WORDS:])
               if use_context and i > 0 else None)
        la = (" ".join(chunks[i + 1].split()[:LOOKAHEAD_WORDS])
              if use_lookahead and i < len(chunks) - 1 else None)
        items.append(RenderItem(c, context=ctx, lookahead=la))
    return items


def plan_fragment(engine, text: str, *, engine_name: str = "",
                  eng_cfg: dict | None = None,
                  synth_kwargs: dict | None = None,
                  sentence: int = 0, sentence_text: str | None = None,
                  step: int = 0, ph_key=None,
                  band: "chunking.StreamBand | None" = None) -> FragmentPlan:
    """Plan ONE fragment of a streaming utterance.

    ``step`` is the ramp offset carried from the previous fragments of the
    same utterance (0 for the first); the returned ``next_step`` is what
    the next fragment should be given. ``sentence``/``sentence_text`` are
    stamped onto every item for the wire. ``ph_key`` is the perfstats key
    for the phoneme path (``"<model>:ph"``), used for band selection.

    Returns an empty plan (``items == []``) for text that has nothing to
    say; ``next_step`` is then unchanged.
    """
    synth_kwargs = synth_kwargs or {}
    eng_cfg = eng_cfg or {}
    if not text or not text.strip():
        return FragmentPlan([], step)

    max_chars = chunking.resolve_max_chars(engine, eng_cfg)
    # A fragment is short by construction; when chunking is disabled the
    # whole fragment is one render, which is still a streamed chunk.
    plan_max = max_chars if max_chars else 10 ** 6

    pad = (None if getattr(engine, "TEXT_CLAUSE_PLAN", None) is True
           else PAD_MARKS)
    pieces = _phoneme_items(engine, text, plan_max, synth_kwargs,
                            engine_name, ph_key, band, step)
    if pieces is not None:
        items = [RenderItem(p.text, phonemes=True, context=p.context,
                            lookahead=p.lookahead,
                            style_ref=(p.style_ref if p.style_ref is not None
                                       else fallback),
                            gap_after_ms=p.gap_after_ms, pad_marks=pad)
                 for p, fallback in pieces]
    else:
        items = _text_items(engine, text, max_chars, step)

    if not items:
        return FragmentPlan([], step)

    # The seam to the NEXT fragment. Conditioning never crosses it, so the
    # boundary is audible either way — the gap is what makes it a pause
    # instead of a splice.
    items[-1].gap_after_ms = fragment_gap_ms(engine, text)

    for it in items:
        it.sentence = sentence
        it.sentence_text = text if sentence_text is None else sentence_text
    return FragmentPlan(items, step + len(items))
