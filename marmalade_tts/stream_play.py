"""Chunk-streamed playback — start playing before the full render finishes.

For a single long utterance that chunks, the old flow rendered everything,
concatenated, then played: time-to-first-audio = the whole render. Here
chunks render in the background (parallel when the engine allows it) and
playback starts as soon as the buffered audio can outlast the estimated
remaining render time — one inequality, fed by the observed per-engine
RTF/CPS averages in ``perfstats``:

    every remaining chunk must render before the play head reaches it:
    for each unrendered chunk k (in order),
        SAFETY * est_ready_time(k) <= buffered + audio_before(k)

Playback itself buys render time, so a sub-realtime engine starts after
the first chunk; an engine near RTF 1 buffers a few chunks; one above
RTF 1 buffers most of the input. The per-chunk check matters: with big
chunks the *next* chunk's deadline binds, not the last one's — a large
chunk 2 can miss its slot even when the total render fits. Render-time
estimates are deliberately sequential-pessimistic (no division by
workers): parallel engines open the gate slightly late rather than gap.
Combined with the ramped chunk sizes from ``chunk_for_streaming`` (small
early, larger late), early deadlines stay cheap and TTFA stays low.
Cold start (no stats yet) simply waits for the first chunk's render,
which itself becomes the first measurement. If the estimate is ever
wrong, playback degrades to a gap (the consumer waits for the next
chunk), never to corruption.

Engines that advertise ``PHONEME_STREAM`` (kitten in daemon mode) plan the
whole stream in phoneme space instead of text: one espeak call up front,
sentence runs split on the real marks, sub-sentence chunks conditioned at
exact token indices, uniform inter-run gaps, per-run style rows (each
sentence keeps its own length's register, sub-chunks share it). See
``chunking.ph_stream_plan``. Everything below the plan — the gate, the
worker pool, perfstats — is identical either way.

Used only when: playing (not --no-play), single utterance, no effects
(effects are applied to whole files and would differ chunk-by-chunk),
and the text actually chunks. Everything else keeps the old path.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time

from . import chunking, perfstats

SAFETY = 1.5

# Cross-chunk prosody conditioning: engines advertising STREAM_CONTEXT get
# the last words of the previous chunk as rendered-then-discarded context,
# so chunk N+1 opens as a continuation instead of a fresh utterance.
# STREAM_LOOKAHEAD is the mirror image (Max's i5, 2026-07-29 lab verdict):
# the next chunk's opening words render after the text and are cut away,
# so the chunk's last word coarticulates into a real rendered pause
# instead of end-of-utterance decay.
CONTEXT_WORDS = 4
LOOKAHEAD_WORDS = 2

# Phoneme-direct streaming (engines advertising PHONEME_STREAM): the whole
# utterance is phonemized once and the stream is planned in phoneme space —
# see chunking.ph_stream_plan for what each part buys and which listening
# round decided it.
#
# KEEP_TERMINAL_MARKS: terminal marks NOT swapped for a comma before
# rendering. All real, none swapped — Max's round-15 verdict ("all real is
# the best") once the swap was finally tested in isolation. The swap had
# ridden in from the kittentts wrapper on a confounded A/B (J vs J2 changed
# run-splitting at the same time); the clean per-mark tests showed the real
# mark equal on '!' and better on '.' and '?' — a comma ending makes the
# model drawl toward a continuation that never comes (P9 10.82s → 9.45s).
KEEP_TERMINAL_MARKS = ".!?"
# The model renders an intra-sentence colon in ~87ms, which reads as no
# pause at all; topped up to a floor of inserted silence.
PAD_MARKS = {":": 150}


def should_start(buffered_audio_s: float, remaining_chunk_chars: list,
                 est: "tuple[float, float] | None",
                 safety: float = SAFETY) -> bool:
    """The playback gate. ``remaining_chunk_chars`` lists the sizes of the
    not-yet-rendered chunks in play order — plain char counts,
    ``(kept_chars, rendered_chars)`` pairs when conditioning makes a chunk
    render more audio than it keeps, or ``(kept, rendered, gap_s)``
    triples when the chunk also carries deterministic inserted silence
    (inter-run gaps, mark top-ups): the gap plays before the next chunk
    is needed, so it buys render time exactly like kept audio. ``est`` is
    (rtf, chars_per_audio_s) or None; with pairs, ``rtf`` should be the
    MARGINAL rtf (render cost per rendered second) so the estimate is
    chunk-size-independent — the plain per-kept-second EMA rises on small
    chunks (fixed overhead), and on an engine like kokoro that poisoned
    the gate into waiting extra chunks for nothing (the same size→rtf
    feedback the marginal stat was built to kill). Every remaining chunk
    must be renderable before the play head reaches it."""
    if not remaining_chunk_chars:
        return True
    if est is None:
        return False
    rtf, cps = est
    if cps <= 0:
        return False
    t_ready = 0.0                    # est render-completion time of chunk k
    t_needed = buffered_audio_s      # play head reaches chunk k at this time
    for item in remaining_chunk_chars:
        gap_s = 0.0
        if isinstance(item, int):
            kept = rendered = item
        elif len(item) == 2:
            kept, rendered = item
        else:
            kept, rendered, gap_s = item
        t_ready += rtf * (rendered / cps)
        if safety * t_ready > t_needed:
            return False
        t_needed += kept / cps + gap_s
    return True


def _phoneme_plan(engine, text: str, max_chars: int, synth_kwargs: dict,
                  engine_name: str, mkey: "str | None", band=None):
    """(plan, style_ref) for the phoneme-direct path, or (None, None) when
    the engine doesn't support it or phonemization fails — the caller then
    plans in text space. Never fatal: this is an optimization over a path
    that already works.

    A caller may pass its own ``band`` when it has no time-to-first-audio
    deadline to protect (the non-streamed render in ``synth`` uses one flat
    max-size target); by default chunk sizes follow the device's measured
    marginal RTF."""
    if not getattr(engine, "PHONEME_STREAM", False):
        return None, None
    try:
        ph = engine.phonemize(text, **synth_kwargs)
    except Exception:  # daemon down, engine mid-refactor — use the text path
        return None, None
    if not ph or not ph.strip():
        return None, None
    # Chunk sizes and conditioning depth follow the device's measured
    # marginal RTF (chunking._STREAM_BANDS explains the cost model and
    # what each band trades away).
    banded = band is None
    if banded:
        # Strict tuple check: mocked engines return truthy attributes.
        eng_bands = getattr(engine, "STREAM_BANDS", None)
        band = chunking.band_for_rtf(
            perfstats.estimate_marginal(engine_name, mkey),
            perfstats.band(engine_name, mkey),
            bands=eng_bands if isinstance(eng_bands, tuple) else None)
    # How style rows are chosen is per-engine. kitten's pack is indexed by
    # TEXT sentence length (stock-faithful per-run registers — Max's R16-1
    # verdict); kokoro's by PHONEME length with no sentence split, so its
    # stock-faithful row is utterance-wide ("ph-utterance"; per-run rows =
    # "ph-sentence" — the K1 lab A/B decides). Anything unrecognized (incl.
    # mocked engines) gets kitten's rule.
    mode = getattr(engine, "STYLE_ROWS", None)
    if mode not in ("ph-utterance", "ph-sentence"):
        mode = "text-sentence"
    plan = chunking.ph_stream_plan(
        ph, max_chars, keep_marks=KEEP_TERMINAL_MARKS, band=band,
        text=text if mode == "text-sentence" else None,
        ph_rows=(mode == "ph-sentence"),
        # Strict identity checks: mocked engines return truthy attributes.
        split_quote_ends=getattr(engine, "QUOTE_END_RUNS", None) is True,
        eager_head=getattr(engine, "EAGER_HEAD", None) is True)
    if plan and banded:
        perfstats.set_band(engine_name, mkey, band.name)
    # Pieces carry per-run style rows; the second value is the fallback for
    # a piece without one (mode-matched: ph-length rule for the ph modes).
    fallback = max(0, len(ph) - 1) if mode != "text-sentence" else len(ph)
    return (plan, fallback) if plan else (None, None)


def _effective_workers(engine, n_rest: int) -> int:
    if getattr(engine, "PARALLEL_CHUNKS", False):
        return max(1, min(4, os.cpu_count() or 1, n_rest))
    return 1


def try_stream_single(
    utt: str,
    out_path: str,
    *,
    engine,
    engine_name: str,
    eng_cfg: dict,
    config: dict,
    synth_kwargs: dict,
    preprocess_mode,
    custom_rules: list | None,
    play=None,
):
    """Stream-render+play one utterance. Returns a SynthResult, or None
    when this input doesn't qualify (no chunking → caller uses the normal
    path). The concatenated WAV still lands at ``out_path`` afterwards —
    the one-input-one-WAV contract holds.
    """
    from . import cli, synth
    if play is None:
        from .playback import play_wav as play

    processed = synth.apply_preprocessing(
        utt, engine_name=engine_name, eng_cfg=eng_cfg, config=config,
        preprocess_mode=preprocess_mode, custom_rules=custom_rules)
    if not processed.strip():
        return None

    max_chars = chunking.resolve_max_chars(engine, eng_cfg)
    if max_chars is None:
        return None

    # Phoneme chars ≈ 1.1× text chars, so the two paths keep separate
    # stats: a shared chars-per-audio-second EMA would drift the gate.
    mkey = perfstats.model_key(engine, eng_cfg)
    ph_key = f"{mkey}:ph"

    plan, style_ref = _phoneme_plan(engine, processed, max_chars,
                                    synth_kwargs, engine_name, ph_key)
    if plan is not None:
        chunks = [p.text for p in plan]
        mkey = ph_key
    else:
        chunks = chunking.chunk_for_streaming(processed, max_chars)
    if len(chunks) < 2:
        return None
    n = len(chunks)
    workers = _effective_workers(engine, n - 1)

    tmp_paths: list[str] = []
    for i in range(n):
        fd, p = tempfile.mkstemp(prefix=f"marmalade-stream-{i:03d}-",
                                 suffix=".wav")
        os.close(fd)
        tmp_paths.append(p)

    ready: dict[int, float] = {}          # chunk index → audio seconds
    errors: list[BaseException] = []
    cond = threading.Condition()

    use_context = bool(getattr(engine, "STREAM_CONTEXT", False))
    use_lookahead = bool(getattr(engine, "STREAM_LOOKAHEAD", False))

    def _render_one(i: int):
        t0 = time.monotonic()
        kwargs = dict(synth_kwargs)
        if plan is not None:
            piece = plan[i]
            row = piece.style_ref if piece.style_ref is not None else style_ref
            engine.synthesize_phonemes(
                piece.text, tmp_paths[i], context=piece.context,
                lookahead=piece.lookahead, style_ref=row,
                pad_marks=PAD_MARKS, **kwargs)
        else:
            if use_context and i > 0:
                kwargs["context"] = " ".join(
                    chunks[i - 1].split()[-CONTEXT_WORDS:])
            if use_lookahead and i < n - 1:
                kwargs["lookahead"] = " ".join(
                    chunks[i + 1].split()[:LOOKAHEAD_WORDS])
            engine.synthesize(chunks[i], tmp_paths[i], **kwargs)
        dt = time.monotonic() - t0
        # Recorded audio duration is post-trim while render time includes
        # the discarded context prefix — the EMA absorbs the overhead, so
        # the gate stays honest about the true cost per emitted second.
        dur = cli.wav_duration(tmp_paths[i])
        cond_chars = 0
        if plan is not None:
            cond_chars = (len(plan[i].context or "")
                          + len(plan[i].lookahead or ""))
        # Chunk 0 always renders alone (it warms the daemon), so it is the
        # one uncontended sample per utterance — the honest read on device
        # speed that band selection needs.
        perfstats.record(engine_name, mkey, len(chunks[i]), dt, dur,
                         cond_chars=cond_chars, solo=(i == 0))
        if plan is not None and plan[i].gap_after_ms:
            # Inter-run silence rides on the chunk that closes the run, so
            # it plays and concatenates with no extra bookkeeping. Recorded
            # after perfstats: it is not rendered audio.
            chunking.pad_wav_end(tmp_paths[i], plan[i].gap_after_ms)
            dur += plan[i].gap_after_ms / 1000.0
        with cond:
            ready[i] = dur
            cond.notify_all()

    def _guarded(fn, *a):
        try:
            fn(*a)
        except BaseException as e:  # surfaces to the consumer via cond
            with cond:
                errors.append(e)
                cond.notify_all()

    def _coordinate():
        # First chunk alone: warms/auto-starts the daemon race-free and
        # doubles as the cold-start RTF measurement.
        _guarded(_render_one, 0)
        with cond:
            if errors:
                return
        idx_q: "queue.SimpleQueue[int]" = queue.SimpleQueue()
        for i in range(1, n):
            idx_q.put(i)

        def _worker():
            while True:
                try:
                    i = idx_q.get_nowait()
                except queue.Empty:
                    return
                with cond:
                    if errors:
                        return
                _guarded(_render_one, i)

        for _ in range(workers):
            threading.Thread(target=_worker, daemon=True,
                             name="marmalade-stream-render").start()

    def _gate_open(next_play: int) -> bool:
        buffered = 0.0
        i = next_play
        while i in ready:
            buffered += ready[i]
            i += 1
        if plan is not None:
            # Phoneme path: conditioning chars are known exactly, so the
            # gate can budget rendered (kept + discarded) audio at the
            # size-independent marginal rtf — see should_start.
            remaining = [(len(chunks[j]),
                          len(chunks[j]) + len(plan[j].context or "")
                          + len(plan[j].lookahead or ""),
                          plan[j].gap_after_ms / 1000.0)
                         for j in range(n) if j not in ready]
        else:
            remaining = [len(chunks[j]) for j in range(n) if j not in ready]
        est = perfstats.estimate(engine_name, mkey)
        if plan is not None and est is not None:
            mrtf = perfstats.estimate_marginal(engine_name, mkey)
            if mrtf is not None:
                est = (mrtf, est[1])
        return should_start(buffered, remaining, est)

    threading.Thread(target=_coordinate, daemon=True,
                     name="marmalade-stream-coord").start()

    played = 0
    started = False
    total_dur = 0.0
    try:
        while played < n:
            with cond:
                while played not in ready and not errors:
                    cond.wait()
                if errors:
                    raise errors[0]
                if not started:
                    while (not errors and len(ready) < n
                           and not _gate_open(played)):
                        cond.wait()
                    if errors:
                        raise errors[0]
                    started = True
            play(tmp_paths[played])  # outside the lock — blocking playback
            total_dur += ready[played]
            played += 1
        chunking.concat_wavs(tmp_paths, out_path)
    finally:
        for p in tmp_paths:
            if os.path.abspath(p) != os.path.abspath(out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return synth.SynthResult(out=out_path, text=processed, raw_text=utt,
                             duration=total_dur)
