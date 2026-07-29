"""Chunk-streamed playback — start playing before the full render finishes.

For a single long utterance that chunks, the old flow rendered everything,
concatenated, then played: time-to-first-audio = the whole render. Here
chunks render in the background (parallel when the engine allows it) and
playback starts as soon as the buffered audio can outlast the estimated
remaining render time — one inequality, fed by the observed per-engine
RTF/CPS averages in ``perfstats``:

    start when  SAFETY * est_remaining_render <= buffered + drain

``drain`` is the audio time that will play before the *last* chunk is
needed (everything remaining except that last chunk) — playback itself
buys render time, and the last chunk's deadline is the binding one. That
single rule scales across devices and engines with no tuning tiers: a
fast engine starts right after the first chunk; an engine near RTF 1
buffers a few chunks; one above RTF 1 buffers most of the input — exactly
the behavior needed to stay gapless. Cold start (no stats yet) simply
waits for the first chunk's render, which itself becomes the first
measurement. If the estimate is ever wrong, playback degrades to a gap
(the consumer waits for the next chunk), never to corruption.

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


def should_start(buffered_audio_s: float, remaining_chunk_chars: list[int],
                 est: "tuple[float, float] | None", workers: int,
                 safety: float = SAFETY) -> bool:
    """The playback gate. ``remaining_chunk_chars`` lists the sizes of the
    not-yet-rendered chunks; ``est`` is (rtf, chars_per_audio_s) or None."""
    if not remaining_chunk_chars:
        return True
    if est is None:
        return False
    rtf, cps = est
    if cps <= 0:
        return False
    audio = [c / cps for c in remaining_chunk_chars]
    est_render_wall = rtf * sum(audio) / max(1, workers)
    # Audio time available before the last unrendered chunk is needed:
    # the buffer plus every remaining chunk that plays before it.
    drain = buffered_audio_s + sum(audio) - audio[-1]
    return safety * est_render_wall <= drain


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
    chunks = chunking.chunk_for_streaming(processed, max_chars)
    if len(chunks) < 2:
        return None

    mkey = perfstats.model_key(engine, eng_cfg)
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

    def _render_one(i: int):
        t0 = time.monotonic()
        engine.synthesize(chunks[i], tmp_paths[i], **synth_kwargs)
        dt = time.monotonic() - t0
        dur = cli.wav_duration(tmp_paths[i])
        perfstats.record(engine_name, mkey, len(chunks[i]), dt, dur)
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
        remaining = [len(chunks[j]) for j in range(n) if j not in ready]
        return should_start(buffered, remaining,
                            perfstats.estimate(engine_name, mkey), workers)

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
