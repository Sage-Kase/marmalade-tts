"""Synthesis loop — runs an engine over one or more utterances.

Used by both cli.py (interactive / batch / streaming) and mcp_server.py
(single-call ``synthesize`` tool). Keeping the loop here means
preprocessing, effects, duration measurement, and ordering rules apply
uniformly to both entry points.

The streaming branch (``run_batch(streaming=True, on_ready=...)``) spawns
a producer thread that renders utterances sequentially and fires
``on_ready(result)`` for each one as it lands; the caller blocks on the
callback (typically to play the WAV). This is Option A from the refactor
spec — the (synth + queue + sentinel) coupling stays in one place and the
CLI just supplies a small playback consumer via the callback.

``wav_duration`` is looked up through the ``cli`` module at call time so
test patches on ``marmalade_tts.cli.wav_duration`` (used heavily by the
streaming tests) keep working after the refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import cli_helpers


@dataclass
class SynthResult:
    """One rendered utterance.

    Kept dict-compatible (via ``__getitem__``) because downstream helpers
    in ``cli_helpers`` (subtitles, --json reporting) were written against
    the dict shape the old ``_synthesize_one`` returned. Reusing the same
    access pattern lets the rest of the refactor stay a pure mechanical
    move."""

    out: str
    text: str         # post-preprocessing text actually sent to the engine
    raw_text: str     # original user input (used for subtitles)
    duration: float

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def apply_preprocessing(
    utt: str,
    *,
    engine_name: str,
    eng_cfg: dict,
    config: dict,
    preprocess_mode,
    custom_rules: list | None,
) -> str:
    """Resolve the preprocessing decision + rule set and apply them.

    Shared by ``synthesize_one`` and the chunk-streaming path
    (``stream_play``) so both send engines identical text.
    """
    from . import preprocessing as pp

    if preprocess_mode is True:
        do_preprocess = True
    elif preprocess_mode is False:
        do_preprocess = False
    else:
        do_preprocess = config.get("defaults", {}).get("preprocessing", True)
        eng_pp = eng_cfg.get("preprocessing")
        if eng_pp is not None:
            if isinstance(eng_pp, bool):
                do_preprocess = eng_pp
            elif isinstance(eng_pp, list):
                do_preprocess = True

    if not do_preprocess:
        return utt

    rules = custom_rules
    if rules is None:
        cfg_rules = eng_cfg.get("preprocessing")
        if isinstance(cfg_rules, list):
            rules = cfg_rules
    if rules is not None:
        return pp.preprocess(utt, engine=engine_name, rules=rules)
    return pp.preprocess(utt, engine=engine_name)


def _phoneme_render(engine, engine_name: str, eng_cfg: dict, processed: str,
                    out_path: str, synth_kwargs: dict,
                    max_chars: "int | None") -> bool:
    """Render one utterance through the phoneme pipeline (kitten in daemon
    mode), so a saved WAV sounds like the same text streamed: real terminal
    marks, padded colons, uniform inter-run gaps, per-run style rows.
    Without it the wrapper's text path applies its own rules (comma-swapped
    marks, blind tail trim) and the two outputs audibly diverge.

    Chunks use one flat max-size target — there is no first-audio deadline
    here, so seams stay at the minimum the engine's input limit forces.

    Returns False (never raises) when the engine lacks the phoneme path or
    phonemization fails — the caller then renders via the text path.
    """
    from . import chunking, cli, perfstats, stream_play

    # Strict identity check: mocked engines return truthy attributes.
    if max_chars is None or getattr(engine, "PHONEME_STREAM", False) is not True:
        return False
    mkey = f"{perfstats.model_key(engine, eng_cfg)}:ph"
    band = chunking.StreamBand("full", (max_chars,),
                               chunking.CONTEXT_UNITS,
                               chunking.LOOKAHEAD_UNITS)
    plan, style_ref = stream_play._phoneme_plan(
        engine, processed, max_chars, synth_kwargs, engine_name, mkey,
        band=band)
    if plan is None:
        return False

    import os
    import tempfile
    import time

    def _render(i: int, path: str):
        piece = plan[i]
        row = piece.style_ref if piece.style_ref is not None else style_ref
        t0 = time.monotonic()
        engine.synthesize_phonemes(
            piece.text, path, context=piece.context,
            lookahead=piece.lookahead, style_ref=row,
            pad_marks=stream_play.PAD_MARKS, **synth_kwargs)
        dt = time.monotonic() - t0
        try:
            cond_chars = (len(piece.context or "")
                          + len(piece.lookahead or ""))
            perfstats.record(engine_name, mkey, len(piece.text), dt,
                             cli.wav_duration(path),
                             cond_chars=cond_chars, solo=(i == 0))
        except Exception:
            pass
        if piece.gap_after_ms:
            chunking.pad_wav_end(path, piece.gap_after_ms)

    if len(plan) == 1:
        _render(0, out_path)
        return True

    tmp_paths: list[str] = []
    try:
        for i in range(len(plan)):
            fd, p = tempfile.mkstemp(
                prefix=f"marmalade-chunk-{i:03d}-", suffix=".wav")
            os.close(fd)
            tmp_paths.append(p)

        # First piece alone: warms/auto-starts the daemon race-free, and is
        # the one uncontended perfstats sample (solo=True).
        _render(0, tmp_paths[0])

        rest = list(range(1, len(plan)))
        workers = 1
        if getattr(engine, "PARALLEL_CHUNKS", False):
            workers = min(4, os.cpu_count() or 1, len(rest))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda i: _render(i, tmp_paths[i]), rest))
        else:
            for i in rest:
                _render(i, tmp_paths[i])
        chunking.concat_wavs(tmp_paths, out_path)
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    return True


def _text_render(engine, engine_name: str, eng_cfg: dict, processed: str,
                 out_path: str, synth_kwargs: dict,
                 max_chars: "int | None") -> None:
    """The text-path render: transparent chunking + ``engine.synthesize``.

    If the text exceeds ``max_chars``, split on sentence boundaries,
    synthesize each chunk into a temp WAV, and concatenate into
    ``out_path``. The user-visible contract is one input → one WAV;
    chunking is implementation detail.
    """
    from . import chunking, cli

    if max_chars is not None and len(processed) > max_chars:
        chunks = chunking.chunk_text(processed, max_chars)
    else:
        chunks = None

    # Every clean render (pre-effects) feeds the per-engine+model RTF/CPS
    # averages that the chunk-streaming gate consumes. Best-effort: stats
    # must never break synthesis.
    import time as _time
    from . import perfstats
    _mkey = perfstats.model_key(engine, eng_cfg)

    def _render(piece: str, path: str):
        t0 = _time.monotonic()
        engine.synthesize(piece, path, **synth_kwargs)
        try:
            perfstats.record(engine_name, _mkey, len(piece),
                             _time.monotonic() - t0, cli.wav_duration(path))
        except Exception:
            pass

    if not chunks or len(chunks) == 1:
        _render(processed, out_path)
        return

    import os as _os
    import tempfile as _tempfile
    tmp_paths: list[str] = []
    try:
        for i in range(len(chunks)):
            fd, p = _tempfile.mkstemp(
                prefix=f"marmalade-chunk-{i:03d}-", suffix=".wav")
            _os.close(fd)
            tmp_paths.append(p)

        # The first chunk always renders alone: it warms/auto-starts the
        # engine's daemon, so parallel submissions can't race the spawn.
        _render(chunks[0], tmp_paths[0])

        rest = list(zip(chunks[1:], tmp_paths[1:]))
        workers = 1
        if getattr(engine, "PARALLEL_CHUNKS", False):
            workers = min(4, _os.cpu_count() or 1, len(rest))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # list() drains the iterator so the first chunk error
                # propagates; output order is fixed by tmp_paths.
                list(pool.map(lambda cp: _render(cp[0], cp[1]), rest))
        else:
            for piece, p in rest:
                _render(piece, p)
        chunking.concat_wavs(tmp_paths, out_path)
    finally:
        for p in tmp_paths:
            try:
                _os.unlink(p)
            except OSError:
                pass


def synthesize_one(
    utt: str,
    out_path: str,
    *,
    engine,
    engine_name: str,
    eng_cfg: dict,
    config: dict,
    synth_kwargs: dict,
    effect_list: list,
    preprocess_mode,        # None = use config default; True/False = force
    custom_rules: list | None,
) -> SynthResult | None:
    """Synthesize a single utterance: preprocess → synth → effects → measure.

    ``preprocess_mode``:
      * ``None`` — use the same config-driven default the CLI uses
        (``defaults.preprocessing`` plus per-engine overrides).
      * ``True`` / ``False`` — force preprocessing on or off, ignoring
        config. Used by MCP (always True) and by ``--preprocessing`` /
        ``--no-preprocessing``.

    ``custom_rules`` overrides the per-engine default rule set when given;
    ``None`` means "use the engine's defaults". Matches the legacy
    ``eng_cfg.get('preprocessing')``-as-list behavior the CLI inherits.

    Returns ``None`` when the preprocessed text is empty (whitespace-only
    line, or stripped to nothing) — the caller should skip it silently to
    preserve batch behavior.
    """
    # Lazy import to avoid module-load circular: cli imports synth, synth
    # only needs cli's namespace at call time (for the patchable
    # ``wav_duration`` reference the streaming tests rely on).
    from . import cli

    # ── Preprocessing ──
    processed = apply_preprocessing(
        utt, engine_name=engine_name, eng_cfg=eng_cfg, config=config,
        preprocess_mode=preprocess_mode, custom_rules=custom_rules)

    if not processed.strip():
        return None

    # ── Synthesis + effects ──
    from . import chunking
    max_chars = chunking.resolve_max_chars(engine, eng_cfg)

    # Engines with a phoneme pipeline (kitten in daemon mode) render
    # through it even when the text wouldn't chunk, so saved WAVs match
    # the streamed sound. Falls back to the text path on any miss.
    if not _phoneme_render(engine, engine_name, eng_cfg, processed, out_path,
                           synth_kwargs, max_chars):
        _text_render(engine, engine_name, eng_cfg, processed, out_path,
                     synth_kwargs, max_chars)

    cli_helpers.apply_effects_if_any(out_path, effect_list, config)

    # ── Duration (AFTER effects — sox tempo/speed/fade change length). ──
    try:
        duration = cli.wav_duration(out_path)
    except Exception:
        duration = 0.0

    return SynthResult(
        out=out_path,
        text=processed,
        raw_text=utt,
        duration=duration,
    )


def run_batch(
    utterances: list[str],
    out_paths: list[str],
    *,
    engine,
    engine_name: str,
    eng_cfg: dict,
    config: dict,
    synth_kwargs: dict,
    effect_list: list,
    preprocess_mode=None,
    custom_rules: list | None = None,
    streaming: bool = False,
    on_ready: Callable[[SynthResult], None] | None = None,
    on_interrupt: Callable[[SynthResult], None] | None = None,
) -> tuple[list[SynthResult], BaseException | None]:
    """Run synthesis over a batch.

    When ``streaming=False`` (or ``on_ready`` is not given), runs
    sequentially and returns ``(results, None)`` — exceptions propagate.

    When ``streaming=True`` and ``on_ready`` is given, spawns a producer
    thread (``marmalade-producer``) that renders sequentially, and fires
    ``on_ready(result)`` for each utterance as it lands. The callback runs
    in the caller's thread, so the caller can do blocking work (like
    ``play_wav``) and still overlap with the next render.

    If the producer raises, the consumer finishes draining what's already
    in the queue, then ``run_batch`` returns ``(partial_results, exc)`` —
    it does NOT re-raise. The caller decides whether to re-raise (cli does).

    ``on_interrupt(result)`` is called for each queued-but-unplayed result
    when the consumer receives a ``KeyboardInterrupt`` — lets the caller
    clean up tmp WAVs that won't be touched again. The interrupt then
    propagates out of ``run_batch``.
    """
    if not streaming or on_ready is None:
        # Simple sequential path — no thread.
        results: list[SynthResult] = []
        for utt, out_path in zip(utterances, out_paths):
            r = synthesize_one(
                utt, out_path,
                engine=engine, engine_name=engine_name,
                eng_cfg=eng_cfg, config=config,
                synth_kwargs=synth_kwargs, effect_list=effect_list,
                preprocess_mode=preprocess_mode, custom_rules=custom_rules,
            )
            if r is not None:
                results.append(r)
        return results, None

    # ── Streaming: producer thread + queue, consumer in caller's thread. ──
    import queue
    import threading

    play_q: "queue.Queue" = queue.Queue()
    results: list[SynthResult] = []
    producer_error: list[BaseException] = []

    def _produce():
        try:
            for utt, out_path in zip(utterances, out_paths):
                r = synthesize_one(
                    utt, out_path,
                    engine=engine, engine_name=engine_name,
                    eng_cfg=eng_cfg, config=config,
                    synth_kwargs=synth_kwargs, effect_list=effect_list,
                    preprocess_mode=preprocess_mode, custom_rules=custom_rules,
                )
                if r is None:
                    continue
                results.append(r)
                play_q.put(r)
        except BaseException as e:
            producer_error.append(e)
        finally:
            play_q.put(None)  # sentinel: end of stream

    prod = threading.Thread(
        target=_produce, daemon=True, name="marmalade-producer")
    prod.start()

    # Consumer: drain queue in FIFO (= input) order, hand to on_ready.
    try:
        while True:
            r = play_q.get()
            if r is None:
                break
            on_ready(r)
    except KeyboardInterrupt:
        # Producer is a daemon thread, so it dies with the process; we
        # just notify the caller about whatever was already enqueued so
        # tmp files can be cleaned up.
        if on_interrupt is not None:
            try:
                while True:
                    leftover = play_q.get_nowait()
                    if leftover is None:
                        continue
                    on_interrupt(leftover)
            except queue.Empty:
                pass
        raise

    prod.join()
    return results, (producer_error[0] if producer_error else None)
