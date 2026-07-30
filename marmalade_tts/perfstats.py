"""Observed synthesis-performance stats, persisted per engine+model.

Every chunk render yields two free measurements: RTF (render seconds per
second of audio) and chars-per-audio-second. Smoothed with an exponential
moving average, they let the chunk-streaming gate (stream_play.py) decide
when playback can start without ever running a benchmark — the first real
render IS the benchmark, and recent renders dominate the average so the
estimate tracks the device's current state (thermal throttle, load).

Deliberately per engine+model, NOT per voice — voices share a model's
compute profile, and finer granularity just starves the EMA of samples.
"""

from __future__ import annotations

import json
import os
import threading

STATS_PATH = os.path.expanduser("~/.config/marmalade-tts/perf.json")

# Fast-moving average: ~10 chunks to mostly forget an old device state.
ALPHA = 0.3

_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(STATS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    tmp = STATS_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, STATS_PATH)
    except OSError:
        pass  # stats are an optimization; never fail synthesis over them


def _key(engine_name: str, model: str | None) -> str:
    return f"{engine_name}:{model or 'default'}"


def record(engine_name: str, model: str | None,
           chars: int, render_s: float, audio_s: float,
           cond_chars: int = 0, solo: bool = False) -> None:
    """Fold one chunk render into the EMAs. No-op on degenerate inputs.

    ``cond_chars`` is the conditioning text rendered and then discarded
    (context + lookahead), which feeds a second, *marginal* RTF average:
    render seconds per second of audio actually put through the model,
    rather than per second of audio kept.

    ``solo`` marks a render that had the engine to itself. Only those
    update the marginal average, because a render competing with three
    siblings for the same cores takes roughly twice the wall clock — fine
    for the gate (which reasons about the same contended pipeline) but
    wrong for judging how fast the device is. A device's first-ever record
    seeds the marginal average whatever it was, so there is always an
    estimate to work from.
    """
    if chars <= 0 or render_s <= 0 or audio_s <= 0:
        return
    rtf = render_s / audio_s
    cps = chars / audio_s
    # Marginal RTF is size-independent by construction, and that is what
    # makes it safe to CHOOSE chunk size from. Feeding the plain rtf back
    # into sizing is a spiral: smaller chunks amortize the fixed per-chunk
    # costs worse, which raises rtf, which would shrink chunks again
    # (Max's feedback-loop footgun, 2026-07-29).
    mrtf = render_s / (audio_s + (cond_chars / cps if cond_chars else 0.0))
    with _lock:
        data = _load()
        entry = data.get(_key(engine_name, model))
        if not isinstance(entry, dict) or "rtf" not in entry:
            # A band may already be stored from a plan made before the
            # first render completed — keep it.
            entry = dict(entry if isinstance(entry, dict) else {},
                         rtf=rtf, cps=cps, mrtf=mrtf, n=1)
        else:
            entry["rtf"] = ALPHA * rtf + (1 - ALPHA) * entry["rtf"]
            entry["cps"] = ALPHA * cps + (1 - ALPHA) * entry["cps"]
            if solo or "mrtf" not in entry:
                entry["mrtf"] = (ALPHA * mrtf
                                 + (1 - ALPHA) * float(entry.get("mrtf", mrtf)))
            entry["n"] = int(entry.get("n", 0)) + 1
        data[_key(engine_name, model)] = entry
        _save(data)


def estimate(engine_name: str, model: str | None) -> "tuple[float, float] | None":
    """Return (rtf, chars_per_audio_second) or None if never measured."""
    with _lock:
        entry = _load().get(_key(engine_name, model))
    if not isinstance(entry, dict):
        return None
    try:
        return float(entry["rtf"]), float(entry["cps"])
    except (KeyError, TypeError, ValueError):
        return None


def estimate_marginal(engine_name: str, model: str | None) -> "float | None":
    """Marginal RTF (render seconds per second of audio rendered), or None.

    Falls back to the plain RTF for entries written before conditioning
    was priced separately — an overestimate, which errs toward larger
    chunks."""
    with _lock:
        entry = _load().get(_key(engine_name, model))
    if not isinstance(entry, dict):
        return None
    try:
        return float(entry.get("mrtf", entry["rtf"]))
    except (KeyError, TypeError, ValueError):
        return None


def band(engine_name: str, model: str | None) -> "str | None":
    """The chunk-size band last used for this engine+model (hysteresis)."""
    with _lock:
        entry = _load().get(_key(engine_name, model))
    return entry.get("band") if isinstance(entry, dict) else None


def set_band(engine_name: str, model: str | None, name: str) -> None:
    with _lock:
        data = _load()
        entry = data.get(_key(engine_name, model))
        if not isinstance(entry, dict):
            entry = {}
        if entry.get("band") == name:
            return
        entry["band"] = name
        data[_key(engine_name, model)] = entry
        _save(data)


def model_key(engine, eng_cfg: dict) -> str | None:
    """Best-effort model identity for the stats key.

    ``model_size`` (kitten) or ``model`` (config override) when present;
    otherwise None → "default". Voice deliberately not used — for engines
    like piper where the voice selects the model file, stats merge across
    voices, which is acceptable (same order of magnitude) and keeps the
    key rule trivial.
    """
    return (getattr(engine, "model_size", None)
            or eng_cfg.get("model_size") or eng_cfg.get("model"))
