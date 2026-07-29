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
           chars: int, render_s: float, audio_s: float) -> None:
    """Fold one chunk render into the EMAs. No-op on degenerate inputs."""
    if chars <= 0 or render_s <= 0 or audio_s <= 0:
        return
    rtf = render_s / audio_s
    cps = chars / audio_s
    with _lock:
        data = _load()
        entry = data.get(_key(engine_name, model))
        if not isinstance(entry, dict) or "rtf" not in entry:
            entry = {"rtf": rtf, "cps": cps, "n": 1}
        else:
            entry["rtf"] = ALPHA * rtf + (1 - ALPHA) * entry["rtf"]
            entry["cps"] = ALPHA * cps + (1 - ALPHA) * entry["cps"]
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
