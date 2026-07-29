"""Tests for perfstats (EMA store) and stream_play (gate + pipeline)."""

import os
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from marmalade_tts import perfstats
from marmalade_tts.chunking import chunk_for_streaming
from marmalade_tts.stream_play import should_start, try_stream_single


def _silent_wav(path: str, duration_s: float = 0.1, rate: int = 22050):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * duration_s))


# ── perfstats ────────────────────────────────────────────────────────────────


class TestPerfstats:
    def test_no_data_returns_none(self):
        assert perfstats.estimate("kitten", "micro") is None

    def test_first_record_sets_estimate(self):
        perfstats.record("kitten", "micro", chars=100, render_s=2.0,
                         audio_s=10.0)
        rtf, cps = perfstats.estimate("kitten", "micro")
        assert rtf == pytest.approx(0.2)
        assert cps == pytest.approx(10.0)

    def test_ema_moves_toward_new_samples(self):
        perfstats.record("kitten", "micro", 100, 2.0, 10.0)   # rtf 0.2
        perfstats.record("kitten", "micro", 100, 10.0, 10.0)  # rtf 1.0
        rtf, _ = perfstats.estimate("kitten", "micro")
        assert 0.2 < rtf < 1.0

    def test_keys_are_engine_plus_model(self):
        perfstats.record("kitten", "nano", 100, 1.0, 10.0)
        assert perfstats.estimate("kitten", "mini") is None
        assert perfstats.estimate("kokoro", "nano") is None

    def test_degenerate_samples_ignored(self):
        perfstats.record("kitten", "micro", 0, 1.0, 1.0)
        perfstats.record("kitten", "micro", 10, 0.0, 1.0)
        perfstats.record("kitten", "micro", 10, 1.0, 0.0)
        assert perfstats.estimate("kitten", "micro") is None

    def test_corrupt_stats_file_starts_fresh(self):
        os.makedirs(os.path.dirname(perfstats.STATS_PATH), exist_ok=True)
        with open(perfstats.STATS_PATH, "w") as f:
            f.write("{not json")
        perfstats.record("kitten", "micro", 100, 1.0, 10.0)
        assert perfstats.estimate("kitten", "micro") is not None


# ── chunk_for_streaming ──────────────────────────────────────────────────────


class TestChunkForStreaming:
    def test_short_text_single_chunk(self):
        assert chunk_for_streaming("Hello there.", 500) == ["Hello there."]

    def test_first_chunk_is_small(self):
        text = ("A first sentence of reasonable length sits here. " * 1
                + "Following sentences fill out the rest of the text. " * 20)
        chunks = chunk_for_streaming(text.strip(), 500)
        assert len(chunks) >= 2
        assert len(chunks[0]) <= 200
        assert all(len(c) <= 500 for c in chunks)

    def test_tiny_first_sentence_merged_forward(self):
        text = ("Yes! " + "This much longer second sentence would strand a "
                "tiny first chunk while the pipeline renders it alone. " * 8)
        chunks = chunk_for_streaming(text.strip(), 500)
        assert len(chunks[0]) >= 80  # "Yes!" was merged forward

    def test_nothing_lost(self):
        text = ("One sentence here. " * 30).strip()
        chunks = chunk_for_streaming(text, 200)
        assert " ".join(chunks).split() == text.split()


# ── the gate ─────────────────────────────────────────────────────────────────


class TestShouldStart:
    def test_all_rendered_starts(self):
        assert should_start(0.0, [], None, 1)

    def test_no_estimate_waits(self):
        assert not should_start(5.0, [100], None, 1)

    def test_fast_engine_starts_after_first_chunk(self):
        # rtf 0.2, 10 chars/audio-s, five 100-char chunks left: 10s render
        # (1 worker) vs 45s of drain — playback buys the render time, so a
        # sub-realtime engine starts even single-threaded.
        assert should_start(5.0, [100] * 5, (0.2, 10.0), 1)

    def test_slow_engine_waits(self):
        # rtf 1.5, 1 worker: 75s render vs 45s drain.
        assert not should_start(5.0, [100] * 5, (1.5, 10.0), 1)

    def test_slow_engine_eventually_starts(self):
        # Same slow engine, one small chunk left: 3s render vs 5s buffer.
        assert should_start(5.0, [20], (1.5, 10.0), 1)

    def test_workers_divide_render_time(self):
        est = (2.0, 10.0)  # rtf 2: three 100-char chunks → 60s render
        assert not should_start(10.0, [100] * 3, est, 1)  # 90 > 30 drain
        assert should_start(10.0, [100] * 3, est, 8)      # 11.25 ≤ 30

    def test_last_chunk_deadline_is_the_binding_one(self):
        # rtf 0.9, 1 worker, plenty of chunks: render 90s vs drain 90s —
        # fails only because of the safety factor; with a bigger buffer the
        # same engine passes. The drain term is doing the work here.
        est = (0.9, 10.0)
        assert not should_start(10.0, [100] * 10, est, 1)
        assert should_start(50.0, [100] * 10, est, 1)


# ── try_stream_single pipeline ───────────────────────────────────────────────


def _fake_engine(render_delay=0.01, max_chars=120, parallel=False):
    class Fake:
        MAX_CHARS = max_chars
        PARALLEL_CHUNKS = parallel
        model_size = "test"

        def __init__(self):
            self.calls = []
            self._lock = threading.Lock()

        def synthesize(self, text, out_path, **kw):
            time.sleep(render_delay)
            _silent_wav(out_path, duration_s=0.05)
            with self._lock:
                self.calls.append(text)

    return Fake()


_COMMON = dict(eng_cfg={}, config={"defaults": {"preprocessing": False}},
               synth_kwargs={}, preprocess_mode=False, custom_rules=None)


class TestTryStreamSingle:
    def test_short_input_returns_none(self, tmp_path):
        eng = _fake_engine()
        r = try_stream_single(
            "Short.", str(tmp_path / "o.wav"),
            engine=eng, engine_name="fake", play=lambda p: None, **_COMMON)
        assert r is None
        assert eng.calls == []

    def test_streams_all_chunks_in_order_and_concats(self, tmp_path):
        eng = _fake_engine()
        played = []
        text = ("This is a sentence that has some length to it. " * 12).strip()
        out = str(tmp_path / "o.wav")
        r = try_stream_single(
            text, out,
            engine=eng, engine_name="fake", play=played.append, **_COMMON)
        assert r is not None
        assert len(played) >= 3
        # Played in submission order: file names embed the chunk index.
        indices = [p.split("marmalade-stream-")[1][:3] for p in played]
        assert indices == sorted(indices)
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0
        # tmp chunks cleaned up
        assert all(not os.path.exists(p) for p in played)
        assert r.duration > 0

    def test_parallel_engine_overlaps_renders(self, tmp_path):
        eng = _fake_engine(render_delay=0.05, parallel=True)
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        orig = eng.synthesize

        def counted(text, out_path, **kw):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            orig(text, out_path, **kw)
            with lock:
                state["active"] -= 1

        eng.synthesize = counted
        text = ("This is a sentence that has some length to it. " * 12).strip()
        r = try_stream_single(
            text, str(tmp_path / "o.wav"),
            engine=eng, engine_name="fake", play=lambda p: None, **_COMMON)
        assert r is not None
        if (os.cpu_count() or 1) > 1:
            assert state["peak"] >= 2

    def test_render_error_raises_and_cleans_up(self, tmp_path):
        eng = _fake_engine()
        orig = eng.synthesize
        calls = {"n": 0}

        def failing(text, out_path, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("render exploded")
            orig(text, out_path, **kw)

        eng.synthesize = failing
        text = ("This is a sentence that has some length to it. " * 12).strip()
        with pytest.raises(RuntimeError, match="render exploded"):
            try_stream_single(
                text, str(tmp_path / "o.wav"),
                engine=eng, engine_name="fake",
                play=lambda p: None, **_COMMON)

    def test_gate_defers_playback_for_slow_engine(self, tmp_path):
        """With a punishing observed RTF on record, playback must not begin
        until enough chunks are buffered — verified by counting how many
        chunks were already rendered when the first play happened."""
        # Poison the stats: rtf 50 (render 50x slower than realtime).
        perfstats.record("fake", "test", chars=100, render_s=250.0,
                         audio_s=5.0)
        eng = _fake_engine(render_delay=0.01)
        rendered_at_first_play = []

        def play(p):
            if not rendered_at_first_play:
                rendered_at_first_play.append(len(eng.calls))

        text = ("This is a sentence that has some length to it. " * 12).strip()
        r = try_stream_single(
            text, str(tmp_path / "o.wav"),
            engine=eng, engine_name="fake", play=play, **_COMMON)
        assert r is not None
        total = len(eng.calls)
        assert total >= 3
        # rtf 50 with 1 worker: gate can't open until (nearly) everything
        # is rendered.
        assert rendered_at_first_play[0] >= total - 1
