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

    def test_ramp_small_first_then_growing(self):
        text = ("A first sentence of reasonable length sits here. "
                + "Following sentences fill out the rest of the text. " * 20)
        chunks = chunk_for_streaming(text.strip(), 500)
        assert len(chunks) >= 3
        assert len(chunks[0]) <= 100   # ramp start (60 target, clause-whole)
        assert all(len(c) <= 500 for c in chunks)
        # Later chunks grow — the last full-size chunk beats the first.
        assert max(len(c) for c in chunks[1:]) > len(chunks[0])

    def test_never_cuts_mid_clause(self):
        text = ("The library closes at nine tonight. If we leave now, we "
                "can still catch the last hour. Bring your notes, because "
                "the study room upstairs is usually quiet. " * 5)
        for c in chunk_for_streaming(text.strip(), 500):
            assert c[-1] in ".!?;:," or c[-1] in "\"'”"

    def test_dialogue_comma_before_quote_is_not_a_boundary(self):
        # Was a boundary until Max's 2026-07-29 lab round: that seam was
        # the one audible break in the G/H variants.
        text = ('Then the lighthouse keeper said, "The ship is coming too '
                'close to the shoreline!" Everyone ran for the rocks below '
                'while the horn kept sounding across the dark water.')
        chunks = chunk_for_streaming(text, 500)
        assert chunks[0] == ('Then the lighthouse keeper said, "The ship is '
                             'coming too close to the shoreline!"')

    def test_semicolons_and_colons_are_boundaries(self):
        text = ("The plan was simple: leave before dawn; travel light; "
                "and tell absolutely nobody where we were headed that day.")
        chunks = chunk_for_streaming(text, 500)
        # Clause units pack up to the 60-char ramp target, never past a
        # clause boundary.
        assert chunks[0] == ("The plan was simple: leave before dawn; "
                             "travel light;")

    def test_plain_commas_are_not_boundaries(self):
        text = ("This sentence, with a short interjection, keeps its commas "
                "inside one chunk. And a second sentence follows it here. "
                "And then a third sentence closes out the whole passage.")
        chunks = chunk_for_streaming(text, 500)
        assert chunks[0] == ("This sentence, with a short interjection, "
                             "keeps its commas inside one chunk.")

    def test_overlong_clause_falls_back_to_word_split(self):
        text = ("word " * 200).strip() + ". Then a normal sentence follows."
        chunks = chunk_for_streaming(text, 300)
        assert all(len(c) <= 300 for c in chunks)

    def test_nothing_lost(self):
        text = ("One sentence here. " * 30).strip()
        chunks = chunk_for_streaming(text, 200)
        assert " ".join(chunks).split() == text.split()


# ── the gate ─────────────────────────────────────────────────────────────────


class TestShouldStart:
    def test_all_rendered_starts(self):
        assert should_start(0.0, [], None)

    def test_no_estimate_waits(self):
        assert not should_start(5.0, [100], None)

    def test_fast_engine_starts_after_first_chunk(self):
        # rtf 0.2, 10 chars/audio-s: chunk k renders in 2s, plays for 10s —
        # every deadline is met with a 5s buffer.
        assert should_start(5.0, [100] * 5, (0.2, 10.0))

    def test_slow_engine_waits(self):
        # rtf 1.5: chunk 1 renders in 15s but is needed at 5s.
        assert not should_start(5.0, [100] * 5, (1.5, 10.0))

    def test_slow_engine_eventually_starts(self):
        # Same slow engine, one small chunk left: 3s render vs 5s buffer.
        assert should_start(5.0, [20], (1.5, 10.0))

    def test_big_next_chunk_binds_not_the_total(self):
        # rtf 0.5: a huge chunk right after a small buffer misses its
        # deadline (25s render, needed at 4s) even though the total render
        # easily fits inside the total audio. The per-chunk check catches
        # what an aggregate check would miss.
        est = (0.5, 10.0)
        assert not should_start(8.0, [500, 100], est)
        # Same engine and buffer with uniform small chunks passes: every
        # deadline stays cheap.
        assert should_start(8.0, [100] * 5, est)

    def test_safety_factor_bites_near_rtf_one(self):
        # rtf 0.9: raw deadlines are met (9s render vs 10s slot) but the
        # 1.5x safety margin correctly refuses; more buffer fixes it.
        est = (0.9, 10.0)
        assert not should_start(10.0, [100] * 10, est)
        assert should_start(50.0, [100] * 10, est)


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
