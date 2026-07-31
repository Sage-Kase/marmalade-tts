"""Tests for marmalade_tts.chunking — text splitting + WAV concatenation."""

import sys
import os
import wave
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from marmalade_tts.chunking import (
    band_for_rtf, chunk_text, concat_wavs, pad_wav_end, ph_pack,
    ph_sentence_runs, ph_stream_plan, resolve_max_chars,
)


# ── chunk_text ───────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_input_returns_unchunked(self):
        assert chunk_text("Hello world.", max_chars=500) == ["Hello world."]

    def test_empty_returns_empty_list(self):
        assert chunk_text("", max_chars=500) == []
        assert chunk_text("   \n  ", max_chars=500) == []

    def test_zero_or_none_max_chars_disables_chunking(self):
        long = "x " * 1000
        assert chunk_text(long, max_chars=0) == [long]
        assert chunk_text(long, max_chars=None) == [long]

    def test_paragraph_splits_first(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        # max_chars small enough to force a split but each paragraph fits
        chunks = chunk_text(text, max_chars=30)
        assert chunks == ["First paragraph here.", "Second paragraph here."]

    def test_sentences_packed_greedily(self):
        # Three sentences, ~25 chars each. With max=60, expect two chunks
        # (two sentences in the first, one in the second).
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, max_chars=60)
        assert len(chunks) == 2
        assert "sentence one" in chunks[0]
        assert "sentence two" in chunks[0]
        assert "sentence three" in chunks[1]

    def test_long_single_sentence_falls_back_to_word_split(self):
        # One sentence with no internal periods; must split on words.
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunks = chunk_text(text, max_chars=25)
        # Every chunk fits the limit
        assert all(len(c) <= 25 for c in chunks)
        # All input words still present, in order
        joined = " ".join(chunks)
        assert joined.split() == text.split()

    def test_no_chunk_loses_whitespace_but_preserves_words(self):
        text = "a very long sentence that needs to be broken up by words"
        chunks = chunk_text(text, max_chars=20)
        # Reconstructable word list
        assert " ".join(chunks).split() == text.split()

    def test_word_longer_than_limit_kept_as_own_chunk(self):
        text = "antidisestablishmentarianism short"
        chunks = chunk_text(text, max_chars=10)
        # The long word becomes a chunk by itself; "short" follows.
        assert "antidisestablishmentarianism" in chunks
        assert "short" in chunks


# ── concat_wavs ──────────────────────────────────────────────────────────────


def _silent_wav(path: str, duration_s: float = 0.1, rate: int = 22050):
    frames = int(round(duration_s * rate))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


class TestConcatWavs:
    def test_single_input_copies_to_output(self, tmp_path):
        a = str(tmp_path / "a.wav")
        out = str(tmp_path / "out.wav")
        _silent_wav(a, 0.5)
        concat_wavs([a], out)
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0

    def test_three_inputs_concatenate(self, tmp_path):
        paths = []
        for i in range(3):
            p = str(tmp_path / f"part-{i}.wav")
            _silent_wav(p, duration_s=0.2)
            paths.append(p)
        out = str(tmp_path / "out.wav")
        concat_wavs(paths, out)

        with wave.open(out, "rb") as w:
            framerate = w.getframerate()
            nframes = w.getnframes()
        # Total duration ≈ 0.6s (3 × 0.2s); allow a frame or two of rounding
        total_s = nframes / framerate
        assert 0.59 <= total_s <= 0.61

    def test_format_mismatch_raises(self, tmp_path):
        a = str(tmp_path / "a.wav")
        b = str(tmp_path / "b.wav")
        _silent_wav(a, duration_s=0.1, rate=22050)
        _silent_wav(b, duration_s=0.1, rate=16000)  # different rate
        with pytest.raises(ValueError, match="format mismatch"):
            concat_wavs([a, b], str(tmp_path / "out.wav"))

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError):
            concat_wavs([], str(tmp_path / "out.wav"))


# ── resolve_max_chars ────────────────────────────────────────────────────────


class _DummyEngine:
    MAX_CHARS = 500


class TestResolveMaxChars:
    def test_config_override_wins(self):
        assert resolve_max_chars(_DummyEngine(), {"max_chars": 200}) == 200

    def test_config_null_disables_chunking(self):
        assert resolve_max_chars(_DummyEngine(), {"max_chars": None}) is None

    def test_config_zero_disables_chunking(self):
        assert resolve_max_chars(_DummyEngine(), {"max_chars": 0}) is None

    def test_engine_default_used_when_no_config(self):
        assert resolve_max_chars(_DummyEngine(), {}) == 500

    def test_no_max_chars_anywhere_returns_none(self):
        class NoLimit:
            pass
        assert resolve_max_chars(NoLimit(), {}) is None

    def test_bool_is_not_a_valid_limit(self):
        class BoolEngine:
            MAX_CHARS = True  # would be 1 if we naively coerced
        assert resolve_max_chars(BoolEngine(), {}) is None


# ── Integration: synthesize_one chunks long input transparently ─────────────


class TestSynthesizeOneChunking:
    """When the input exceeds the engine's MAX_CHARS, synth.synthesize_one
    should split the text, call engine.synthesize for each chunk into a
    temp WAV, and concat the result into the requested out_path. The
    user-facing contract: one input → one WAV."""

    def test_long_input_calls_engine_multiple_times(self, tmp_path):
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        engine = MagicMock()
        engine.MAX_CHARS = 30
        # Each call writes a tiny silent WAV at the requested path.
        engine.synthesize.side_effect = (
            lambda text, out_path, **kw: _silent_wav(out_path, duration_s=0.1)
        )

        # Three sentences that won't fit together in a 30-char chunk.
        text = "First sentence here. Second sentence here. Third sentence here."
        out = str(tmp_path / "combined.wav")

        result = synthesize_one(
            text, out,
            engine=engine, engine_name="kokoro",
            eng_cfg={}, config={"defaults": {"preprocessing": False}},
            synth_kwargs={}, effect_list=[],
            preprocess_mode=False, custom_rules=None,
        )

        assert result is not None
        assert engine.synthesize.call_count >= 2  # at least one split happened
        # Combined output is a real WAV
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0

    def test_short_input_calls_engine_once(self, tmp_path):
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        engine = MagicMock()
        engine.MAX_CHARS = 500
        engine.synthesize.side_effect = (
            lambda text, out_path, **kw: _silent_wav(out_path, duration_s=0.1)
        )

        out = str(tmp_path / "single.wav")
        synthesize_one(
            "Short text.", out,
            engine=engine, engine_name="kokoro",
            eng_cfg={}, config={"defaults": {"preprocessing": False}},
            synth_kwargs={}, effect_list=[],
            preprocess_mode=False, custom_rules=None,
        )

        assert engine.synthesize.call_count == 1
        assert engine.synthesize.call_args[0][0] == "Short text."

    def test_parallel_engine_fans_out_after_first_chunk(self, tmp_path):
        """PARALLEL_CHUNKS engines render chunk 0 alone (daemon warm-up),
        then overlap the rest; concat still sees chunks in input order."""
        import threading
        import time
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        state = {"active": 0, "peak": 0, "first_done_at": None,
                 "rest_started_at": []}
        state_lock = threading.Lock()
        calls: list[tuple[str, str]] = []

        def synth(text, out_path, **kw):
            with state_lock:
                first = not calls
                calls.append((text, out_path))
                if not first:
                    state["rest_started_at"].append(time.monotonic())
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            _silent_wav(out_path, duration_s=0.1)
            with state_lock:
                state["active"] -= 1
                if first:
                    state["first_done_at"] = time.monotonic()

        engine = MagicMock()
        engine.MAX_CHARS = 25
        engine.PARALLEL_CHUNKS = True
        engine.synthesize.side_effect = synth

        text = ("Sentence number one here. Sentence number two here. "
                "Sentence number three here. Sentence number four here.")
        out = str(tmp_path / "parallel.wav")
        result = synthesize_one(
            text, out,
            engine=engine, engine_name="kitten",
            eng_cfg={}, config={"defaults": {"preprocessing": False}},
            synth_kwargs={}, effect_list=[],
            preprocess_mode=False, custom_rules=None,
        )

        assert result is not None
        assert len(calls) >= 3
        # First chunk finished before any of the rest began.
        assert all(t >= state["first_done_at"]
                   for t in state["rest_started_at"])
        if (os.cpu_count() or 1) > 1:
            assert state["peak"] >= 2  # the rest actually overlapped
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0

    def test_sequential_engine_never_overlaps(self, tmp_path):
        """PARALLEL_CHUNKS=False (the Engine default) keeps the old
        one-at-a-time chunk loop."""
        import threading
        import time
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        state = {"active": 0, "peak": 0}
        state_lock = threading.Lock()

        def synth(text, out_path, **kw):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            _silent_wav(out_path, duration_s=0.1)
            with state_lock:
                state["active"] -= 1

        engine = MagicMock()
        engine.MAX_CHARS = 25
        engine.PARALLEL_CHUNKS = False
        engine.synthesize.side_effect = synth

        text = ("Sentence number one here. Sentence number two here. "
                "Sentence number three here. Sentence number four here.")
        out = str(tmp_path / "sequential.wav")
        synthesize_one(
            text, out,
            engine=engine, engine_name="kokoro",
            eng_cfg={}, config={"defaults": {"preprocessing": False}},
            synth_kwargs={}, effect_list=[],
            preprocess_mode=False, custom_rules=None,
        )

        assert state["peak"] == 1

    def test_parallel_chunk_error_propagates_and_cleans_tmp(self, tmp_path):
        """A failing chunk inside the pool surfaces to the caller and no
        marmalade-chunk temp WAVs are left behind."""
        import glob
        import tempfile
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        def synth(text, out_path, **kw):
            if "three" in text:
                raise RuntimeError("chunk exploded")
            _silent_wav(out_path, duration_s=0.1)

        engine = MagicMock()
        engine.MAX_CHARS = 25
        engine.PARALLEL_CHUNKS = True
        engine.synthesize.side_effect = synth

        pattern = os.path.join(tempfile.gettempdir(), "marmalade-chunk-*")
        preexisting = set(glob.glob(pattern))

        text = ("Sentence number one here. Sentence number two here. "
                "Sentence number three here. Sentence number four here.")
        with pytest.raises(RuntimeError, match="chunk exploded"):
            synthesize_one(
                text, str(tmp_path / "boom.wav"),
                engine=engine, engine_name="kitten",
                eng_cfg={}, config={"defaults": {"preprocessing": False}},
                synth_kwargs={}, effect_list=[],
                preprocess_mode=False, custom_rules=None,
            )
        assert set(glob.glob(pattern)) == preexisting

    def test_config_max_chars_overrides_engine_default(self, tmp_path):
        """engines.<name>.max_chars in config wins over the class attribute."""
        from unittest.mock import MagicMock
        from marmalade_tts.synth import synthesize_one

        engine = MagicMock()
        engine.MAX_CHARS = 9999  # engine default is huge
        engine.synthesize.side_effect = (
            lambda text, out_path, **kw: _silent_wav(out_path, duration_s=0.1)
        )

        out = str(tmp_path / "x.wav")
        # Force chunking via config to a very small value
        synthesize_one(
            "First sentence. Second sentence. Third sentence.",
            out,
            engine=engine, engine_name="kokoro",
            eng_cfg={"max_chars": 20},
            config={"defaults": {"preprocessing": False}},
            synth_kwargs={}, effect_list=[],
            preprocess_mode=False, custom_rules=None,
        )

        assert engine.synthesize.call_count >= 2


# ── phoneme-space streaming plan ─────────────────────────────────────────────

# Real espeak output from the kitten daemon (en-us, 2026-07-30).
PH_DIALOGUE = ('"wˌɛɹ dˈɪd juː pˌʊt ðə kˈiːz?" ʃiː ˈæskt. '
               '"ɔnðə hˈʊk," hiː sˈɛd.')
PH_COLON = 'nˈaʊ lˈʊk æt ðə pˈænəl: ɪz ðə ɹˈɛd lˈaɪt stˈɪl blˈɪŋkɪŋ? ɡˈʊd.'


class TestPhSentenceRuns:
    def test_terminal_marks_become_commas(self):
        assert (ph_sentence_runs("wˈʌn. tˈuː? θɹˈiː!")
                == ["wˈʌn,", "tˈuː,", "θɹˈiː,"])

    def test_closing_quote_stays_with_its_sentence(self):
        # The text path split INSIDE the quotation and destroyed the '?';
        # splitting on the mark itself keeps '?" she asked' together.
        runs = ph_sentence_runs(PH_DIALOGUE)
        assert len(runs) == 2
        assert runs[0].startswith('"wˌɛɹ') and 'ˈæskt,' in runs[0]
        assert 'kˈiːz?"' in runs[0]

    def test_swap_happens_inside_a_closing_quote(self):
        assert ph_sentence_runs('hˈɛloʊ."') == ['hˈɛloʊ,"']

    def test_keep_marks_leaves_that_mark_alone(self):
        runs = ph_sentence_runs("wˈʌn. tˈuː! θɹˈiː?", keep_marks="!")
        assert runs == ["wˈʌn,", "tˈuː!", "θɹˈiː,"]

    def test_intra_sentence_marks_untouched(self):
        assert ph_sentence_runs(PH_COLON)[0].count(":") == 1


class TestPhPack:
    def test_short_run_is_one_piece(self):
        assert ph_pack("wˈʌn tˈuː θɹˈiː,", max_chars=500) == ["wˈʌn tˈuː θɹˈiː,"]

    def test_long_run_follows_the_ramp(self):
        run = " ".join(["wˈʌnwˈʌn"] * 60)  # 8-char words, no clause marks
        pieces = ph_pack(run, max_chars=500)
        assert len(pieces) > 3
        assert len(pieces[0]) <= 60 and len(pieces[1]) <= 100
        assert " ".join(pieces).split() == run.split()

    def test_max_chars_caps_the_ramp(self):
        run = " ".join(["wˈʌnwˈʌn"] * 60)
        assert all(len(p) <= 40 for p in ph_pack(run, max_chars=40))

    def test_closes_on_a_clause_mark_near_target(self):
        run = ("wˈʌnwˈʌn wˈʌnwˈʌn wˈʌnwˈʌn wˈʌnwˈʌn wˈʌnwˈʌn, "
               "tˈuːtˈuː tˈuːtˈuː tˈuːtˈuː tˈuːtˈuː tˈuːtˈuː,")
        assert ph_pack(run, max_chars=500)[0].endswith(",")

    def test_clause_mark_before_a_quote_is_not_a_boundary(self):
        # The dialogue comma before a quotation was the one seam Max
        # singled out as audible (2026-07-29).
        run = 'wˈʌnwˈʌn ' * 5 + 'sˈɛd, {}hˈɛloʊ,'
        # Same run, same lengths: a comma before a plain word closes the
        # chunk; the same comma before an opening quote does not.
        assert len(ph_pack(run.format(""), max_chars=500)) == 2
        assert len(ph_pack(run.format('"'), max_chars=500)) == 1


class TestPhStreamPlan:
    def test_one_piece_per_run_when_runs_are_short(self):
        plan = ph_stream_plan(PH_DIALOGUE, max_chars=500)
        assert len(plan) == 2
        assert [p.text for p in plan] == ph_sentence_runs(PH_DIALOGUE)

    def test_gap_after_every_run_but_the_last(self):
        plan = ph_stream_plan(PH_DIALOGUE, max_chars=500)
        assert [p.gap_after_ms for p in plan] == [150, 0]

    def test_no_conditioning_across_a_run_gap(self):
        # J2c: conditioning across sentence gaps gives continuation prosody
        # that contradicts the inserted silence. Max: "really bad".
        for p in ph_stream_plan(PH_DIALOGUE, max_chars=500):
            assert p.context is None and p.lookahead is None

    def test_sub_chunks_are_conditioned_both_ways(self):
        run = " ".join(["wˈʌnwˈʌn"] * 60) + "."
        plan = ph_stream_plan(run, max_chars=500)
        assert len(plan) > 2
        assert plan[0].context is None and plan[0].lookahead is not None
        assert plan[1].context is not None and plan[1].lookahead is not None
        assert plan[-1].lookahead is None
        # context/lookahead are exact substrings of the neighbouring pieces
        assert plan[1].context == " ".join(plan[0].text.split()[-2:])
        assert plan[0].lookahead == " ".join(plan[1].text.split()[:2])

    def test_only_the_run_final_piece_carries_the_gap(self):
        ph = " ".join(["wˈʌnwˈʌn"] * 40) + ". " + " ".join(["tˈuːtˈuː"] * 10)
        plan = ph_stream_plan(ph, max_chars=500)
        gaps = [i for i, p in enumerate(plan) if p.gap_after_ms]
        assert len(gaps) == 1
        assert plan[gaps[0] + 1].context is None  # next run starts fresh

    def test_ramp_continues_across_runs(self):
        # Chunk sizes are about time-to-first-audio, so the ramp counts
        # pieces emitted so far — it does not restart at every sentence.
        ph = ". ".join(" ".join(["wˈʌnwˈʌn"] * 30) for _ in range(3))
        plan = ph_stream_plan(ph, max_chars=500)
        assert len(plan[0].text) <= 60
        assert len(plan[-1].text) > 60

    def test_empty_input_plans_nothing(self):
        assert ph_stream_plan("   ", max_chars=500) == []


class TestPadWavEnd:
    def test_appends_exact_silence(self, tmp_path):
        p = str(tmp_path / "a.wav")
        _silent_wav(p, 1.0, rate=24000)
        pad_wav_end(p, 150)
        with wave.open(p, "rb") as w:
            assert w.getnframes() == 24000 + 3600
            assert w.getframerate() == 24000

    def test_zero_is_a_noop(self, tmp_path):
        p = str(tmp_path / "a.wav")
        _silent_wav(p, 1.0, rate=24000)
        pad_wav_end(p, 0)
        with wave.open(p, "rb") as w:
            assert w.getnframes() == 24000


# ── RTF bands ────────────────────────────────────────────────────────────────


class TestBandForRtf:
    def test_no_measurement_yet_is_optimistic(self):
        assert band_for_rtf(None).name == "fast"

    def test_bands_step_with_measured_rtf(self):
        assert band_for_rtf(0.09).name == "fast"
        assert band_for_rtf(0.25).name == "moderate"
        assert band_for_rtf(0.55).name == "slow"

    def test_every_band_conditions_its_seams(self):
        # Seam conditioning is quality, not a budget line: slow devices
        # get bigger chunks, never worse seams.
        for rtf in (0.09, 0.25, 0.55, 2.0):
            band = band_for_rtf(rtf)
            assert band.context_units == 2 and band.lookahead_units == 2

    def test_first_chunk_shrinks_as_the_device_slows(self):
        assert band_for_rtf(0.25).ramp[0] < band_for_rtf(0.09).ramp[0]

    def test_hysteresis_holds_the_current_band_near_an_edge(self):
        # 0.16 is over the fast band's 0.15 edge but inside the margin.
        assert band_for_rtf(0.16, current="fast").name == "fast"
        assert band_for_rtf(0.20, current="fast").name == "moderate"

    def test_hysteresis_applies_coming_back_down(self):
        assert band_for_rtf(0.14, current="moderate").name == "moderate"
        assert band_for_rtf(0.12, current="moderate").name == "fast"

    def test_unknown_stored_band_is_ignored(self):
        assert band_for_rtf(0.09, current="turbo").name == "fast"


class TestPlanUsesTheBand:
    def test_band_ramp_sizes_the_chunks(self):
        ph = " ".join(["wˈʌnwˈʌn"] * 60) + "."
        fast = ph_stream_plan(ph, 500, band=band_for_rtf(0.09))
        moderate = ph_stream_plan(ph, 500, band=band_for_rtf(0.25))
        assert len(moderate[0].text) < len(fast[0].text)

    def test_slow_band_uses_bigger_chunks(self):
        ph = " ".join(["wˈʌnwˈʌn"] * 60) + "."
        slow = ph_stream_plan(ph, 500, band=band_for_rtf(0.55))
        moderate = ph_stream_plan(ph, 500, band=band_for_rtf(0.25))
        assert len(slow[0].text) > len(moderate[0].text)
        assert all(p.context is not None for p in slow[1:])
