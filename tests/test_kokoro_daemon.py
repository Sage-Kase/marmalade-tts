"""Tests for the kokoro daemon's acoustic cut helpers.

Kokoro's pred_dur alignment is an approximation, not a boundary (the BOS
token claims ~100ms more than the real lead silence — 2026-07-30 probe),
so every cut here is alignment-located but waveform-placed. These tests
drive the pure helpers with synthetic waveforms (plain lists; numpy lives
in the kokoro venv, not the repo's test env).
"""

import importlib.util
import os
import sys
import types

import pytest

_DAEMON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "daemon", "kokoro-daemon.py",
)
_spec = importlib.util.spec_from_file_location("kokoro_daemon", _DAEMON_PATH)
kokoro_daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kokoro_daemon)

FRAME = kokoro_daemon.FRAME
LOUD, QUIET = 0.1, 0.0


def _wav(*segments):
    """Concatenate (level, n_samples) segments into a sample list."""
    out = []
    for level, n in segments:
        out.extend([level] * n)
    return out


# ── acoustic edges ──────────────────────────────────────────────────────────

def test_first_and_last_loud():
    wav = _wav((QUIET, 1000), (LOUD, 500), (QUIET, 300))
    assert kokoro_daemon._first_loud(wav) == 1000
    assert kokoro_daemon._last_loud_end(wav) == 1500


def test_all_silence_edges():
    wav = _wav((QUIET, 1000))
    assert kokoro_daemon._first_loud(wav) == 1000
    assert kokoro_daemon._last_loud_end(wav) == 0


def test_quiet_span_finds_longest_run():
    wav = _wav((LOUD, 100), (QUIET, 50), (LOUD, 100), (QUIET, 200),
               (LOUD, 100))
    s, e = kokoro_daemon._quiet_span(wav, 0, len(wav))
    assert (s, e) == (250, 450)


# ── conditioning cuts ───────────────────────────────────────────────────────

def test_context_cut_snaps_to_gap_end_with_backoff():
    # speech · 3-frame gap · speech; alignment says the text starts one
    # frame INTO the kept speech (the measured late-boundary failure mode).
    # Two frames of backoff: soft fricative onsets sit under the silence
    # threshold, so the gap's found end can already be inside the word.
    gap_s, gap_e = 5 * FRAME, 8 * FRAME
    wav = _wav((LOUD, gap_s), (QUIET, gap_e - gap_s), (LOUD, 5 * FRAME))
    cut = kokoro_daemon._context_cut(wav, gap_e + FRAME)
    assert cut == gap_e - 2 * FRAME


def test_context_cut_without_gap_returns_approx():
    wav = _wav((LOUD, 20 * FRAME))
    assert kokoro_daemon._context_cut(wav, 10 * FRAME) == 10 * FRAME


def test_lookahead_cut_keeps_at_most_tail_keep():
    # A long model-chosen pause before the lookahead must not survive.
    gap_s = 5 * FRAME
    gap_e = gap_s + 10 * FRAME
    wav = _wav((LOUD, gap_s), (QUIET, gap_e - gap_s), (LOUD, 5 * FRAME))
    cut = kokoro_daemon._lookahead_cut(wav, gap_e - FRAME)
    assert cut == gap_s + kokoro_daemon.TAIL_KEEP * FRAME


def test_lookahead_cut_short_gap_keeps_it_whole():
    gap_s, gap_e = 5 * FRAME, 5 * FRAME + FRAME
    wav = _wav((LOUD, gap_s), (QUIET, FRAME), (LOUD, 5 * FRAME))
    assert kokoro_daemon._lookahead_cut(wav, gap_e) == gap_e


def test_lookahead_cut_without_gap_returns_approx():
    wav = _wav((LOUD, 20 * FRAME))
    assert kokoro_daemon._lookahead_cut(wav, 10 * FRAME) == 10 * FRAME


def test_context_cut_ignores_previous_words_gap():
    # Fused join ("ðə mˈɑɹkət" — K1-4b P8): no gap at the boundary, but the
    # PREVIOUS word gap sits inside the search window. Snapping to it would
    # replay the whole context word after the seam; the cut must stay at
    # the alignment boundary instead.
    gap_s, gap_e = 5 * FRAME, 6 * FRAME
    approx = gap_e + 4 * FRAME  # a full word between the gap and the boundary
    wav = _wav((LOUD, gap_s), (QUIET, gap_e - gap_s), (LOUD, 10 * FRAME))
    assert kokoro_daemon._context_cut(wav, approx) == approx


def test_lookahead_cut_ignores_far_gap():
    # Mirror image: a far-off earlier gap must not truncate kept words.
    gap_s, gap_e = 5 * FRAME, 6 * FRAME
    approx = gap_e + 4 * FRAME
    wav = _wav((LOUD, gap_s), (QUIET, gap_e - gap_s), (LOUD, 10 * FRAME))
    assert kokoro_daemon._lookahead_cut(wav, approx) == approx


def _fric(n):
    """Synthetic frication: alternating-sign low-amplitude samples."""
    return [0.01 if i % 2 else -0.01 for i in range(n)]


def test_fricative_backoff_rescues_onset_fricative():
    # speech · gap · fricative butting the cut: the cut walks back across
    # the frication (the P10 "slowly"→"lowly" fix).
    wav = (_wav((LOUD, 5 * FRAME), (QUIET, 2 * FRAME))
           + _fric(2 * FRAME) + _wav((LOUD, 5 * FRAME)))
    cut = 9 * FRAME  # right after the frication
    assert kokoro_daemon._fricative_backoff(wav, cut) == 7 * FRAME


def test_fricative_backoff_stays_put_on_silence():
    wav = _wav((LOUD, 5 * FRAME), (QUIET, 4 * FRAME), (LOUD, 5 * FRAME))
    assert kokoro_daemon._fricative_backoff(wav, 8 * FRAME) == 8 * FRAME


def test_fricative_backoff_stays_put_on_voiced():
    # Voiced audio has low zero-crossing rate — never walked across.
    wav = _wav((LOUD, 10 * FRAME))
    assert kokoro_daemon._fricative_backoff(wav, 8 * FRAME) == 8 * FRAME


def test_fricative_backoff_is_capped():
    wav = _fric(30 * FRAME)
    cut = 20 * FRAME
    assert kokoro_daemon._fricative_backoff(wav, cut) == 12 * FRAME


def test_context_cut_rescues_fricative_before_a_tiny_closure_gap():
    # The measured P10 shape: /s/ frication, THEN a tiny closure gap,
    # then the voiced onset — the snap lands at the gap (beheading the
    # /s/) and the walk-back reclaims it.
    tiny = 300  # 12.5ms — over _MIN_GAP_SAMPLES, under a real word gap
    wav = (_wav((LOUD, 5 * FRAME)) + _fric(2 * FRAME)
           + _wav((QUIET, tiny)) + _wav((LOUD, 5 * FRAME)))
    cut = kokoro_daemon._context_cut(wav, 7 * FRAME + tiny)
    assert cut == 5 * FRAME  # start of the frication, /s/ kept


# ── pause top-ups ───────────────────────────────────────────────────────────

def _bounds(durs):
    """bounds[i] = start sample of phoneme char i, from per-char frames."""
    out, acc = [0], 0
    for d in durs:
        acc += d * FRAME
        out.append(acc)
    return out


def test_pause_insert_tops_up_short_mark():
    # "a: b" — the colon renders 2 frames (50ms) of silence; target 150ms.
    chars = ["ɐ", ":", " ", "b"]
    durs = [4, 2, 1, 4]
    bounds = _bounds(durs)
    wav = _wav((LOUD, 4 * FRAME), (QUIET, 3 * FRAME), (LOUD, 4 * FRAME))
    result = kokoro_daemon._pause_inserts(wav, chars, bounds, {":": 150})
    assert len(result) == 1
    at, n = result[0]
    assert 4 * FRAME < at < 7 * FRAME
    assert n == int(24000 * 0.150) - 3 * FRAME


def test_pause_insert_skips_mark_with_no_rendered_pause():
    chars = ["ɐ", ":", " ", "b"]
    durs = [4, 2, 1, 4]
    wav = _wav((LOUD, 11 * FRAME))  # no silence anywhere — leave it alone
    assert kokoro_daemon._pause_inserts(wav, chars, _bounds(durs),
                                        {":": 150}) == []


def test_pause_insert_skips_trailing_mark():
    chars = ["ɐ", ":"]
    durs = [4, 4]
    wav = _wav((LOUD, 4 * FRAME), (QUIET, 4 * FRAME))
    assert kokoro_daemon._pause_inserts(wav, chars, _bounds(durs),
                                        {":": 150}) == []


def test_pause_insert_already_long_enough():
    chars = ["ɐ", ":", " ", "b"]
    durs = [4, 6, 1, 4]  # 6 frames = 150ms rendered — meets the target
    wav = _wav((LOUD, 4 * FRAME), (QUIET, 6 * FRAME), (LOUD, 5 * FRAME))
    assert kokoro_daemon._pause_inserts(wav, chars, _bounds(durs),
                                        {":": 150}) == []


def test_splice_inserts_silence():
    np = pytest.importorskip("numpy")
    wav = np.full(1000, LOUD, dtype=np.float32)
    out = kokoro_daemon._splice(wav, [(500, 200)], 0, 1000)
    assert len(out) == 1200
    assert not out[500:700].any()
    assert (out[:500] == LOUD).all() and (out[700:] == LOUD).all()


def test_splice_ignores_out_of_range_inserts():
    np = pytest.importorskip("numpy")
    wav = np.full(1000, LOUD, dtype=np.float32)
    out = kokoro_daemon._splice(wav, [(50, 200)], 100, 900)
    assert len(out) == 800


# ── multi-language serving ──────────────────────────────────────────────────
#
# The daemon preloads KOKORO_LANG but serves any lang: voice embedding and
# G2P language are orthogonal in kokoro, so an unloaded lang costs a misaki
# front end, not another copy of the weights. kokoro/numpy/soundfile live in
# the kokoro venv, so everything below is stubbed.

class _FakeAudio:
    def numpy(self):
        return [0.1, 0.2]


class _FakeResult:
    audio = _FakeAudio()
    phonemes = "fəʊ"


class FakeKPipeline:
    """Stub KPipeline. `model=True` mints a fresh model (the startup load);
    anything else is taken as given (False = G2P-only, or a shared KModel)."""

    created = []

    def __init__(self, lang_code, model=True, device=None):
        self.lang_code = lang_code
        self.model = object() if model is True else model
        self.device = device
        self.calls = []
        FakeKPipeline.created.append(self)

    def __call__(self, text, voice=None, speed=1.0):
        self.calls.append((text, voice, speed))
        return [_FakeResult()]


@pytest.fixture
def kokoro_env(monkeypatch):
    """Stub out the kokoro venv's imports and reset the created-pipeline log."""
    FakeKPipeline.created = []

    kokoro = types.ModuleType("kokoro")
    kokoro.KPipeline = FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", kokoro)

    sf = types.ModuleType("soundfile")
    sf.written = []
    sf.write = lambda path, data, rate: sf.written.append((path, data, rate))
    monkeypatch.setitem(sys.modules, "soundfile", sf)

    np = types.ModuleType("numpy")
    np.concatenate = lambda chunks: [x for c in chunks for x in c]
    monkeypatch.setitem(sys.modules, "numpy", np)

    monkeypatch.setattr(kokoro_daemon, "DEFAULT_LANG", "a")
    return sf


def _text_req(tmp_path, lang, name="o.wav"):
    return {"text": "hello", "lang": lang, "out": str(tmp_path / name)}


def test_preloads_default_lang_only(kokoro_env):
    pipelines = kokoro_daemon.load_model()
    # Warm start: the full pipeline plus its G2P twin, nothing else.
    assert [(p.lang_code, p.device) for p in FakeKPipeline.created] == [
        ("a", "cpu"), ("a", None)]
    assert pipelines.model is FakeKPipeline.created[0].model


def test_serves_a_lang_other_than_the_preloaded_one(kokoro_env, tmp_path):
    pipelines = kokoro_daemon.load_model()
    kokoro_daemon.synth(pipelines, _text_req(tmp_path, "b"))
    assert pipelines.full("b").calls == [("hello", "af_heart", 1.0)]
    assert kokoro_env.written[0][0] == str(tmp_path / "o.wav")


def test_pipelines_are_cached_per_lang(kokoro_env, tmp_path):
    pipelines = kokoro_daemon.load_model()
    kokoro_daemon.synth(pipelines, _text_req(tmp_path, "b", "1.wav"))
    made = len(FakeKPipeline.created)
    kokoro_daemon.synth(pipelines, _text_req(tmp_path, "b", "2.wav"))
    assert len(FakeKPipeline.created) == made  # reused, not rebuilt
    assert len(pipelines.full("b").calls) == 2


def test_full_pipelines_share_one_model(kokoro_env):
    pipelines = kokoro_daemon.load_model()
    shared = pipelines.full("a").model
    assert pipelines.full("b").model is shared
    assert pipelines.full("j").model is shared
    assert shared is pipelines.model


def test_missing_lang_falls_back_to_the_preloaded_one(kokoro_env, tmp_path):
    pipelines = kokoro_daemon.load_model()
    kokoro_daemon.synth(pipelines, {"text": "hello",
                                    "out": str(tmp_path / "o.wav")})
    assert len(pipelines.full("a").calls) == 1
    assert len(FakeKPipeline.created) == 2  # no new front end


def test_phonemize_uses_a_model_free_pipeline_for_the_requested_lang(
        kokoro_env, tmp_path):
    pipelines = kokoro_daemon.load_model()
    out = tmp_path / "ph.txt"
    kokoro_daemon.synth(pipelines, {"op": "phonemize", "text": "hello",
                                    "lang": "b", "out": str(out)})
    assert out.read_text(encoding="utf-8") == "fəʊ"
    assert pipelines.g2p("b").model is not pipelines.model  # G2P-only
    assert pipelines.full("b").calls == []  # inference never ran
