"""Regression tests for the kitten daemon's espeak phoneme fixups.

espeak-ng has no dictionary entry for "yeah"; letter-to-sound emits
/jɛh/ with a literal aspirated H. The daemon corrects it to flat /jæ/
at the phonemizer boundary — not the lexically faithful /jɛə/, which
Kitten renders poorly; /jæ/ was ear-picked (2026-07-27 A/B lab). See
fix_en_phonemes in kitten-daemon.py.
"""

import importlib.util
import os

_DAEMON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "daemon", "kitten-daemon.py",
)
_spec = importlib.util.spec_from_file_location("kitten_daemon", _DAEMON_PATH)
kitten_daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kitten_daemon)

fix = kitten_daemon.fix_en_phonemes


def test_standalone_yeah():
    assert fix("jˈɛh") == "jˈæ"


def test_yeah_in_sentence():
    # "I said yeah to that." per espeak 1.51/1.52 en-us
    assert fix("aɪ sˈɛd jˈɛh tə ðˈæt") == "aɪ sˈɛd jˈæ tə ðˈæt"


def test_possessive_keeps_trailing_consonant():
    # "yeah's" — espeak appends /z/ directly, no right-hand boundary
    assert fix("jˈɛhz") == "jˈæz"


def test_stress_variants():
    assert fix("jˌɛh") == "jˌæ"
    assert fix("jɛh") == "jæ"


def test_word_internal_sequence_untouched():
    assert fix("bəjˈɛh") == "bəjˈɛh"


def test_unrelated_words_untouched():
    assert fix("jˈɛs jˈeɪ hɚɹˈɑː") == "jˈɛs jˈeɪ hɚɹˈɑː"


# ── Duration-aware seam surgery (pure helpers) ──────────────────────────────

FRAME = kitten_daemon.FRAME
SPACE = kitten_daemon.SPACE_ID
trim = kitten_daemon._trim_run
cut = kitten_daemon._context_cut


def _wav_for(dur):
    return list(range(sum(d * FRAME for d in dur)))


def test_trim_removes_lead_and_tail_pads():
    # BOS pad 20 frames, content, EOS pad 18 frames.
    dur = [20, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim([0, 30, 31, 0], wav, dur)
    # Head keeps HEAD_KEEP frames of pad, tail keeps TAIL_KEEP.
    start = (20 - kitten_daemon.HEAD_KEEP) * FRAME
    end = len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME
    assert out == wav[start:end]


def test_trailing_punctuation_pause_trimmed_with_pad():
    # ensure_punctuation's comma (id 3) renders as pause frames before
    # the EOS pad — the whole trailing non-speech group is trimmed.
    dur = [20, 5, 5, 12, 18]
    ids = [0, 30, 31, 3, 0]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur)
    end = len(wav) - (12 + 18 - kitten_daemon.TAIL_KEEP) * FRAME
    assert out[-1] == wav[end - 1]


def test_trim_leaves_mismatched_duration_alone():
    dur = [20, 5, 18]
    wav = _wav_for(dur)[:-100]  # break the sum(duration)*FRAME contract
    assert trim([0, 30, 0], wav, dur) == wav


def test_context_cut_lands_on_nearest_space():
    # tokens: BOS, 3 context phonemes, SPACE, 4 text phonemes, EOS
    ids = [0, 30, 31, 32, SPACE, 40, 41, 42, 43, 0]
    dur = [10, 2, 2, 2, 4, 3, 3, 3, 3, 8]
    # context = 3 phonemes → target index 4 → the space at index 4;
    # cut at its END: cumulative samples through index 4.
    expect = sum(dur[:5]) * FRAME
    assert cut(ids, dur, 3) == expect


def test_context_cut_snaps_when_phoneme_count_is_off():
    ids = [0, 30, 31, 32, SPACE, 40, 41, 42, 43, 0]
    dur = [10, 2, 2, 2, 4, 3, 3, 3, 3, 8]
    # Phonemizer counted 5 context phonemes (off by two) — still snaps to
    # the only space token.
    assert cut(ids, dur, 5) == sum(dur[:5]) * FRAME


def test_trim_with_context_starts_at_word_gap():
    ids = [0, 30, 31, SPACE, 40, 41, 0]
    dur = [20, 3, 3, 4, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_context_phonemes=2)
    assert out[0] == wav[sum(dur[:4]) * FRAME]
    assert out[-1] == wav[len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME - 1]
