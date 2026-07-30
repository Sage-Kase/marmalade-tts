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


def test_context_cut_lands_on_nth_space():
    # tokens: BOS, word, SPACE, word, SPACE, 4 text phonemes, EOS —
    # a 2-word context ends at the 2nd space; cut at its END.
    ids = [0, 30, SPACE, 31, 32, SPACE, 40, 41, 42, 43, 0]
    dur = [10, 2, 3, 2, 2, 4, 3, 3, 3, 3, 8]
    assert cut(ids, dur, 2) == sum(dur[:6]) * FRAME


def test_context_cut_short_words_stay_intact():
    # 1-word context followed by a 1-phoneme word ("the"): the cut must
    # land after the FIRST space, not the gap after "the" — the failure
    # mode of the old phoneme-count-and-snap cut.
    ids = [0, 30, 31, 32, SPACE, 40, SPACE, 41, 42, 0]
    dur = [10, 2, 2, 2, 4, 3, 4, 3, 3, 8]
    assert cut(ids, dur, 1) == sum(dur[:5]) * FRAME


def test_context_cut_word_count_exceeds_spaces():
    ids = [0, 30, 31, 0]
    dur = [10, 2, 2, 8]
    assert cut(ids, dur, 3) == 0


def test_trim_with_context_starts_at_word_gap():
    ids = [0, 30, 31, SPACE, 40, 41, 0]
    dur = [20, 3, 3, 4, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_context_words=1)
    assert out[0] == wav[sum(dur[:4]) * FRAME]
    assert out[-1] == wav[len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME - 1]


# ── Lookahead (i5): tail-side conditioning cut ──────────────────────────────

lcut = kitten_daemon._lookahead_cut
COMMA = kitten_daemon.VOCAB[","]


def test_lookahead_cut_keeps_gap_on_tail():
    # tokens: BOS, text word, SPACE, text word, SPACE, la word, comma, EOS —
    # 1 lookahead word: cut at the END of the last space, keeping the
    # rendered word gap on the chunk's tail.
    ids = [0, 30, 31, SPACE, 32, 33, SPACE, 40, 41, COMMA, 0]
    dur = [20, 3, 3, 4, 3, 3, 5, 3, 3, 12, 18]
    assert lcut(ids, dur, 1) == sum(dur[:7]) * FRAME


def test_lookahead_cut_two_words():
    ids = [0, 30, 31, SPACE, 40, SPACE, 41, 42, COMMA, 0]
    dur = [20, 3, 3, 4, 3, 5, 3, 3, 12, 18]
    assert lcut(ids, dur, 2) == sum(dur[:4]) * FRAME


def test_lookahead_cut_no_gap_available():
    # Run has fewer word gaps than lookahead words (lookahead-only run
    # after kittentts's sentence split) — no cut; caller drops the run.
    ids = [0, 40, SPACE, 41, COMMA, 0]
    dur = [20, 3, 4, 3, 12, 18]
    assert lcut(ids, dur, 2) is None


def test_trim_with_context_and_lookahead():
    ids = [0, 30, SPACE, 40, 41, SPACE, 50, COMMA, 0]
    dur = [20, 3, 4, 5, 5, 4, 3, 12, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_context_words=1, n_lookahead_words=1)
    assert out[0] == wav[sum(dur[:3]) * FRAME]
    assert out[-1] == wav[sum(dur[:6]) * FRAME - 1]


def test_speech_word_and_onset_counts_agree():
    # "ɹˈæn ," — espeak spaced the comma out as its own 'word': the
    # punctuation-only group must not count on either representation.
    assert kitten_daemon._speech_word_count("ɹˈæn ,") == 1
    assert kitten_daemon._speech_word_count("bɪlˈoʊ: ðə hˈɔːɹn") == 3
    ran = [0, 60, SPACE, COMMA, 0]
    assert len(kitten_daemon._speech_onsets(ran)) == 1
    two = [0, 60, 61, COMMA, SPACE, 62, 0]  # "so, on"
    assert len(kitten_daemon._speech_onsets(two)) == 2


def test_context_cut_ignores_spaced_out_punctuation():
    # espeak may emit "word :" with the punctuation as its own token group;
    # the cut must still land on the next WORD's onset, leaving the
    # punctuation pause on the discarded context side.
    ids = [0, 30, 31, SPACE, COMMA, SPACE, 40, 41, 0]
    dur = [10, 2, 2, 3, 9, 3, 4, 4, 8]
    assert cut(ids, dur, 1) == sum(dur[:6]) * FRAME


def test_trim_lookahead_fallback_to_tail_pad():
    # No word gap for the lookahead cut → normal tail-pad trim applies.
    ids = [0, 30, 31, 0]
    dur = [20, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_lookahead_words=2)
    end = len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME
    assert out[-1] == wav[end - 1]
