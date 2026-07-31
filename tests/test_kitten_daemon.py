"""Regression tests for the kitten daemon's espeak phoneme fixups.

espeak-ng has no dictionary entry for "yeah"; letter-to-sound emits
/jɛh/ with a literal aspirated H. The daemon corrects it to flat /jæ/
at the phonemizer boundary — not the lexically faithful /jɛə/, which
Kitten renders poorly; /jæ/ was ear-picked (2026-07-27 A/B lab). See
fix_en_phonemes in kitten-daemon.py.
"""

import importlib.util
import os

import pytest

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


def test_context_cut_snaps_to_word_onset_with_backoff():
    # BOS, 3 ctx phonemes, SPACE, 4 text phonemes, EOS. n_ctx=3 phonemes
    # → text onset at index 5, backed off one frame into the gap so the
    # word keeps its attack.
    ids = [0, 30, 31, 32, SPACE, 40, 41, 42, 43, 0]
    dur = [10, 2, 2, 2, 4, 3, 3, 3, 3, 8]
    assert cut(ids, dur, 3) == sum(dur[:5]) * FRAME - FRAME


def test_context_cut_survives_function_word_fusion():
    # espeak fused the context's last word with the text's first
    # ("from the" → one group): standalone count says 3 phonemes but the
    # fused group starts at count 2. Snapping to the nearest onset keeps
    # the fused group intact rather than landing mid-word.
    ids = [0, 30, 31, SPACE, 40, 41, 42, 43, SPACE, 50, 51, 0]
    dur = [10, 2, 2, 3, 2, 2, 2, 2, 3, 4, 4, 8]
    # onsets: (1, 0), (4, 2), (9, 6). n_ctx=3 → nearest is (4, 2).
    assert cut(ids, dur, 3) == sum(dur[:4]) * FRAME - FRAME


def test_context_cut_tie_snaps_earlier():
    # Equidistant onsets: duplicated sliver of context beats a clipped
    # word, so ties resolve to the earlier onset.
    ids = [0, 30, 31, SPACE, 40, 41, SPACE, 50, 51, 0]
    dur = [10, 2, 2, 3, 2, 2, 3, 4, 4, 8]
    # onsets at speech-counts 0, 2, 4; n_ctx=3 ties (2, 4) → picks 2.
    assert cut(ids, dur, 3) == sum(dur[:4]) * FRAME - FRAME


def test_context_cut_no_speech():
    assert cut([0, SPACE, 0], [10, 3, 8], 3) == 0


def test_trim_with_context_starts_at_word_gap():
    ids = [0, 30, 31, SPACE, 40, 41, 0]
    dur = [20, 3, 3, 4, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_context_phonemes=2)
    assert out[0] == wav[sum(dur[:4]) * FRAME - FRAME]
    assert out[-1] == wav[len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME - 1]


# ── Lookahead (i5): tail-side conditioning cut ──────────────────────────────

lcut = kitten_daemon._lookahead_cut
COMMA = kitten_daemon.VOCAB[","]


def test_lookahead_cut_keeps_punct_pause_drops_space():
    # BOS, word(2), COMMA, SPACE, la word(2), comma, EOS. n_la=2 phonemes
    # → la onset at index 5; walking back off the space keeps the comma's
    # pause on the tail but drops the gap frames (next word's onset bleed).
    ids = [0, 30, 31, COMMA, SPACE, 40, 41, COMMA, 0]
    dur = [20, 3, 3, 6, 4, 3, 3, 12, 18]
    assert lcut(ids, dur, 2) == sum(dur[:4]) * FRAME


def test_lookahead_cut_survives_fusion():
    # Lookahead standalone counts 4 phonemes but fused with the text's
    # last word into one group — the nearest-onset snap must not clip
    # that fused group's text half; it keeps the whole group (tie toward
    # keeping more, cut later).
    ids = [0, 30, 31, SPACE, 40, 41, 42, 43, 44, 45, COMMA, 0]
    dur = [20, 3, 3, 4, 3, 3, 3, 3, 3, 3, 12, 18]
    # onsets: (1,0), (4,2); total=8, target=8-4=4 → |0-4|=4 vs |2-4|=2 →
    # picks (4,2): the fused group is kept out entirely... cut before it.
    assert lcut(ids, dur, 4) == sum(dur[:4]) * FRAME - dur[3] * FRAME


def test_lookahead_cut_whole_run_is_lookahead():
    # Target lands at/before the first onset — nothing of the run is text.
    ids = [0, 40, SPACE, 41, COMMA, 0]
    dur = [20, 3, 4, 3, 12, 18]
    assert lcut(ids, dur, 2) is None


def test_trim_with_context_and_lookahead():
    ids = [0, 30, SPACE, 40, 41, SPACE, 50, COMMA, 0]
    dur = [20, 3, 4, 5, 5, 4, 3, 12, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_context_phonemes=1, n_lookahead_phonemes=1)
    assert out[0] == wav[sum(dur[:3]) * FRAME - FRAME]
    # tail: la onset at index 6, space at 5 walked off → cut at cum[:5]
    assert out[-1] == wav[sum(dur[:5]) * FRAME - 1]


def test_speech_phoneme_count_ignores_punct_and_space():
    assert kitten_daemon._speech_phoneme_count("ɹˈæn ,") == 4
    assert kitten_daemon._speech_phoneme_count("ðə bˈeɪ; ") == 6


# ── Phoneme-direct path helpers ─────────────────────────────────────────────

V = kitten_daemon.VOCAB


def test_ph_tokens_matches_wrapper_pipeline():
    # Punctuation splits into its own space-joined token, exactly like
    # basic_english_tokenize + TextCleaner: "bɪlˈoʊ:" → "bɪlˈoʊ :".
    assert kitten_daemon._ph_tokens("bɪlˈoʊ:") == \
        [V[c] for c in "bɪlˈoʊ :"]
    # Unknown characters are dropped, words joined by single spaces.
    assert kitten_daemon._ph_tokens("ðə  bˈeɪ") == [V[c] for c in "ðə bˈeɪ"]


def test_ph_request_ids_boundaries_exact():
    ids, i_text, i_la = kitten_daemon._ph_request_ids("ðə", "bˈeɪ", "ænd")
    ctx, text, la = ([V[c] for c in s] for s in ("ðə", "bˈeɪ", "ænd"))
    assert ids == [0] + ctx + [SPACE] + text + [SPACE] + la + [10, 0]
    assert ids[i_text:i_text + len(text)] == text
    assert ids[i_la:i_la + len(la)] == la
    assert ids[i_la - 1] == SPACE


def test_ph_request_ids_no_conditioning():
    ids, i_text, i_la = kitten_daemon._ph_request_ids("", "bˈeɪ", "")
    assert ids == [0] + [V[c] for c in "bˈeɪ"] + [10, 0]
    assert i_text == 1 and i_la is None


def test_trim_lookahead_fallback_to_tail_pad():
    # No text of its own → no lookahead cut; normal tail-pad trim applies.
    ids = [0, 30, 31, 0]
    dur = [20, 5, 5, 18]
    wav = _wav_for(dur)
    out = trim(ids, wav, dur, n_lookahead_phonemes=2)
    end = len(wav) - (18 - kitten_daemon.TAIL_KEEP) * FRAME
    assert out[-1] == wav[end - 1]


# ── Punctuation pause top-up (Max: the colon reads as no pause) ─────────────

VOCAB = kitten_daemon.VOCAB
inserts = kitten_daemon._pause_inserts
splice = kitten_daemon._splice
SR = kitten_daemon.SAMPLE_RATE


def _loud(n):
    return [1.0] * n


def _quiet(n):
    return [0.0] * n


def test_pause_targets_maps_marks_to_token_ids():
    assert kitten_daemon._pause_targets({"pad_marks": {":": 150}}) == {
        VOCAB[":"]: 150}
    assert kitten_daemon._pause_targets({}) == {}
    assert kitten_daemon._pause_targets({"pad_marks": {"€": 1}}) == {}


def test_pause_insert_lands_inside_the_rendered_silence():
    # word ':' space word — the mark's frames carry 2 frames of real
    # silence, topped up to 150ms and spliced at the middle of it.
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 1, 1, 1]
    wav = _loud(FRAME) + _quiet(2 * FRAME) + _loud(FRAME)
    (at, n), = inserts(wav, ids, dur, {VOCAB[":"]: 150})
    assert at == 2 * FRAME                       # centre of the silence
    assert n == int(SR * 0.150) - 2 * FRAME


def test_pause_insert_ignores_the_token_boundary_when_it_is_voiced():
    # The alignment puts the boundary inside the next vowel (this is what
    # happened on "the panel: is the" — the first attempt chopped it).
    # The silence one frame earlier is the real pause.
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 1, 1, 1]
    wav = _loud(FRAME) + _quiet(FRAME) + _loud(2 * FRAME)
    (at, _), = inserts(wav, ids, dur, {VOCAB[":"]: 150})
    boundary = 3 * FRAME
    assert at < boundary
    assert abs(float(wav[at])) < 0.006          # spliced into silence


def test_no_insert_when_the_mark_has_no_rendered_pause():
    # Never cut through voiced audio to make a pause that isn't there.
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 1, 1, 1]
    assert inserts(_loud(4 * FRAME), ids, dur, {VOCAB[":"]: 150}) == []


def test_pause_insert_measures_the_silence_it_found():
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 2, 1, 1]
    wav = _loud(FRAME) + _quiet(3 * FRAME) + _loud(FRAME)
    (_, n), = inserts(wav, ids, dur, {VOCAB[":"]: 150})
    assert n == int(SR * 0.150) - 3 * FRAME


def test_no_insert_when_the_pause_is_already_long_enough():
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 8, 1, 1]
    wav = _loud(FRAME) + _quiet(9 * FRAME) + _loud(FRAME)
    assert inserts(wav, ids, dur, {VOCAB[":"]: 150}) == []


def test_trailing_mark_is_left_to_the_tail_trim():
    ids = [VOCAB["a"], VOCAB[":"], 0]
    dur = [1, 1, 1]
    wav = _loud(FRAME) + _quiet(2 * FRAME)
    assert inserts(wav, ids, dur, {VOCAB[":"]: 150}) == []


def test_unmarked_punctuation_is_untouched():
    ids = [VOCAB["a"], VOCAB[";"], VOCAB[" "], VOCAB["b"]]
    dur = [1, 1, 1, 1]
    wav = _loud(FRAME) + _quiet(2 * FRAME) + _loud(FRAME)
    assert inserts(wav, ids, dur, {VOCAB[":"]: 150}) == []


def test_splice_inserts_silence_inside_the_kept_slice():
    # _splice is the one helper needing the array type the daemon uses;
    # numpy lives in the kitten venv, not the repo's test env.
    np = pytest.importorskip("numpy")
    wav = np.arange(100, dtype=np.float32)
    out = splice(wav, [(50, 10)], 20, 80)
    assert len(out) == 70
    assert out[29] == 49 and list(out[30:40]) == [0] * 10 and out[40] == 50


def test_splice_ignores_inserts_outside_the_slice():
    np = pytest.importorskip("numpy")
    wav = np.arange(100, dtype=np.float32)
    out = splice(wav, [(5, 10), (95, 10)], 20, 80)
    assert len(out) == 60


def test_pause_measures_silence_running_past_the_search_window():
    # The mark's frames only overlap part of the pause; measuring just
    # that part would overshoot the target (150ms asked, 190ms delivered
    # on the first try).
    ids = [VOCAB["a"], VOCAB[":"], VOCAB[" "], VOCAB["b"]]
    dur = [2, 1, 1, 1]
    wav = _loud(FRAME) + _quiet(4 * FRAME) + _loud(FRAME)
    (_, n), = inserts(wav, ids, dur, {VOCAB[":"]: 150})
    assert n == int(SR * 0.150) - 4 * FRAME


# ── Lookahead tail cap (the shallow-conditioning seam bug) ─────────────────

cap = kitten_daemon._cap_tail_silence


def test_tail_cap_trims_a_runaway_pause():
    # 650ms of model-rendered silence before the lookahead becomes 75ms.
    wav = _loud(FRAME) + _quiet(26 * FRAME)
    assert cap(wav, 0, len(wav)) == FRAME + 3 * FRAME


def test_tail_cap_leaves_a_short_pause_alone():
    # The seams Max approved end with almost no silence — don't pad them.
    wav = _loud(FRAME) + _quiet(FRAME)
    assert cap(wav, 0, len(wav)) == len(wav)


def test_tail_cap_never_cuts_into_speech():
    wav = _loud(4 * FRAME)
    assert cap(wav, 0, len(wav)) == len(wav)


def test_tail_cap_respects_the_context_cut():
    wav = _quiet(10 * FRAME)
    assert cap(wav, 5 * FRAME, len(wav)) == 5 * FRAME + 3 * FRAME
