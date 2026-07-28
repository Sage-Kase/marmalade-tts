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
