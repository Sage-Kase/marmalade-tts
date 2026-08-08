"""Kokoro engine unit tests — language resolution.

The precedence contract (kokoro.py module docstring):
    --lang flag > engines.kokoro.lang > voice's natural language > "a".
The natural-language step only works if DEFAULT_CONFIG ships no kokoro
lang — a default there would deep-merge above it for every user (the
regression fixed in 2026-08). Guarded here and in test_config.py.
"""

from marmalade_tts.engines.kokoro import (
    KokoroEngine, natural_lang, resolve_voice,
)


def make_engine(cfg=None):
    return KokoroEngine(cfg or {})


class TestResolveLang:
    def test_cli_flag_wins(self):
        eng = make_engine({"lang": "a"})
        assert eng._resolve_lang("bm_george", "j") == "j"

    def test_config_beats_natural(self):
        eng = make_engine({"lang": "a"})
        assert eng._resolve_lang("bm_george", None) == "a"

    def test_natural_lang_when_unset(self):
        # The headline behavior: no --lang, no config lang → the voice's
        # own language, not American English.
        eng = make_engine({})
        assert eng._resolve_lang("bm_george", None) == "b"
        assert eng._resolve_lang("jm_kumo", None) == "j"
        assert eng._resolve_lang("zf_xiaobei", None) == "z"
        assert eng._resolve_lang("af_heart", None) == "a"

    def test_unknown_voice_falls_back_to_a(self):
        eng = make_engine({})
        assert eng._resolve_lang("xx_mystery", None) == "a"

    def test_auto_never_reaches_the_model(self):
        """"auto" is resolved at the utterance boundary. If a caller ever
        misses that, degrade to the no-lang precedence instead of shipping
        the literal string to the daemon."""
        eng = make_engine({})
        assert eng._resolve_lang("bm_george", "auto") == "b"
        assert eng._resolve_lang("xx_mystery", "auto") == "a"

    def test_auto_in_config_never_reaches_the_model(self):
        eng = make_engine({"lang": "auto"})
        assert eng._resolve_lang("jm_kumo", None) == "j"
        assert eng._resolve_lang("xx_mystery", None) == "a"


class TestVoiceHelpers:
    def test_bare_name_resolves(self):
        assert resolve_voice("george") == "bm_george"

    def test_canonical_passthrough(self):
        assert resolve_voice("bm_george") == "bm_george"

    def test_natural_lang_lookup(self):
        assert natural_lang("bm_george") == "b"
        assert natural_lang("nonsense") is None
