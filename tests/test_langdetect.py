"""Language detection for --lang auto."""

import pytest

from marmalade_tts import langdetect


def test_table_loads():
    det = langdetect.get_detector()
    assert det.langs == ["en", "es", "fr", "it", "pt"]
    assert len(det.floor) == len(det.langs)
    assert det.table


@pytest.mark.parametrize("text,expected", [
    ("The download has finished and your file is ready to open.", "en"),
    ("Su descarga ha terminado y el archivo ya está listo.", "es"),
    ("Votre téléchargement est terminé et le fichier est prêt.", "fr"),
    ("Il download è terminato e il file è pronto per essere aperto.", "it"),
    ("O seu download terminou e o ficheiro já está pronto.", "pt"),
])
def test_detects_modern_sentences(text, expected):
    assert langdetect.detect(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("こんにちは", "ja"),
    ("你好今天天气怎么样", "zh"),
    ("नमस्ते आप कैसे हैं", "hi"),
    # Kana present alongside Han → Japanese, not Mandarin.
    ("東京タワーは高いです", "ja"),
])
def test_script_check(text, expected):
    assert langdetect.detect(text) == expected


@pytest.mark.parametrize("text", ["OK", "123", "", "   ", "hi"])
def test_short_or_ambiguous_abstains(text):
    assert langdetect.detect(text) is None


def test_mostly_latin_with_a_few_kanji_stays_latin():
    """The script check only fires when the script outweighs latin letters."""
    text = "The kanji 東京 appears in this otherwise English sentence."
    assert langdetect.detect(text) == "en"


@pytest.mark.parametrize("detected,natural,expected", [
    ("en", "a", "a"),
    ("en", "b", "b"),        # a British voice keeps its own English variant
    ("en", "f", "a"),        # non-English voice → American as last resort
    ("en", None, "a"),
    ("es", "b", "e"),
    ("fr", "a", "f"),
    ("it", None, "i"),
    ("pt", None, "p"),
    ("ja", "a", "j"),
    ("zh", None, "z"),
    ("hi", None, "h"),
])
def test_to_kokoro_lang(detected, natural, expected):
    assert langdetect.to_kokoro_lang(detected, natural) == expected


class _FakeEngine:
    name = "kokoro"
    lang = None
    voice = "heart"


def test_resolve_auto_lang_passthrough_when_not_auto():
    kwargs = {"lang": "b", "speed": 1.0}
    assert langdetect.resolve_auto_lang(_FakeEngine(), "hello", kwargs) is kwargs


def test_resolve_auto_lang_detects_and_does_not_mutate():
    kwargs = {"lang": "auto", "voice": "heart", "speed": 1.0}
    out = langdetect.resolve_auto_lang(
        _FakeEngine(), "Votre téléchargement est terminé et le fichier est prêt.",
        kwargs)
    assert out["lang"] == "f"
    assert kwargs["lang"] == "auto"      # shared dict untouched


def test_resolve_auto_lang_english_uses_voice_variant():
    kwargs = {"lang": "auto", "voice": "george"}   # bm_george → British
    out = langdetect.resolve_auto_lang(
        _FakeEngine(), "The download has finished and your file is ready.",
        kwargs)
    assert out["lang"] == "b"


def test_resolve_auto_lang_uncertain_drops_lang():
    """Uncertain detection falls back to the engine's own precedence."""
    kwargs = {"lang": "auto", "voice": "heart"}
    out = langdetect.resolve_auto_lang(_FakeEngine(), "OK", kwargs)
    assert "lang" not in out


def test_resolve_auto_lang_from_engine_config():
    """auto set via engines.kokoro.lang resolves the same way."""
    eng = _FakeEngine()
    eng.lang = "auto"
    out = langdetect.resolve_auto_lang(
        eng, "Il download è terminato e il file è pronto per essere aperto.", {})
    assert out["lang"] == "i"


def test_resolve_auto_lang_config_auto_never_leaks():
    eng = _FakeEngine()
    eng.lang = "auto"
    out = langdetect.resolve_auto_lang(eng, "OK", {})
    assert out.get("lang") != "auto"


def _silent_wav(path: str, duration_s: float = 0.1, rate: int = 22050):
    import wave
    frames = int(round(duration_s * rate))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


def test_detection_runs_once_per_utterance_not_per_chunk(tmp_path, monkeypatch):
    """Detection is an utterance-level decision. A chunked render must not
    re-detect (and possibly re-decide) language mid-utterance."""
    from unittest.mock import MagicMock
    from marmalade_tts.synth import synthesize_one

    calls = []

    def counting_detect(text):
        calls.append(text)
        return "fr"

    monkeypatch.setattr(langdetect, "detect", counting_detect)

    engine = MagicMock()
    engine.MAX_CHARS = 30
    engine.lang = None
    engine.synthesize.side_effect = (
        lambda text, out_path, **kw: _silent_wav(out_path, duration_s=0.1)
    )

    text = "First sentence here. Second sentence here. Third sentence here."
    out = str(tmp_path / "combined.wav")

    synthesize_one(
        text, out,
        engine=engine, engine_name="kokoro",
        eng_cfg={}, config={"defaults": {"preprocessing": False}},
        synth_kwargs={"lang": "auto"}, effect_list=[],
        preprocess_mode=False, custom_rules=None,
    )

    assert engine.synthesize.call_count >= 2   # it really did chunk
    assert len(calls) == 1
    assert calls[0] == text
    langs = {c.kwargs.get("lang") for c in engine.synthesize.call_args_list}
    assert langs == {"f"}
