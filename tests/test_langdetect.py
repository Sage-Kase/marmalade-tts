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


# Accuracy battery: varied register per language — casual speech, UI-string
# prose, formal prose, questions. Every one must be called, and called right.
ACCURACY_BATTERY = {
    "en": [
        "Could you turn the lights off before you leave the house?",
        "Battery saver is on, so background activity is limited.",
        "The committee will publish its final report next Thursday.",
        "I honestly can't believe how good that movie was.",
        "Remember to water the plants while we're away this weekend.",
        "What time does the next train to the city centre leave?",
    ],
    "es": [
        "¿Puedes apagar las luces antes de salir de casa?",
        "El ahorro de batería está activado y limita la actividad.",
        "El comité publicará su informe final el próximo jueves.",
        "La verdad, no puedo creer lo buena que estuvo la película.",
        "Acuérdate de regar las plantas mientras estamos fuera.",
        "¿A qué hora sale el próximo tren hacia el centro?",
    ],
    "fr": [
        "Peux-tu éteindre les lumières avant de quitter la maison ?",
        "L'économiseur de batterie est activé et limite l'activité.",
        "Le comité publiera son rapport final jeudi prochain.",
        "Franchement, je n'arrive pas à croire à quel point ce film était bon.",
        "Pense à arroser les plantes pendant notre absence ce week-end.",
        "À quelle heure part le prochain train pour le centre-ville ?",
    ],
    "it": [
        "Puoi spegnere le luci prima di uscire di casa?",
        "Il risparmio energetico è attivo e limita le attività in background.",
        "Il comitato pubblicherà la relazione finale giovedì prossimo.",
        "Sinceramente non riesco a credere a quanto fosse bello quel film.",
        "Ricordati di annaffiare le piante mentre siamo via nel fine settimana.",
        "A che ora parte il prossimo treno per il centro città?",
    ],
    "pt": [
        "Você pode apagar as luzes antes de sair de casa?",
        "A economia de bateria está ativada e limita a atividade.",
        "O comitê publicará seu relatório final na próxima quinta-feira.",
        "Sinceramente, não acredito em como aquele filme foi bom.",
        "Lembre-se de regar as plantas enquanto estivermos fora.",
        "A que horas sai o próximo trem para o centro da cidade?",
    ],
    "ja": [
        "明日の朝、駅で待ち合わせしましょう。",
        "バッテリーセーバーがオンになっています。",
        "この本はとても面白かったです。",
    ],
    "zh": [
        "明天早上我们在车站见面吧。",
        "省电模式已开启,后台活动受到限制。",
        "这本书非常有意思,我推荐你看。",
    ],
    "hi": [
        "कल सुबह हम स्टेशन पर मिलेंगे।",
        "बैटरी सेवर चालू है और गतिविधि सीमित है।",
        "यह किताब बहुत दिलचस्प थी।",
    ],
}


@pytest.mark.parametrize(
    "expected,text",
    [(lang, s) for lang, sents in ACCURACY_BATTERY.items() for s in sents],
)
def test_accuracy_battery(expected, text):
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
    assert langdetect.resolve_auto_lang(_FakeEngine(), "kokoro", "hello",
                                        kwargs) is kwargs


def test_resolve_auto_lang_detects_and_does_not_mutate():
    kwargs = {"lang": "auto", "voice": "heart", "speed": 1.0}
    out = langdetect.resolve_auto_lang(
        _FakeEngine(), "kokoro",
        "Votre téléchargement est terminé et le fichier est prêt.", kwargs)
    assert out["lang"] == "f"
    assert kwargs["lang"] == "auto"      # shared dict untouched


def test_resolve_auto_lang_english_uses_voice_variant():
    kwargs = {"lang": "auto", "voice": "george"}   # bm_george → British
    out = langdetect.resolve_auto_lang(
        _FakeEngine(), "kokoro",
        "The download has finished and your file is ready.", kwargs)
    assert out["lang"] == "b"


def test_resolve_auto_lang_uncertain_drops_lang():
    """Uncertain detection falls back to the engine's own precedence."""
    kwargs = {"lang": "auto", "voice": "heart"}
    out = langdetect.resolve_auto_lang(_FakeEngine(), "kokoro", "OK", kwargs)
    assert "lang" not in out


def test_resolve_auto_lang_from_engine_config():
    """auto set via engines.kokoro.lang resolves the same way."""
    eng = _FakeEngine()
    eng.lang = "auto"
    out = langdetect.resolve_auto_lang(
        eng, "kokoro",
        "Il download è terminato e il file è pronto per essere aperto.", {})
    assert out["lang"] == "i"


def test_resolve_auto_lang_config_auto_never_leaks():
    eng = _FakeEngine()
    eng.lang = "auto"
    out = langdetect.resolve_auto_lang(eng, "kokoro", "OK", {})
    assert out.get("lang") != "auto"


# ── kitten: detection moves the phonemizer, never the voice ──────────────────
#
# Max, 2026-08-08: Kitten's accented non-English is better than nothing, and
# on the CLI the user named the engine, so there is no reroute to make. The
# mirror of Android's UtteranceLanguageTest.

class _FakeKitten:
    name = "kitten"
    lang = None
    voice = "Bella"


@pytest.mark.parametrize("detected,expected", [
    ("en", "en-us"),     # never bare "en" — espeak has no such voice
    ("es", "es"),
    ("fr", "fr-fr"),     # phonemizer's IETF-ish names, not file basenames
    ("it", "it"),
    ("pt", "pt-br"),
    ("ja", "ja"),
    ("hi", "hi"),
    ("zh", "en-us"),     # tone marks aren't in Kitten's vocab; see _ESPEAK_LANG
])
def test_to_espeak_lang(detected, expected):
    assert langdetect.to_espeak_lang(detected) == expected


def test_kitten_auto_english_takes_the_english_path():
    """The whole point of the byte-identical requirement: an English
    utterance under `--lang auto` phonemizes with en-us, exactly as it did
    before detection existed."""
    kwargs = {"lang": "auto", "voice": "Bella"}
    out = langdetect.resolve_auto_lang(
        _FakeKitten(), "kitten",
        "The download has finished and your file is ready.", kwargs)
    assert out["lang"] == "en-us"
    assert out["voice"] == "Bella"


def test_kitten_auto_spanish_keeps_the_voice_and_moves_espeak():
    kwargs = {"lang": "auto", "voice": "Bella"}
    out = langdetect.resolve_auto_lang(
        _FakeKitten(), "kitten",
        "La descarga ha terminado y el archivo está listo para abrirse.",
        kwargs)
    assert out["lang"] == "es"
    assert out["voice"] == "Bella"      # no reroute: the voice the user picked
    assert kwargs["lang"] == "auto"     # shared batch dict untouched


def test_kitten_auto_uncertain_falls_back_to_english():
    """Dropping the key leaves the daemon's own en-us default — the English
    path, not a guess."""
    out = langdetect.resolve_auto_lang(
        _FakeKitten(), "kitten", "OK", {"lang": "auto"})
    assert "lang" not in out


def test_kitten_explicit_lang_is_not_touched():
    kwargs = {"lang": "es", "voice": "Bella"}
    assert langdetect.resolve_auto_lang(
        _FakeKitten(), "kitten", "hello", kwargs) is kwargs


def test_auto_engines_is_kokoro_and_kitten():
    assert langdetect.AUTO_ENGINES == {"kokoro", "kitten"}


def test_unknown_engine_drops_auto_rather_than_leaking_it():
    """Belt and braces: the CLI rejects auto on other engines, but if one
    ever arrives here the sentinel must not continue downstream."""
    class _FakePiper:
        name = "piper"
        lang = None
    out = langdetect.resolve_auto_lang(
        _FakePiper(), "piper", "The download has finished.", {"lang": "auto"})
    assert "lang" not in out


@pytest.mark.parametrize("given,expected", [
    ("es", "es"),
    ("fr", "fr-fr"),       # ISO code the detector would emit → espeak's name
    ("pt", "pt-br"),
    ("en", "en-us"),
    ("cmn", "cmn"),        # espeak's own names pass through untranslated
    ("en-gb", "en-gb"),
    (None, None),
])
def test_kitten_espeak_voice_translation(given, expected):
    from marmalade_tts.engines.kitten import espeak_voice
    assert espeak_voice(given) == expected


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
