"""Language detection for `--lang auto` — script check + char-trigram NB.

Two stages, run once per utterance:

  1. Script check: kana → ja, Han without kana → zh, Devanagari → hi.
     Each only fires when that script's character count is at least the
     count of latin letters, so a mostly-English line quoting a few kanji
     stays English.
  2. Char-trigram naive Bayes over en/es/fr/it/pt, using the table in
     ``langdetect.tab`` (trained by tools/langdetect-train/train.py).

Returns a language code or None. None means "not confident" — the caller
falls back to its normal language precedence.

The normalization in :func:`_trigram_detect` is in lockstep with the
training script and with the Android port; changing it invalidates the
table.
"""

import base64
import os
import re

TABLE_PATH = os.path.join(os.path.dirname(__file__), "langdetect.tab")

_SPACES = re.compile(r"\s+")

# Stage-2 tuning (validated by tools/langdetect-train/validate.py).
MIN_TRIGRAMS = 6      # below this the guess is noise -> None
MIN_MARGIN = 2.0      # scaled-cost gap per trigram between best and runner-up

# Detected language -> kokoro (misaki) single-letter code. "en" is resolved
# against the voice's own English variant: en-US vs en-GB is never guessed
# from text.
_KOKORO_LANG = {
    "es": "e", "fr": "f", "it": "i", "pt": "p",
    "ja": "j", "zh": "z", "hi": "h",
}

# Every code :func:`detect` can return — the trigram table's five Latin
# languages plus the three the script check decides.
DETECTED_LANGS = frozenset(
    ("en", "es", "fr", "it", "pt", "ja", "zh", "hi"))

# The espeak voice kitten phonemizes with when nothing else is asked for.
# The daemon carries the same constant (it runs in its own venv and can't
# import this package); the two must agree.
KITTEN_ESPEAK_DEFAULT = "en-us"

# Detected language -> the espeak voice the kitten daemon phonemizes with.
# These are `phonemizer`'s EspeakBackend language names, which are IETF-ish
# ("fr-fr", "pt-br") and disagree with raw espeak's file basenames both
# ways — the same list Android's LangDetector.espeakCodeFor returns, so a
# sentence phonemizes identically on both platforms.
#
# Chinese maps to English rather than espeak's "cmn": Mandarin IPA is
# tone-marked, and Kitten's token vocabulary has no tone marks, so the
# tones are dropped and what reaches the model is a mangled syllable
# stream. Reading the Han through English rules is no worse and matches
# Android. `--lang cmn` still reaches espeak verbatim for anyone who
# wants to hear it.
_ESPEAK_LANG = {
    "es": "es", "fr": "fr-fr", "it": "it", "pt": "pt-br",
    "ja": "ja", "hi": "hi", "zh": "en-us",
}


# Han-only text (no kana) needs a zh/ja tiebreak. Each set holds characters
# effectively exclusive to one language's modern writing:
#   _HAN_ZH — PRC simplifications that differ from Japanese shinjitai
#     (这≠這, 时≠時, 读≠読 …), Chinese-only grammar/pronoun characters
#     (们 吗 呢 吧 你 她), and their traditional forms where Japanese uses a
#     different glyph (們 嗎 讓 麼 沒 氣).
#   _HAN_JA — kokuji (characters coined in Japan: 込 働 峠 …) and shinjitai
#     that differ from BOTH the simplified and traditional Chinese forms
#     (図≠图/圖, 円≠圆/圓, 気≠气/氣 …).
# Shared-glyph characters (国 学 会 写 没 万 …) are deliberately absent.
# Mirrored in Android's LangDetector.kt — keep the two in lockstep.
_HAN_ZH = set(
    "这说对时东车书长门问间语读关开见觉认识谁让过还进远边达选们么军动头"
    "买卖妈红电华个为从发汉现乐你她它吗呢吧哪咱啥們嗎讓麼沒氣"
)
_HAN_JA = set("込働峠畑辻枠匂塀笹図円売読絵駅験単桜気帰歯労楽実徳縄渋択沢変遅剣塩拝恵姫黒悪圧")

# A kana-free Han run this long is Chinese: real Japanese sentences carry
# kana within a few characters (okurigana, particles). Below it, an
# unmarked Han run could be either — the system default language breaks
# the tie when it is itself ja or zh (Max, 2026-08-08); any other default
# says nothing about Han text, so the detector abstains.
_HAN_ZH_MIN = 6


def _system_cjk_lang() -> str | None:
    """"ja" or "zh" when the system locale is one of them, else None."""
    loc = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    if not loc:
        import locale
        try:
            loc = locale.getlocale()[0] or ""
        except ValueError:
            loc = ""
    loc = loc.lower()
    if loc.startswith("ja"):
        return "ja"
    if loc.startswith("zh"):
        return "zh"
    return None


def _is_kana(cp: int) -> bool:
    return 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F


def _is_han(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
    )


def _is_devanagari(cp: int) -> bool:
    return 0x0900 <= cp <= 0x097F


class LangDetector:
    """Loaded trigram table plus the two detection stages."""

    def __init__(self, table_path: str = TABLE_PATH,
                 system_lang: str | None = None):
        # The Han-ambiguity tiebreak. `system_lang` is an ISO code override
        # for tests; the default reads the process locale once.
        self._sys_cjk = (system_lang if system_lang in ("ja", "zh")
                         else None if system_lang else _system_cjk_lang())
        with open(table_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lines[0] != "marmalade-langdetect 1":
            raise ValueError(f"unrecognised langdetect table: {table_path}")
        self.langs = lines[1].split()
        self.scale = int(lines[2])
        self.floor = [int(f) for f in lines[3].split()]
        keys = lines[4].replace("_", " ")
        costs = base64.b64decode(lines[5])
        n = len(self.langs)
        self.table = {
            keys[i * 3:i * 3 + 3]: costs[i * n:(i + 1) * n]
            for i in range(len(keys) // 3)
        }

    def detect(self, text: str) -> str | None:
        kana = han = deva = latin = 0
        for ch in text:
            cp = ord(ch)
            if _is_kana(cp):
                kana += 1
            elif _is_han(cp):
                han += 1
            elif _is_devanagari(cp):
                deva += 1
            elif ch.isalpha():
                latin += 1
        cjk = kana + han
        if cjk > 0 and cjk >= latin:
            if kana > 0:
                return "ja"
            return self._han_detect(text, han)
        if deva > 0 and deva >= latin:
            return "hi"
        if latin == 0:
            return None
        return self._trigram_detect(text)

    def _han_detect(self, text: str, han: int) -> str | None:
        """zh/ja for a kana-free Han run — see the marker-set comment."""
        if any(ch in _HAN_ZH for ch in text):
            return "zh"
        if any(ch in _HAN_JA for ch in text):
            return "ja"
        if han >= _HAN_ZH_MIN:
            return "zh"
        return self._sys_cjk

    def _trigram_detect(self, text: str) -> str | None:
        chars = [c if c.isalpha() else " " for c in text.lower()]
        norm = _SPACES.sub(" ", "".join(chars)).strip()
        if not norm:
            return None
        norm = f" {norm} "
        n_tri = len(norm) - 2
        if n_tri < MIN_TRIGRAMS:
            return None
        totals = [0] * len(self.langs)
        for i in range(n_tri):
            row = self.table.get(norm[i:i + 3])
            if row is None:
                for li in range(len(self.langs)):
                    totals[li] += self.floor[li]
            else:
                for li in range(len(self.langs)):
                    totals[li] += row[li]
        order = sorted(range(len(self.langs)), key=lambda li: totals[li])
        best, second = order[0], order[1]
        if (totals[second] - totals[best]) / n_tri < MIN_MARGIN:
            return None
        return self.langs[best]


_detector: LangDetector | None = None


def get_detector() -> LangDetector:
    """The process-wide detector; the table is read once, on first use."""
    global _detector
    if _detector is None:
        _detector = LangDetector()
    return _detector


def detect(text: str) -> str | None:
    """Detect the language of one utterance.

    Returns "en", "es", "fr", "it", "pt", "ja", "zh", "hi" — or None when
    the text is too short or too ambiguous to call.
    """
    return get_detector().detect(text)


def to_kokoro_lang(detected: str, natural: str | None) -> str:
    """Map a detected language to a kokoro (misaki) single-letter code.

    A detected "en" resolves to the voice's own English variant — American
    (``a``) or British (``b``) — never guessed from the text itself, and
    falls back to American when the voice isn't an English one.
    """
    if detected == "en":
        return natural if natural in ("a", "b") else "a"
    return _KOKORO_LANG[detected]


def to_espeak_lang(detected: str) -> str:
    """Map a detected language to the espeak voice kitten phonemizes with.

    English never leaves ``en-us``: Kitten's model is trained on American
    IPA and there is no British Kitten voice to take a region from.
    """
    return _ESPEAK_LANG.get(detected, KITTEN_ESPEAK_DEFAULT)


def _resolve_kokoro(engine, detected: str, synth_kwargs: dict) -> str:
    from .engines import kokoro as kokoro_engine
    voice = synth_kwargs.get("voice") or getattr(engine, "voice", None)
    natural = (kokoro_engine.natural_lang(kokoro_engine.resolve_voice(voice))
               if voice else None)
    return to_kokoro_lang(detected, natural)


def _resolve_kitten(engine, detected: str, synth_kwargs: dict) -> str:
    # No engine reroute: on the CLI the user names the engine, so detection
    # only ever moves the phonemizer. The Kitten voice stays and reads the
    # detected language's IPA — accented and imperfect, and better than
    # reading Spanish through English letter rules (Max, 2026-08-08).
    return to_espeak_lang(detected)


# Engines that can act on `--lang auto`, and how each turns a detected
# language into its own vocabulary. An engine absent here rejects "auto"
# at the CLI/MCP boundary.
_RESOLVERS = {
    "kokoro": _resolve_kokoro,
    "kitten": _resolve_kitten,
}

AUTO_ENGINES = frozenset(_RESOLVERS)


def resolve_auto_lang(engine, engine_name: str, text: str,
                      synth_kwargs: dict) -> dict:
    """Resolve a ``lang="auto"`` request against one utterance's text.

    Called at the utterance boundary (once per utterance, never per
    chunk). Returns ``synth_kwargs`` unchanged when auto wasn't requested;
    otherwise returns a NEW dict — the caller's dict is shared across every
    line of a ``--batch`` run and must not be mutated.

    ``engine_name`` picks the resolver: kokoro speaks misaki letters,
    kitten espeak voice names.

    On an uncertain detection the "lang" key is dropped entirely, so the
    engine's normal precedence applies (kokoro: config lang → voice natural
    language → "a"; kitten: en-us, i.e. exactly today's English path). The
    literal string "auto" never continues downstream.
    """
    requested = synth_kwargs.get("lang") or getattr(engine, "lang", None)
    if requested != "auto":
        return synth_kwargs

    resolved = dict(synth_kwargs)
    resolved.pop("lang", None)

    resolver = _RESOLVERS.get(engine_name)
    if resolver is None:
        return resolved

    detected = detect(text)
    if detected is None:
        return resolved

    resolved["lang"] = resolver(engine, detected, synth_kwargs)
    return resolved
