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

    def __init__(self, table_path: str = TABLE_PATH):
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
            return "ja" if kana > 0 else "zh"
        if deva > 0 and deva >= latin:
            return "hi"
        if latin == 0:
            return None
        return self._trigram_detect(text)

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


def resolve_auto_lang(engine, text: str, synth_kwargs: dict) -> dict:
    """Resolve a ``lang="auto"`` request against one utterance's text.

    Called at the utterance boundary (once per utterance, never per
    chunk). Returns ``synth_kwargs`` unchanged when auto wasn't requested;
    otherwise returns a NEW dict — the caller's dict is shared across every
    line of a ``--batch`` run and must not be mutated.

    On an uncertain detection the "lang" key is dropped entirely, so the
    engine's normal precedence (config lang → voice natural language → "a")
    applies. The literal string "auto" never continues downstream.
    """
    requested = synth_kwargs.get("lang") or getattr(engine, "lang", None)
    if requested != "auto":
        return synth_kwargs

    resolved = dict(synth_kwargs)
    resolved.pop("lang", None)

    detected = detect(text)
    if detected is None:
        return resolved

    from .engines import kokoro as kokoro_engine
    voice = synth_kwargs.get("voice") or getattr(engine, "voice", None)
    natural = (kokoro_engine.natural_lang(kokoro_engine.resolve_voice(voice))
               if voice else None)
    resolved["lang"] = to_kokoro_lang(detected, natural)
    return resolved
