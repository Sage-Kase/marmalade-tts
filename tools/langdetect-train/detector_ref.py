#!/usr/bin/env python3
"""Reference detector — will become marmalade_tts/langdetect.py.

Two stages, per utterance:
  1. Script check: kana -> ja, Han without kana -> zh, Devanagari -> hi.
  2. Char-trigram naive Bayes over en/es/fr/it/pt (table from train.py).
Returns a language code or None (caller falls back).
"""
import base64
import json
import re
from pathlib import Path

_SPACES = re.compile(r"\s+")

# stage-2 tuning (validated in validate.py)
MIN_TRIGRAMS = 6      # below this the guess is noise -> None
MIN_MARGIN = 2.0      # scaled-cost gap per trigram between best and runner-up


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
    def __init__(self, table_path: Path):
        lines = Path(table_path).read_text(encoding="utf-8").splitlines()
        assert lines[0] == "marmalade-langdetect 1"
        self.langs = lines[1].split()
        self.scale = int(lines[2])
        self.floor = [int(f) for f in lines[3].split()]
        keys = lines[4].replace("_", " ")
        costs = base64.b64decode(lines[5])
        n = len(self.langs)
        self.table = {
            keys[i * 3 : i * 3 + 3]: costs[i * n : (i + 1) * n]
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
            row = self.table.get(norm[i : i + 3])
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
