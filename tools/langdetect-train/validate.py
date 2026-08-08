#!/usr/bin/env python3
"""Validate the trained table on held-out Gutenberg text + script-check cases.

Reports accuracy / abstain rate on full sentences and short fragments, and a
margin sweep so MIN_MARGIN can be picked from data.
"""
import random
import re
from pathlib import Path

from detector_ref import LangDetector, MIN_TRIGRAMS

HERE = Path(__file__).parent
LANGS = ["en", "es", "fr", "it", "pt"]
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")

SCRIPT_CASES = [
    ("こんにちは、元気ですか?", "ja"),
    ("東京タワーは高いです", "ja"),          # han + kana -> ja
    ("你好,今天天气怎么样?", "zh"),
    ("我喜欢喝茶。", "zh"),
    ("नमस्ते, आप कैसे हैं?", "hi"),
    ("Play 東京 for me", None),               # latin-dominant -> stage 2 (en or None ok)
]


def sentences(lang: str, min_len: int, max_len: int, n: int, rng: random.Random):
    text = (HERE / "corpus" / f"{lang}.holdout.txt").read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", text)
    sents = [s.strip() for s in SENT_SPLIT.split(text)]
    good = [s for s in sents if min_len <= len(s) <= max_len and sum(c.isalpha() for c in s) > 0.6 * len(s)]
    rng.shuffle(good)
    return good[:n]


def evaluate(det: LangDetector, band: str, min_len: int, max_len: int, n_per: int) -> None:
    rng = random.Random(42)
    correct = wrong = abstain = 0
    wrong_examples = []
    for lang in LANGS:
        for s in sentences(lang, min_len, max_len, n_per, rng):
            got = det.detect(s)
            if got == lang:
                correct += 1
            elif got is None:
                abstain += 1
            else:
                wrong += 1
                if len(wrong_examples) < 8:
                    wrong_examples.append((lang, got, s[:70]))
    total = correct + wrong + abstain
    decided = correct + wrong
    acc = correct / decided * 100 if decided else 0
    print(f"{band}: n={total} decided={decided} acc-on-decided={acc:.2f}% "
          f"abstain={abstain/total*100:.1f}% wrong={wrong}")
    for lang, got, s in wrong_examples:
        print(f"   WRONG {lang}->{got}: {s}")


def margin_sweep(det: LangDetector) -> None:
    rng = random.Random(7)
    data = [(lang, s) for lang in LANGS for s in sentences(lang, 25, 200, 400, rng)]
    for margin in (0, 1, 2, 3, 4, 6, 8):
        correct = wrong = abstain = 0
        for lang, s in data:
            got = _detect_with_margin(det, s, margin)
            if got == lang:
                correct += 1
            elif got is None:
                abstain += 1
            else:
                wrong += 1
        total = len(data)
        decided = correct + wrong
        print(f"margin={margin}: acc-on-decided={correct/decided*100:.2f}% "
              f"abstain={abstain/total*100:.2f}% wrong={wrong}/{total}")


def _detect_with_margin(det: LangDetector, text: str, margin: float):
    import detector_ref as dr
    old = dr.MIN_MARGIN
    dr.MIN_MARGIN = margin
    try:
        return det.detect(text)
    finally:
        dr.MIN_MARGIN = old


def main() -> None:
    det = LangDetector(HERE / "langdetect.tab")
    for text, want in SCRIPT_CASES:
        got = det.detect(text)
        ok = got == want or (want is None and got in (None, "en"))
        print(f"{'OK ' if ok else 'FAIL'} script: {text[:24]!r} -> {got} (want {want})")
    evaluate(det, "full (40-200ch) ", 40, 200, 400)
    evaluate(det, "short (20-40ch) ", 20, 40, 400)
    evaluate(det, "tiny (10-20ch)  ", 10, 20, 400)
    print()
    margin_sweep(det)


if __name__ == "__main__":
    main()
