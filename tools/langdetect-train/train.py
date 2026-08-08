#!/usr/bin/env python3
"""Train the char-trigram naive-Bayes language table for en/es/fr/it/pt.

Output: langdetect.json — shared by the CLI (Python) and Android (Kotlin)
detectors. Format:

{
  "version": 1,
  "langs": ["en","es","fr","it","pt"],
  "scale": 12,                  # cost = round(-ln P(tri|lang) * scale)
  "floor": [f_en,...],          # cost for unseen trigrams, per language
  "keys": "abc de ...",         # all trigrams concatenated, 3 chars each
  "costs": "<base64 uint8>"     # row-major: trigram-major, lang-minor
}

Held out for validation: the LAST book of each language's corpus is excluded
from training (see holdout split below).
"""
import base64
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
LANGS = ["en", "es", "fr", "it", "pt"]
TOP_K = 2500      # per-language vocabulary contribution
ALPHA = 0.5       # Laplace smoothing
SCALE = 12
HOLDOUT_FRACTION = 0.15  # tail of each language file held out for validation

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase; letters kept, everything else becomes a space.

    Must stay in lockstep with the Kotlin implementation: Char.isLetter()
    gate, then single-space collapse, then one leading/trailing space pad.
    """
    chars = [c if c.isalpha() else " " for c in text.lower()]
    collapsed = _SPACES.sub(" ", "".join(chars)).strip()
    return f" {collapsed} " if collapsed else ""


def trigrams(text: str):
    norm = normalize(text)
    for i in range(len(norm) - 2):
        yield norm[i : i + 3]


def main() -> None:
    counts: dict[str, Counter] = {}
    for lang in LANGS:
        raw = (HERE / "corpus" / f"{lang}.txt").read_text(encoding="utf-8")
        cut = int(len(raw) * (1 - HOLDOUT_FRACTION))
        train_text = raw[:cut]
        (HERE / "corpus" / f"{lang}.holdout.txt").write_text(raw[cut:], encoding="utf-8")
        counts[lang] = Counter(trigrams(train_text))
        print(f"{lang}: {sum(counts[lang].values())} trigram tokens, {len(counts[lang])} types")

    vocab = set()
    for lang in LANGS:
        vocab.update(t for t, _ in counts[lang].most_common(TOP_K))
    keys = sorted(vocab)
    print(f"vocab: {len(keys)} trigrams")

    floors = []
    cost_rows = []
    for lang in LANGS:
        total = sum(counts[lang].values())
        denom = total + ALPHA * (len(keys) + 1)
        floors.append(min(255, round(-math.log(ALPHA / denom) * SCALE)))
        cost_rows.append(
            [min(255, round(-math.log((counts[lang][t] + ALPHA) / denom) * SCALE)) for t in keys]
        )

    costs = bytearray()
    for i in range(len(keys)):
        for li in range(len(LANGS)):
            costs.append(cost_rows[li][i])

    # Line-based format (not JSON: org.json is a stub in Android plain-JVM
    # unit tests). Spaces inside trigram keys are stored as "_" — the
    # normalizer can never produce a literal underscore.
    lines = [
        "marmalade-langdetect 1",
        " ".join(LANGS),
        str(SCALE),
        " ".join(str(f) for f in floors),
        "".join(keys).replace(" ", "_"),
        base64.b64encode(bytes(costs)).decode("ascii"),
    ]
    out = HERE / "langdetect.tab"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
