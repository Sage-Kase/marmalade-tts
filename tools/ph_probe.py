"""Print the phoneme string the kitten daemon produces for each argument —
exactly what the model sees (espeak via phonemizer + fix_en_phonemes).

    python3 tools/ph_probe.py "biweekly" "Some sentence."
    python3 tools/ph_probe.py --file passage.txt

Handy for auditing pronunciations: read the IPA, spot the ones espeak's
letter-to-sound fallback got wrong.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from marmalade_tts.engines.kitten import KittenEngine  # noqa: E402

args = sys.argv[1:]
if args[:1] == ["--file"]:
    with open(args[1], encoding="utf-8") as f:
        args = [line.rstrip("\n") for line in f if line.strip()]

eng = KittenEngine({})
for w in args:
    print(repr(w), "->", eng.phonemize(w))
