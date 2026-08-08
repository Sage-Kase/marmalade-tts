# langdetect-train

Training provenance for `marmalade_tts/langdetect.tab`, the table behind
`--lang auto`.

The detector is a character-trigram naive Bayes classifier over
en/es/fr/it/pt, preceded by a script check that handles ja/zh/hi. The table
is trained on public-domain Project Gutenberg text — `manifest.json` lists
the books, ten per language, all US public domain (Gutenberg
copyright-cleared). The trained table is our own derived work and ships
under the repository's MIT license.

## Regenerating the table

```
python3 fetch_corpus.py    # downloads the corpus listed in manifest.json
python3 train.py           # writes the table
python3 validate.py        # held-out accuracy + script checks
```

`fetch_corpus.py` holds out the last book of each language from training so
`validate.py` measures on unseen text.

## Validation results (current table)

* Full sentences (40–200 characters): **99.95%** accuracy on decided cases
  with a **0.3%** abstain rate; short fragments (20–40 chars) 99.0%.
* Table trained at `TOP_K = 4000` (~8.4k trigram vocabulary, 81 KB);
  `MIN_MARGIN = 2` picked from the sweep (margins 1–8 all measure 100%
  accuracy on 25–200-char held-out sentences; 2 keeps ordinary es/pt
  sentences — the closest pair — from abstaining).
* Script check: ja / zh / hi all pass.

Abstentions are by design — `detect()` returns `None` and the caller falls
back to its normal language precedence.

## Keeping ports in lockstep

`detector_ref.py` is the reference implementation that
`marmalade_tts/langdetect.py` (and the Android port) follow. The text
normalization — lowercase, non-alphabetic to space, collapse runs of space,
one space of padding each side — is part of the table's contract. Changing
it invalidates the table.
