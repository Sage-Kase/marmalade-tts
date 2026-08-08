#!/usr/bin/env python3
"""Fetch public-domain Project Gutenberg text per language for trigram training.

All sources are US public domain (Gutenberg copyright-cleared); the derived
trigram tables are our own work, shippable under MIT.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

LANGS = ["en", "es", "fr", "it", "pt"]
PER_LANG_BOOKS = 10
TARGET_BYTES = 4_000_000  # per language, post-strip

CORPUS = Path(__file__).parent / "corpus"

START_RE = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "marmalade-langdetect-train/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def strip_boilerplate(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def main() -> None:
    manifest = {}
    for lang in LANGS:
        out = CORPUS / f"{lang}.txt"
        got, books = 0, []
        page_url = f"https://gutendex.com/books/?languages={lang}&sort=popular"
        listing = json.loads(fetch(page_url))
        for book in listing["results"]:
            if len(books) >= PER_LANG_BOOKS or got >= TARGET_BYTES:
                break
            # skip multi-language editions (parallel texts would poison counts)
            if book.get("languages") != [lang]:
                continue
            urls = [v for k, v in book["formats"].items() if k.startswith("text/plain")]
            if not urls:
                continue
            try:
                raw = fetch(urls[0])
            except Exception as e:  # noqa: BLE001
                print(f"  skip {book['id']}: {e}", file=sys.stderr)
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            text = strip_boilerplate(text)
            if len(text) < 50_000:
                continue
            books.append({"id": book["id"], "title": book["title"], "bytes": len(text)})
            got += len(text)
            with out.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"{lang}: #{book['id']} {book['title'][:60]} ({len(text)//1000}kB, total {got//1000}kB)")
            time.sleep(1)
        manifest[lang] = books
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    for f in CORPUS.glob("*.txt"):
        f.unlink()
    main()
