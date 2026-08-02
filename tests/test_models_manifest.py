"""Tests for the model manifest's per-model metadata (models.json)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MANIFEST = os.path.join(os.path.dirname(__file__), "..",
                        "marmalade_tts", "models.json")


def _models() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)["models"]


class TestManifestLanguage:
    def test_every_model_declares_a_language(self):
        for model_id, entry in _models().items():
            assert entry.get("language"), f"{model_id} has no language"

    def test_known_languages(self):
        models = _models()
        assert models["piper-en_US-lessac-medium"]["language"] == "en-US"
        assert models["emojivoice-paige"]["language"] == "en"
