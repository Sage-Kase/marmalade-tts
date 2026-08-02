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
        assert models["piper-de_DE-thorsten-medium"]["language"] == "de-DE"
        assert models["piper-fr_FR-siwis-medium"]["language"] == "fr-FR"
        assert models["piper-es_ES-davefx-medium"]["language"] == "es-ES"
        assert models["piper-nl_NL-pim-medium"]["language"] == "nl-NL"
        assert models["piper-it_IT-paola-medium"]["language"] == "it-IT"
        assert models["emojivoice-paige"]["language"] == "en"

    def test_every_model_declares_a_license(self):
        for model_id, entry in _models().items():
            assert entry.get("license"), f"{model_id} has no license"

    def test_piper_entries_match_installer_recipe(self):
        """Every piper model the installer fetches exists in the manifest,
        and every piper manifest entry is fetched by the installer."""
        from marmalade_tts.installer import INSTALL_RECIPES
        manifest_piper = {m for m, e in _models().items()
                         if e["engine"] == "piper"}
        assert manifest_piper == set(INSTALL_RECIPES["piper"]["models"])

    def test_piper_files_have_upstream_hashes(self):
        """Piper voices download over plain https — a sha256 is required so
        a truncated or rate-limit-page download can never pass as a model."""
        for model_id, entry in _models().items():
            if entry["engine"] != "piper":
                continue
            for f in entry["files"]:
                assert f.get("sha256"), f"{model_id}: {f['dest']} has no sha256"
