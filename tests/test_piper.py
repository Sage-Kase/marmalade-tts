"""Tests for the Piper engine — noise_scale / noise_w_scale expressivity knobs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from marmalade_tts.engines import Engine, EngineError


class TestPiperStructure:
    def test_inherits_from_engine(self):
        from marmalade_tts.engines.piper import PiperEngine
        assert issubclass(PiperEngine, Engine)

    def test_has_name_attribute(self):
        from marmalade_tts.engines.piper import PiperEngine
        assert PiperEngine.name == "piper"

    def test_noise_knobs_default_to_none(self):
        # None means "use Piper's own default" — the engine must not pass
        # the flag at all in that case.
        from marmalade_tts.engines.piper import PiperEngine
        eng = PiperEngine({})
        assert eng.noise_scale is None
        assert eng.noise_w_scale is None


class TestPiperLocaleFromStem:
    """Piper voice filenames embed the locale — `--list` and the HTTP
    server's voice listing both parse it out of the stem."""

    @pytest.mark.parametrize("stem,expected", [
        ("en_US-lessac-medium", "en_US"),
        ("en_GB-alba-medium", "en_GB"),
        ("fa_IR-amir-medium", "fa_IR"),
        ("zh_CN-huayan-x_low", "zh_CN"),
        ("ca_ES-upc_ona-medium", "ca_ES"),
    ])
    def test_parses_locale_prefix(self, stem, expected):
        from marmalade_tts.engines.piper import locale_from_stem
        assert locale_from_stem(stem) == expected

    @pytest.mark.parametrize("stem", [
        "my-custom-voice",       # no locale prefix
        "model",                 # no dashes at all
        "EN_US-lessac-medium",   # wrong case for the language subtag
        "",
    ])
    def test_non_conforming_names_yield_none(self, stem):
        from marmalade_tts.engines.piper import locale_from_stem
        assert locale_from_stem(stem) is None

    def test_list_voices_shows_locale(self, tmp_path, capsys):
        from marmalade_tts.engines import piper
        (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"")
        (tmp_path / "custom.onnx").write_bytes(b"")
        with patch.object(piper, "PIPER_VOICES_DIR", str(tmp_path)):
            piper.PiperEngine({}).list_voices()
        out = capsys.readouterr().out
        assert "[en_US]" in out
        # A non-conforming filename is listed unannotated, not dropped.
        assert "custom.onnx" in out
        assert "custom.onnx  [" not in out


def _mock_subprocess():
    # The venv check + run now live in the shared engines.run_in_venv helper,
    # so patch there. (Piper's _find_model still uses piper.os, but these
    # tests always pass an explicit model, so that path isn't hit.)
    fake_proc = MagicMock(returncode=0, stderr=b"")
    exists_patch = patch("marmalade_tts.engines.os.path.exists",
                         return_value=True)
    run_patch = patch("marmalade_tts.engines.subprocess.run",
                      return_value=fake_proc)
    return exists_patch, run_patch


class TestPiperVoiceOverride:
    """`--voice` must route to --model (matches matcha/coqui pattern)."""

    def test_voice_kwarg_passes_through_to_model_flag(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        exists_patch, run_patch = _mock_subprocess()
        with exists_patch, run_patch as mock_run:
            PiperEngine({"model": "/configured.onnx"}).synthesize(
                "Hi", str(tmp_path / "o.wav"),
                voice="/override.onnx")
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--model")
        # The override wins over the configured model
        assert cmd[idx + 1] == "/override.onnx"

    def test_explicit_model_kwarg_also_works(self, tmp_path):
        # Older callers (and tests) may still pass model= directly.
        from marmalade_tts.engines.piper import PiperEngine
        exists_patch, run_patch = _mock_subprocess()
        with exists_patch, run_patch as mock_run:
            PiperEngine({"model": "/configured.onnx"}).synthesize(
                "Hi", str(tmp_path / "o.wav"),
                model="/explicit.onnx")
        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("--model") + 1] == "/explicit.onnx"


class TestPiperMissingVenv:
    def test_missing_venv_raises_engine_error(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        # venv binary absent → EngineError with an install hint, not sys.exit.
        with patch("marmalade_tts.engines.os.path.exists", return_value=False):
            with pytest.raises(EngineError):
                PiperEngine({"model": "/m.onnx"}).synthesize(
                    "Hi", str(tmp_path / "o.wav"))


class TestPiperNoiseKnobsSubprocess:
    def test_noise_scale_propagates(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        exists_patch, run_patch = _mock_subprocess()
        with exists_patch, run_patch as mock_run:
            PiperEngine({"model": "/m.onnx", "noise_scale": 0.85}).synthesize(
                "Hi", str(tmp_path / "o.wav"))
        cmd = mock_run.call_args[0][0]
        assert "--noise-scale" in cmd
        assert cmd[cmd.index("--noise-scale") + 1] == "0.85"

    def test_noise_w_scale_propagates(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        exists_patch, run_patch = _mock_subprocess()
        with exists_patch, run_patch as mock_run:
            PiperEngine({"model": "/m.onnx", "noise_w_scale": 0.5}).synthesize(
                "Hi", str(tmp_path / "o.wav"))
        cmd = mock_run.call_args[0][0]
        assert "--noise-w-scale" in cmd
        assert cmd[cmd.index("--noise-w-scale") + 1] == "0.5"

    def test_unset_knobs_omit_flags(self, tmp_path):
        # No config → no flag → Piper's own defaults apply.
        from marmalade_tts.engines.piper import PiperEngine
        exists_patch, run_patch = _mock_subprocess()
        with exists_patch, run_patch as mock_run:
            PiperEngine({"model": "/m.onnx"}).synthesize(
                "Hi", str(tmp_path / "o.wav"))
        cmd = mock_run.call_args[0][0]
        assert "--noise-scale" not in cmd
        assert "--noise-w-scale" not in cmd


class TestPiperNoiseKnobsDaemon:
    def test_knobs_routed_in_daemon_request(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        with patch("marmalade_tts.engines.piper.dmgr.synthesize") as mock_dmgr:
            PiperEngine({"daemon": True, "noise_scale": 0.9,
                         "noise_w_scale": 0.4}).synthesize(
                "Hi", str(tmp_path / "o.wav"))
        req = mock_dmgr.call_args[0][1]
        assert req["noise_scale"] == 0.9
        assert req["noise_w_scale"] == 0.4

    def test_unset_knobs_omitted_from_request(self, tmp_path):
        from marmalade_tts.engines.piper import PiperEngine
        with patch("marmalade_tts.engines.piper.dmgr.synthesize") as mock_dmgr:
            PiperEngine({"daemon": True}).synthesize(
                "Hi", str(tmp_path / "o.wav"))
        req = mock_dmgr.call_args[0][1]
        assert "noise_scale" not in req
        assert "noise_w_scale" not in req
