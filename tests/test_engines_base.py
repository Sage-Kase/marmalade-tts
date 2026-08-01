"""Tests for the engines package base — sox_tempo helper, Engine class."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from marmalade_tts.engines import Engine, EngineError, run_in_venv, sox_tempo


# ── Engine base class ────────────────────────────────────────────────────────


class TestEngineBase:
    def test_synthesize_not_implemented(self):
        eng = Engine()
        try:
            eng.synthesize("hi", "/tmp/x.wav")
        except NotImplementedError:
            return
        raise AssertionError("Engine.synthesize should raise NotImplementedError")

    def test_default_name_is_empty(self):
        assert Engine.name == ""


# ── run_in_venv helper ───────────────────────────────────────────────────────


class TestRunInVenv:
    def test_missing_venv_raises_engine_error_with_hint(self):
        # No subprocess is spawned when the venv binary is absent.
        with patch("marmalade_tts.engines.os.path.exists", return_value=False), \
             patch("marmalade_tts.engines.subprocess.run") as mock_run:
            with pytest.raises(EngineError) as exc:
                run_in_venv("/nope/bin/python", ["/nope/bin/python", "-c", "x"],
                            engine_name="pocket")
        mock_run.assert_not_called()
        assert "pocket" in str(exc.value)
        assert "install pocket" in str(exc.value)

    def test_nonzero_exit_raises_with_decoded_stderr(self):
        fake_proc = MagicMock(returncode=1, stderr=b"kaboom")
        with patch("marmalade_tts.engines.os.path.exists", return_value=True), \
             patch("marmalade_tts.engines.subprocess.run", return_value=fake_proc):
            with pytest.raises(EngineError) as exc:
                run_in_venv("/bin/python", ["/bin/python"], engine_name="kokoro")
        assert "kokoro" in str(exc.value)
        assert "kaboom" in str(exc.value)

    def test_success_passes_env_extra_and_stdin(self):
        fake_proc = MagicMock(returncode=0, stderr=b"")
        with patch("marmalade_tts.engines.os.path.exists", return_value=True), \
             patch("marmalade_tts.engines.subprocess.run",
                   return_value=fake_proc) as mock_run:
            run_in_venv("/bin/python", ["/bin/python", "hi"],
                        env_extra={"CUDA_VISIBLE_DEVICES": ""},
                        stdin=b"text", engine_name="piper")
        mock_run.assert_called_once()
        # stdin is forwarded via input=, env_extra lands in the passed env.
        assert mock_run.call_args.kwargs["input"] == b"text"
        assert mock_run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


# ── sox_tempo helper ─────────────────────────────────────────────────────────


class TestSoxTempo:
    def test_speed_1_is_noop(self, tmp_path):
        # No sox call, no file mutation. The path doesn't even need to exist.
        with patch("marmalade_tts.engines.subprocess.run") as mock_run:
            sox_tempo(str(tmp_path / "nope.wav"), 1.0)
        mock_run.assert_not_called()

    def test_falsy_speed_is_noop(self, tmp_path):
        with patch("marmalade_tts.engines.subprocess.run") as mock_run:
            sox_tempo(str(tmp_path / "x.wav"), 0)
            sox_tempo(str(tmp_path / "x.wav"), None)
        mock_run.assert_not_called()

    def test_missing_sox_warns_and_leaves_file(self, tmp_path, capsys):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"riff-fake")
        with patch("marmalade_tts.engines.shutil.which", return_value=None), \
             patch("marmalade_tts.engines.subprocess.run") as mock_run:
            sox_tempo(str(wav), 1.4)
        mock_run.assert_not_called()
        assert wav.read_bytes() == b"riff-fake"
        err = capsys.readouterr().err
        assert "sox" in err.lower()

    def test_passes_speed_factor_to_sox(self, tmp_path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"original")
        fake_proc = MagicMock(returncode=0, stderr=b"")

        def fake_run(cmd, capture_output):
            # sox writes to cmd[2]; simulate by creating the file
            with open(cmd[2], "wb") as f:
                f.write(b"stretched")
            return fake_proc

        with patch("marmalade_tts.engines.shutil.which", return_value="/usr/bin/sox"), \
             patch("marmalade_tts.engines.subprocess.run", side_effect=fake_run) as mock_run:
            sox_tempo(str(wav), 1.4)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "sox"
        assert cmd[1] == str(wav)
        # cmd[2] is a tmp WAV; cmd[3:] is the effect
        assert cmd[3] == "tempo"
        assert cmd[4] == "1.4"
        # File was atomically replaced with the stretched output
        assert wav.read_bytes() == b"stretched"

    def test_sox_failure_warns_and_leaves_original(self, tmp_path, capsys):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"original")
        fake_proc = MagicMock(returncode=1, stderr=b"sox: boom")
        with patch("marmalade_tts.engines.shutil.which", return_value="/usr/bin/sox"), \
             patch("marmalade_tts.engines.subprocess.run", return_value=fake_proc):
            sox_tempo(str(wav), 1.4)
        # Original is untouched and a warning was printed
        assert wav.read_bytes() == b"original"
        assert "sox" in capsys.readouterr().err.lower()


# ── Daemon requests carry model identity ─────────────────────────────────────

class TestDaemonRequestCarriesModel:
    """Daemon-mode requests must tell the daemon which model/lang the config
    resolved to, so a stale daemon refuses instead of silently speaking the
    wrong model (the kitten micro-config/nano-daemon bug)."""

    def _capture(self):
        return patch("marmalade_tts.daemon.synthesize")

    def test_kitten_sends_model_repo(self):
        from marmalade_tts.engines.kitten import KittenEngine
        eng = KittenEngine({"daemon": True, "model_size": "micro"})
        with self._capture() as syn:
            eng.synthesize("hi", "/tmp/o.wav")
        req = syn.call_args.args[1]
        assert req["model"] == "KittenML/kitten-tts-micro-0.8"

    def test_kokoro_sends_resolved_lang(self):
        from marmalade_tts.engines.kokoro import KokoroEngine
        eng = KokoroEngine({"daemon": True})
        with self._capture() as syn:
            eng.synthesize("hi", "/tmp/o.wav", voice="george")  # British voice
        req = syn.call_args.args[1]
        assert req["lang"] == "b"

    def test_piper_sends_model_path(self):
        from marmalade_tts.engines.piper import PiperEngine
        eng = PiperEngine({"daemon": True, "model": "~/v/foo.onnx"})
        with self._capture() as syn:
            eng.synthesize("hi", "/tmp/o.wav")
        req = syn.call_args.args[1]
        assert req["model"] == os.path.expanduser("~/v/foo.onnx")

    def test_coqui_sends_model(self):
        from marmalade_tts.engines.coqui import CoquiEngine
        eng = CoquiEngine({"daemon": True, "model": "tts_models/x/y/z"})
        with self._capture() as syn:
            eng.synthesize("hi", "/tmp/o.wav")
        req = syn.call_args.args[1]
        assert req["model"] == "tts_models/x/y/z"

    def test_kokoro_phonemize_request_shape(self):
        from marmalade_tts.engines.kokoro import KokoroEngine
        eng = KokoroEngine({"daemon": True})
        with self._capture() as syn:
            eng.phonemize("hi there", voice="george")
        req = syn.call_args.args[1]
        assert req["op"] == "phonemize"
        assert req["text"] == "hi there"
        assert req["lang"] == "b"

    def test_kokoro_synthesize_phonemes_request_shape(self):
        from marmalade_tts.engines.kokoro import KokoroEngine
        eng = KokoroEngine({"daemon": True})
        with self._capture() as syn:
            eng.synthesize_phonemes(
                "həlˈO ðˈɛɹ", "/tmp/o.wav", voice="heart",
                context="ktx", lookahead="lˈʌk", style_ref=42,
                pad_marks={":": 150})
        req = syn.call_args.args[1]
        assert req["ph_text"] == "həlˈO ðˈɛɹ"
        assert req["voice"] == "af_heart"
        assert req["lang"] == "a"
        assert req["ph_context"] == "ktx"
        assert req["ph_lookahead"] == "lˈʌk"
        assert req["style_ref"] == 42
        assert req["pad_marks"] == {":": 150}

    def test_kokoro_band_table_gate_and_conditioning_invariants(self):
        # The table's whole point (see the derivation comment in
        # engines/kokoro.py): the fast ramp must open the playback gate
        # right after chunk 0 at kokoro's EFFECTIVE rtf (up to ~0.44 —
        # gate deadline SAFETY*rtf*sum(a[1..k]) <= a[0] + sum(a[1..k-1]))
        # and conditioning must not degrade below 2+2 outside the slow
        # band (R14-1: 2+1 is sanctioned only when streaming can't hold).
        from marmalade_tts import chunking
        from marmalade_tts.engines import kokoro
        fast, slow = kokoro.STREAM_BANDS
        assert fast[3] == fast[4] == chunking.CONTEXT_UNITS == 2
        assert slow[4] == chunking.SLOW_LOOKAHEAD_UNITS
        ramp = fast[2]
        for k in range(1, len(ramp)):
            assert 1.5 * 0.44 * sum(ramp[1:k + 1]) <= \
                ramp[0] + sum(ramp[1:k]), f"gate stalls at ramp step {k}"
        # This desktop (measured mrtf ~0.36) must stay in the fast band.
        assert chunking.band_for_rtf(
            0.36, bands=kokoro.STREAM_BANDS).name == "fast"

    def test_kokoro_phoneme_stream_flag_follows_daemon_mode(self):
        from marmalade_tts.engines.kokoro import KokoroEngine
        assert KokoroEngine({"daemon": True}).PHONEME_STREAM is True
        assert KokoroEngine({}).PHONEME_STREAM is False
        assert KokoroEngine({"daemon": True}).STYLE_ROWS == "ph-utterance"

    def test_matcha_sends_model(self):
        from marmalade_tts.engines.matcha import MatchaEngine
        eng = MatchaEngine({"daemon": True})
        with self._capture() as syn:
            eng.synthesize("hi", "/tmp/o.wav")
        req = syn.call_args.args[1]
        assert req["model"] == "matcha_ljspeech"
