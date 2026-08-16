"""Tests for the venice daemon — the one engine daemon with no model.

Everything here is offline: ``urllib.request.urlopen`` is replaced, so the
tests cover request construction, key resolution, the WAV write, and every
error path a caller can be handed as ``{"ok": false, "error": ...}``.
"""

import importlib.util
import json
import os
import urllib.error

import pytest

_DAEMON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "daemon", "venice-daemon.py",
)
_spec = importlib.util.spec_from_file_location("venice_daemon", _DAEMON_PATH)
venice_daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(venice_daemon)

WAV = b"RIFF" + b"\x00" * 40


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def api(monkeypatch):
    """Capture outgoing requests; return a handle to inspect/steer them."""
    class Api:
        requests = []
        body = WAV
        raise_exc = None

        def urlopen(self, req, timeout=None):
            self.requests.append((req, timeout))
            if self.raise_exc:
                raise self.raise_exc
            return _Resp(self.body)

        @property
        def payload(self):
            return json.loads(self.requests[-1][0].data)

    handle = Api()
    handle.requests = []
    monkeypatch.setattr(venice_daemon.urllib.request, "urlopen",
                        handle.urlopen)
    monkeypatch.setattr(venice_daemon, "KEY_FILE", "/nonexistent/key")
    monkeypatch.setattr(venice_daemon, "KEY_ENV", "MARMALADE_TEST_VENICE_KEY")
    monkeypatch.setenv("MARMALADE_TEST_VENICE_KEY", "test-key")
    return handle


def _req(tmp_path, **over):
    r = {"text": "Hello there.", "out": str(tmp_path / "o.wav")}
    r.update(over)
    return r


# ── request construction ────────────────────────────────────────────────────

class TestRequest:
    def test_posts_to_venice_with_bearer_key(self, api, tmp_path):
        venice_daemon.synth(None, _req(tmp_path))
        req, timeout = api.requests[0]
        assert req.full_url == venice_daemon.API_URL
        assert req.get_header("Authorization") == "Bearer test-key"
        assert req.get_header("Content-type") == "application/json"
        assert timeout == venice_daemon.TIMEOUT

    def test_payload_defaults(self, api, tmp_path, monkeypatch):
        monkeypatch.setattr(venice_daemon, "DEFAULT_MODEL", "tts-kokoro")
        monkeypatch.setattr(venice_daemon, "DEFAULT_VOICE", "af_heart")
        venice_daemon.synth(None, _req(tmp_path))
        assert api.payload == {
            "model": "tts-kokoro",
            "input": "Hello there.",
            "voice": "af_heart",
            "response_format": "wav",
            "speed": 1.0,
        }

    def test_request_overrides_win(self, api, tmp_path):
        # No check_loaded guard here: a stateless HTTP engine has nothing
        # loaded, so per-request model/voice are simply honored.
        venice_daemon.synth(None, _req(tmp_path, model="tts-orpheus",
                                       voice="bm_george", speed=1.4))
        p = api.payload
        assert (p["model"], p["voice"], p["speed"]) == ("tts-orpheus",
                                                        "bm_george", 1.4)

    def test_text_is_stripped(self, api, tmp_path):
        venice_daemon.synth(None, _req(tmp_path, text="  spaced  "))
        assert api.payload["input"] == "spaced"


# ── output ──────────────────────────────────────────────────────────────────

class TestOutput:
    def test_writes_wav_bytes_verbatim(self, api, tmp_path):
        api.body = WAV + b"payload"
        out = tmp_path / "o.wav"
        venice_daemon.synth(None, _req(tmp_path))
        assert out.read_bytes() == WAV + b"payload"

    def test_non_wav_response_errors_without_writing(self, api, tmp_path):
        api.body = b"ID3\x04junk"
        out = tmp_path / "o.wav"
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path, model="tts-qwen3-tts"))
        assert "non-WAV" in str(exc.value)
        assert "tts-qwen3-tts" in str(exc.value)
        assert not out.exists()


# ── key resolution ──────────────────────────────────────────────────────────

class TestApiKey:
    def test_key_file_wins_over_env(self, api, tmp_path, monkeypatch):
        kf = tmp_path / "key"
        kf.write_text("from-file\n")
        kf.chmod(0o600)
        monkeypatch.setattr(venice_daemon, "KEY_FILE", str(kf))
        assert venice_daemon.api_key() == "from-file"

    def test_env_fallback_when_no_key_file(self, api):
        assert venice_daemon.api_key() == "test-key"

    def test_empty_key_file_falls_back_to_env(self, api, tmp_path,
                                              monkeypatch):
        kf = tmp_path / "key"
        kf.write_text("   \n")
        monkeypatch.setattr(venice_daemon, "KEY_FILE", str(kf))
        assert venice_daemon.api_key() == "test-key"

    def test_no_key_anywhere(self, api, monkeypatch):
        monkeypatch.delenv("MARMALADE_TEST_VENICE_KEY")
        assert venice_daemon.api_key() is None

    def test_loose_permissions_warn_but_still_read(self, api, tmp_path,
                                                   monkeypatch, caplog):
        kf = tmp_path / "key"
        kf.write_text("from-file")
        kf.chmod(0o644)
        monkeypatch.setattr(venice_daemon, "KEY_FILE", str(kf))
        with caplog.at_level("WARNING"):
            assert venice_daemon.api_key() == "from-file"
        assert "chmod 600" in caplog.text

    def test_key_is_reread_per_request(self, api, tmp_path, monkeypatch):
        """A key added after startup works without restarting the daemon."""
        kf = tmp_path / "key"
        monkeypatch.setattr(venice_daemon, "KEY_FILE", str(kf))
        monkeypatch.delenv("MARMALADE_TEST_VENICE_KEY")
        with pytest.raises(RuntimeError):
            venice_daemon.synth(None, _req(tmp_path))
        kf.write_text("late-key")
        venice_daemon.synth(None, _req(tmp_path))
        assert api.requests[-1][0].get_header("Authorization") == \
            "Bearer late-key"


# ── startup with no key: warn, never exit ───────────────────────────────────

class TestLoadModel:
    def test_missing_key_does_not_raise(self, api, monkeypatch, capsys):
        monkeypatch.delenv("MARMALADE_TEST_VENICE_KEY")
        assert venice_daemon.load_model() is None
        assert "no API key configured" in capsys.readouterr().out

    def test_key_present_reports_ok(self, api, capsys):
        venice_daemon.load_model()
        assert "key OK" in capsys.readouterr().out

    def test_request_without_key_names_both_sources(self, api, monkeypatch,
                                                    tmp_path):
        monkeypatch.delenv("MARMALADE_TEST_VENICE_KEY")
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        msg = str(exc.value)
        assert msg.startswith("venice: no API key configured")
        assert "/nonexistent/key" in msg
        assert "MARMALADE_TEST_VENICE_KEY" in msg


# ── input validation ────────────────────────────────────────────────────────

class TestInputLimits:
    def test_over_the_char_limit_is_rejected_locally(self, api, tmp_path):
        text = "a" * (venice_daemon.MAX_INPUT_CHARS + 1)
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path, text=text))
        assert "4096" in str(exc.value)
        assert api.requests == []  # never left the machine

    def test_exactly_at_the_limit_is_sent(self, api, tmp_path):
        text = "a" * venice_daemon.MAX_INPUT_CHARS
        venice_daemon.synth(None, _req(tmp_path, text=text))
        assert len(api.payload["input"]) == venice_daemon.MAX_INPUT_CHARS

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_is_rejected(self, api, tmp_path, text):
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path, text=text))
        assert "no text" in str(exc.value)


# ── HTTP / network errors ───────────────────────────────────────────────────

def _http_error(code, body):
    import io
    return urllib.error.HTTPError(
        venice_daemon.API_URL, code, "err", {}, io.BytesIO(body))


class TestErrors:
    @pytest.mark.parametrize("code", [401, 402, 429, 500])
    def test_http_error_reports_status_and_body(self, api, tmp_path, code):
        api.raise_exc = _http_error(
            code, json.dumps({"error": "Nope, not paid up"}).encode())
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert f"HTTP {code}" in str(exc.value)
        assert "Nope, not paid up" in str(exc.value)

    def test_non_json_error_body_passes_through(self, api, tmp_path):
        api.raise_exc = _http_error(502, b"<html>bad gateway</html>")
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert "bad gateway" in str(exc.value)

    def test_error_body_is_truncated(self, api, tmp_path):
        api.raise_exc = _http_error(500, b"x" * 5000)
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert len(str(exc.value)) < 500

    def test_unreachable_host(self, api, tmp_path):
        api.raise_exc = urllib.error.URLError("no route to host")
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert "could not reach" in str(exc.value)

    def test_read_phase_timeout(self, api, tmp_path):
        api.raise_exc = TimeoutError("timed out")
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert "failed" in str(exc.value)

    def test_every_error_is_prefixed_for_the_client(self, api, tmp_path):
        # _common.serve turns the exception text into {"ok": false, "error"},
        # so the engine name has to be in the message itself.
        api.raise_exc = urllib.error.URLError("down")
        with pytest.raises(RuntimeError) as exc:
            venice_daemon.synth(None, _req(tmp_path))
        assert str(exc.value).startswith("venice: ")


# ── registration in the daemon manager ──────────────────────────────────────

class TestRegistration:
    def test_paths_registered(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import marmalade_tts.daemon as daemon_mod
        sock, pid, svc, script = daemon_mod._paths("venice")
        assert sock.endswith("venice.sock")
        assert pid.endswith("venice.pid")
        assert svc == "marmalade-venice.service"
        assert script.endswith("venice-daemon.py")

    def test_service_file_shipped(self):
        svc = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "systemd", "marmalade-venice.service")
        assert os.path.exists(svc)

    def test_daemon_env_from_config(self):
        import marmalade_tts.daemon as daemon_mod
        from unittest.mock import patch
        cfg = {"engines": {"venice": {"model": "tts-orpheus",
                                      "voice": "bm_george",
                                      "timeout": 45}}}
        with patch("marmalade_tts.config.load", return_value=cfg):
            env = daemon_mod._daemon_env("venice")
        assert env["VENICE_MODEL"] == "tts-orpheus"
        assert env["VENICE_VOICE"] == "bm_george"
        assert env["VENICE_TIMEOUT"] == "45"
        assert env["VENICE_API_KEY_ENV"] == "VENICE_API_KEY"
        assert env["VENICE_API_KEY_FILE"].endswith(
            "/.config/marmalade-tts/venice-api-key")
        assert "~" not in env["VENICE_API_KEY_FILE"]

    def test_config_defaults(self):
        from marmalade_tts.config import DEFAULT_CONFIG
        venice = DEFAULT_CONFIG["engines"]["venice"]
        assert venice["model"] == "tts-kokoro"
        assert venice["api_key_env"] == "VENICE_API_KEY"
        assert venice["api_key_file"].endswith("venice-api-key")

    def test_no_venv_python_needed(self):
        import marmalade_tts.daemon as daemon_mod
        # stdlib-only daemon: _find_python falls through to system python3.
        assert "venice" not in daemon_mod.ENGINE_PYTHON
