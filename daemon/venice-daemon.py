#!/usr/bin/env python3
"""marmalade-tts venice daemon — cloud TTS over the marmalade-tts socket.

Every other daemon in here keeps a model in RAM; this one keeps nothing.
It exists so that a machine too weak to hold a local model (or a user who
wants better voices than local kokoro/piper) can still be served through
the same Unix-socket protocol every marmalade-tts consumer already speaks
— point a client's socket path at ``venice.sock`` and nothing else
changes.

Request:  {"text": "...", "out": "/tmp/x.wav", "voice": "af_heart",
           "model": "tts-kokoro", "speed": 1.0}
Response: {"ok": true, "out": "/tmp/x.wav"} | {"ok": false, "error": "..."}

Unlike the local daemons there is no ``check_loaded`` identity guard: a
stateless HTTP engine has nothing loaded, so per-request ``model`` /
``voice`` overrides are simply honored.

**Privacy:** this engine sends the text to Venice's servers. It is opt-in
— nothing changes for the local engines.

Configuration arrives as environment variables (set from ``engines.venice``
in config.yaml by marmalade_tts/daemon.py, or by the systemd unit):
  VENICE_MODEL         default synthesis model      (tts-kokoro)
  VENICE_VOICE         default voice                (af_heart)
  VENICE_API_KEY_FILE  0600 file holding the key    (~/.config/marmalade-tts/venice-api-key)
  VENICE_API_KEY_ENV   env var holding the key      (VENICE_API_KEY)
  VENICE_TIMEOUT       per-request timeout, seconds (30)

A missing key is NOT fatal: the daemon starts, says so in its log, and
fails each request with a clear error. Exiting instead would crash-loop
under systemd's Restart=on-failure.
"""

import json
import logging
import os
import stat
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import serve

API_URL = "https://api.venice.ai/api/v1/audio/speech"

# Venice rejects `input` over 4096 characters. Callers already chunk by
# sentence (marmalade_tts/chunking.py, the voice loop), so the daemon
# rejects an over-long request rather than inventing a split policy of
# its own — a silent re-chunk here would produce seams no client asked
# for and can't hear coming.
MAX_INPUT_CHARS = 4096

# How much of Venice's error body to quote back to the client.
_ERR_BODY_CHARS = 300

DEFAULT_MODEL = os.environ.get("VENICE_MODEL", "tts-kokoro")
DEFAULT_VOICE = os.environ.get("VENICE_VOICE", "af_heart")
KEY_FILE = os.path.expanduser(
    os.environ.get("VENICE_API_KEY_FILE",
                   "~/.config/marmalade-tts/venice-api-key"))
KEY_ENV = os.environ.get("VENICE_API_KEY_ENV", "VENICE_API_KEY")
TIMEOUT = float(os.environ.get("VENICE_TIMEOUT", "30"))

# Requests are network-bound and share no state, so several can be in
# flight at once (the local daemons serialize because one model can only
# run one inference at a time).
MAX_CONCURRENCY = 4

log = logging.getLogger("venice-daemon")


def _no_key_error() -> str:
    return (f"venice: no API key configured (write it to {KEY_FILE} with "
            f"mode 0600, or set the {KEY_ENV} environment variable)")


def _key_from_file() -> str | None:
    """Read the key file, warning if its mode lets anyone else read it."""
    try:
        st = os.stat(KEY_FILE)
    except OSError:
        return None
    if stat.S_IMODE(st.st_mode) & 0o077:
        log.warning("%s is readable by group/other — chmod 600 it", KEY_FILE)
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError as e:
        log.error("could not read %s: %s", KEY_FILE, e)
        return None


def api_key() -> str | None:
    """The key file wins over the env var; both are read per request so a
    key added after startup takes effect without a restart."""
    return _key_from_file() or (os.environ.get(KEY_ENV) or "").strip() or None


def _post(payload: dict, key: str) -> bytes:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace").strip()
        try:
            body = json.dumps(json.loads(body).get("error", body))
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(
            f"venice: HTTP {e.code} from {API_URL}: "
            f"{body[:_ERR_BODY_CHARS]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"venice: could not reach {API_URL}: "
                           f"{e.reason}") from None
    except OSError as e:
        # A timeout or reset during the read phase surfaces as a bare
        # OSError/TimeoutError, not URLError.
        raise RuntimeError(f"venice: request to {API_URL} failed: "
                           f"{e or type(e).__name__}") from None


def load_model():
    """Nothing to load — just report whether a key is available."""
    if api_key():
        log.info("API key found; model=%s voice=%s", DEFAULT_MODEL,
                 DEFAULT_VOICE)
        print(f"[venice-daemon] key OK — model={DEFAULT_MODEL} "
              f"voice={DEFAULT_VOICE}", flush=True)
    else:
        log.warning("%s — requests will fail until one is provided",
                    _no_key_error())
        print(f"[venice-daemon] WARNING: {_no_key_error()}", flush=True)
    return None


def synth(_model, req):
    text = (req.get("text") or "").strip()
    if not text:
        raise RuntimeError("venice: request has no text")
    if len(text) > MAX_INPUT_CHARS:
        raise RuntimeError(
            f"venice: input is {len(text)} chars, over the API limit of "
            f"{MAX_INPUT_CHARS}. Split it into sentences before sending.")

    key = api_key()
    if not key:
        raise RuntimeError(_no_key_error())

    audio = _post({
        "model": req.get("model") or DEFAULT_MODEL,
        "input": text,
        "voice": req.get("voice") or DEFAULT_VOICE,
        "response_format": "wav",
        "speed": float(req.get("speed", 1.0)),
    }, key)

    if audio[:4] != b"RIFF":
        # Some Venice models (tts-qwen3-*) ignore response_format and hand
        # back MP3. Writing those bytes to a .wav would hand the caller a
        # file that fails to play halfway through a sentence.
        raise RuntimeError(
            f"venice: model {req.get('model') or DEFAULT_MODEL!s} returned "
            f"non-WAV audio ({len(audio)} bytes starting {audio[:4]!r}); "
            f"use a model that honors response_format=wav (e.g. tts-kokoro).")

    with open(req["out"], "wb") as f:
        f.write(audio)


if __name__ == "__main__":
    serve("venice", load_model, synth, max_concurrency=MAX_CONCURRENCY)
