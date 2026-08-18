"""Tests for the streaming session daemon (marmalade_tts.stream_daemon).

Every session here runs over a REAL Unix socket against a fake engine —
no model, no engine daemon — so the protocol, the gate, the chunk-file
lifecycle and the FIFO ordering are exercised end to end in-process.

``tests/fixtures/stream/session-basic.jsonl`` is the canonical wire
transcript of a basic session (both directions, wav paths and durations
normalized). It is the cross-repo parity artifact: the voice package
drives its fake daemon from the same file, so a change to the wire on
either side shows up as a diff here. Refresh it with
``MARMALADE_REFRESH_FIXTURES=1 pytest tests/test_stream_daemon.py``.
"""

import json
import os
import socket
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from marmalade_tts import chunking, perfstats
from marmalade_tts.stream_daemon import StreamServer, should_emit_start

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "stream")
BASIC_FIXTURE = os.path.join(FIXTURE_DIR, "session-basic.jsonl")

CHUNK_S = 0.05          # audio every fake render produces
CHUNK_MS = 50
RATE = 24000


def _silent_wav(path, duration_s=CHUNK_S, rate=RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * duration_s))


class FakeEngine:
    """A PHONEME_STREAM engine with no model. Short fragments plan to one
    piece each, which keeps the transcripts readable."""

    MAX_CHARS = 500
    PHONEME_STREAM = True
    STYLE_ROWS = "ph-utterance"
    model_size = "test"

    def __init__(self):
        self.paths = []             # every WAV this engine wrote, in order
        self.entered = threading.Event()
        self.release = None         # set to an Event to block renders
        self.block_from = 0
        self.fail_from = None
        self.lock = threading.Lock()

    def phonemize(self, text, **kw):
        return " ".join("ˈ" + w for w in text.split())

    def synthesize_phonemes(self, ph_text, out_path, **kw):
        with self.lock:
            n = len(self.paths)
            self.paths.append(out_path)
        if self.fail_from is not None and n >= self.fail_from:
            raise RuntimeError("render exploded")
        _silent_wav(out_path)
        if self.release is not None and n >= self.block_from:
            self.entered.set()
            assert self.release.wait(10), "render never released"


CONFIG = {"defaults": {"engine": "fake", "preprocessing": False, "speed": 1.0}}


class Client:
    """A test client that records the whole conversation, in order."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect(path)
        self.buf = b""
        self.transcript = []        # ("in"|"out", obj) in wire order

    def send(self, obj):
        self.transcript.append(("in", obj))
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def send_raw(self, data: bytes):
        self.sock.sendall(data)

    def recv(self):
        while b"\n" not in self.buf:
            data = self.sock.recv(65536)
            if not data:
                raise AssertionError("daemon closed the connection")
            self.buf += data
        line, self.buf = self.buf.split(b"\n", 1)
        obj = json.loads(line)
        self.transcript.append(("out", obj))
        return obj

    def drain_until(self, ev, uid=None):
        """Collect events until (and including) ``ev`` for ``uid``."""
        out = []
        while True:
            e = self.recv()
            out.append(e)
            if e.get("ev") == ev and (uid is None or e.get("id") == uid):
                return out

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Rig:
    def __init__(self, server, engine, path):
        self.server = server
        self.engine = engine
        self.path = path
        self.clients = []

    def client(self):
        c = Client(self.path)
        self.clients.append(c)
        return c


@pytest.fixture
def rig(tmp_path):
    engine = FakeEngine()
    path = str(tmp_path / "stream.sock")
    server = StreamServer(path, config=CONFIG,
                          engine_factory=lambda name, cfg: engine)
    server.start()
    r = Rig(server, engine, path)
    try:
        yield r
    finally:
        for c in r.clients:
            c.close()
        server.close()
        for p in engine.paths:      # tests own emitted chunks, like clients
            try:
                os.unlink(p)
            except OSError:
                pass


def _wait_for(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ── the gate, as a pure rule ────────────────────────────────────────────────

class TestStartGate:
    def test_cold_start_waits_for_the_first_chunk(self):
        # Nothing rendered: there is no audio to start on, and an empty
        # remaining list would otherwise pass trivially.
        assert not should_emit_start(rendered_any=False, ended=False,
                                     outstanding=0, buffered_s=0.0,
                                     remaining=[], est=None)

    def test_first_chunk_with_nothing_known_left_opens_the_gate(self):
        # The TTFA win: a short fragment degenerates to first-chunk-ready.
        assert should_emit_start(rendered_any=True, ended=False, outstanding=0,
                                 buffered_s=0.2, remaining=[], est=None)

    def test_slow_engine_with_known_work_left_waits(self):
        est = (1.5, 10.0)
        assert not should_emit_start(rendered_any=True, ended=False,
                                     outstanding=0, buffered_s=0.5,
                                     remaining=[(100, 100, 0.0)] * 3,
                                     est=est)

    def test_end_with_everything_rendered_opens_the_gate(self):
        est = (50.0, 10.0)          # hopeless engine — irrelevant now
        assert should_emit_start(rendered_any=True, ended=True, outstanding=0,
                                 buffered_s=0.1, remaining=[], est=est)

    def test_end_with_text_still_unplanned_does_not(self):
        assert not should_emit_start(rendered_any=False, ended=True,
                                     outstanding=1, buffered_s=0.0,
                                     remaining=[], est=None)

    def test_empty_utterance_still_starts(self):
        # `end` with nothing to say: one start, then done — no special
        # case for the client to carry.
        assert should_emit_start(rendered_any=False, ended=True, outstanding=0,
                                 buffered_s=0.0, remaining=[], est=None)


# ── a whole session ─────────────────────────────────────────────────────────

class TestSession:
    def test_speak_text_end_produces_chunks_then_done(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        first = c.recv()
        assert first["ev"] == "chunk"          # cold start: chunk, then gate
        assert first["id"] == "u1"
        assert first["seq"] == 0
        assert first["sentence"] == 0
        assert first["text"] == "One one one."
        assert os.path.exists(first["wav"])
        assert c.recv() == {"ev": "start", "id": "u1"}

        c.send({"op": "text", "id": "u1", "text": "Two two two."})
        second = c.recv()
        assert (second["seq"], second["sentence"], second["text"]) == \
            (1, 1, "Two two two.")

        c.send({"op": "end", "id": "u1"})
        assert c.recv() == {"ev": "done", "id": "u1"}

    def test_dur_ms_includes_the_padded_gap(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        chunk = c.recv()
        assert chunk["dur_ms"] == CHUNK_MS + chunking.RUN_GAP_MS
        with wave.open(chunk["wav"], "rb") as w:
            played = w.getnframes() / w.getframerate()
        # The number the client does arithmetic with IS the file's length.
        assert played == pytest.approx(chunk["dur_ms"] / 1000.0, abs=0.001)

    def test_multi_chunk_fragment_keeps_one_sentence_index(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1", "engine": "fake"})
        long_text = ("the market square filled with people carrying baskets "
                     "and the bells rang out over the rooftops until dusk.")
        c.send({"op": "text", "id": "u1", "text": long_text})
        c.send({"op": "end", "id": "u1"})
        events = c.drain_until("done", "u1")
        chunks = [e for e in events if e["ev"] == "chunk"]
        assert len(chunks) >= 2
        assert {e["sentence"] for e in chunks} == {0}
        assert {e["text"] for e in chunks} == {long_text}
        assert [e["seq"] for e in chunks] == list(range(len(chunks)))

    def test_empty_utterance_starts_and_finishes(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "end", "id": "u1"})
        assert c.recv() == {"ev": "start", "id": "u1"}
        assert c.recv() == {"ev": "done", "id": "u1"}

    def test_voice_and_speed_reach_the_engine(self, rig):
        captured = {}
        orig = rig.engine.synthesize_phonemes

        def spy(ph_text, out_path, **kw):
            captured.update(kw)
            orig(ph_text, out_path, **kw)
        rig.engine.synthesize_phonemes = spy

        c = rig.client()
        c.send({"op": "speak", "id": "u1", "voice": "george", "speed": 1.3})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        c.send({"op": "end", "id": "u1"})
        c.drain_until("done", "u1")
        assert captured["voice"] == "george"
        assert captured["speed"] == 1.3

    def test_render_failure_ends_the_id_with_an_error(self, rig):
        rig.engine.fail_from = 0
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] == "u1"
        assert "exploded" in ev["error"]
        # The half-written chunk was never announced, so the daemon owns
        # it — and deleted it.
        assert not os.path.exists(rig.engine.paths[0])

    def test_two_ids_render_fifo(self, rig):
        c = rig.client()
        for uid, text in (("u1", "One one one."), ("u2", "Two two two.")):
            c.send({"op": "speak", "id": uid})
            c.send({"op": "text", "id": uid, "text": text})
            c.send({"op": "end", "id": uid})
        events = c.drain_until("done", "u2")
        ids = [e["id"] for e in events]
        assert ids.index("u2") > max(i for i, x in enumerate(ids)
                                     if x == "u1")
        assert [e["ev"] for e in events if e["id"] == "u2"] == \
            ["chunk", "start", "done"]


# ── the gate, over the wire ─────────────────────────────────────────────────

class TestGateOverTheWire:
    def test_slow_engine_holds_start_until_the_work_is_done(self, rig):
        # Poison the observed stats: rtf 50 means every remaining chunk
        # misses its deadline, so the gate may not open while known work
        # is outstanding.
        perfstats.record("fake", "test:ph", chars=100, render_s=250.0,
                         audio_s=5.0)
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        long_text = ("the market square filled with people carrying baskets "
                     "and the bells rang out over the rooftops until dusk.")
        c.send({"op": "text", "id": "u1", "text": long_text})
        c.send({"op": "end", "id": "u1"})
        events = c.drain_until("done", "u1")
        kinds = [e["ev"] for e in events]
        assert kinds.count("start") == 1
        assert kinds.index("start") == len(kinds) - 2   # last thing before done
        assert kinds.count("chunk") >= 2


# ── cancel and the chunk-file lifecycle ─────────────────────────────────────

class TestCancel:
    def test_cancel_drops_the_unemitted_wav_and_still_says_done(self, rig):
        rig.engine.release = threading.Event()
        rig.engine.block_from = 1               # block the SECOND render
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        emitted = c.recv()
        assert emitted["ev"] == "chunk"
        assert c.recv()["ev"] == "start"

        c.send({"op": "text", "id": "u1", "text": "Two two two."})
        assert rig.engine.entered.wait(10)       # render 2 is in flight
        c.send({"op": "cancel", "id": "u1"})
        # Give the cancel time to land while the render is still blocked,
        # then let it finish: the daemon must throw the WAV away.
        time.sleep(0.05)
        rig.engine.release.set()

        assert c.recv() == {"ev": "done", "id": "u1"}
        in_flight = rig.engine.paths[1]
        assert _wait_for(lambda: not os.path.exists(in_flight))
        # The chunk the client was already given is untouched — it owns it.
        assert os.path.exists(emitted["wav"])

    def test_cancel_before_anything_renders_is_immediate(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "cancel", "id": "u1"})
        assert c.recv() == {"ev": "done", "id": "u1"}

    def test_cancel_of_an_unknown_id_is_a_no_op(self, rig):
        # Racing a `done` already in flight is benign, not an error.
        c = rig.client()
        c.send({"op": "cancel", "id": "ghost"})
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "end", "id": "u1"})
        assert c.recv() == {"ev": "start", "id": "u1"}
        assert c.recv() == {"ev": "done", "id": "u1"}

    def test_a_dropped_connection_cancels_and_cleans_up(self, rig):
        rig.engine.release = threading.Event()
        rig.engine.block_from = 1
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "text", "id": "u1", "text": "One one one."})
        emitted = c.recv()
        assert emitted["ev"] == "chunk"
        c.send({"op": "text", "id": "u1", "text": "Two two two."})
        assert rig.engine.entered.wait(10)

        c.close()
        time.sleep(0.05)
        rig.engine.release.set()

        in_flight = rig.engine.paths[1]
        assert _wait_for(lambda: not os.path.exists(in_flight))
        assert os.path.exists(emitted["wav"])
        # The renderer stopped rather than working through the queue.
        assert _wait_for(lambda: len(rig.engine.paths) == 2)
        time.sleep(0.1)
        assert len(rig.engine.paths) == 2


# ── protocol errors ─────────────────────────────────────────────────────────

class TestProtocolErrors:
    def test_malformed_json_is_reported_and_the_session_survives(self, rig):
        c = rig.client()
        c.send_raw(b"{not json\n")
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] is None
        # Same connection still works.
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "end", "id": "u1"})
        assert c.recv()["ev"] == "start"

    def test_unknown_op(self, rig):
        c = rig.client()
        c.send({"op": "sing", "id": "u1"})
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] == "u1"
        assert "sing" in ev["error"]

    def test_text_for_an_unknown_id(self, rig):
        c = rig.client()
        c.send({"op": "text", "id": "nope", "text": "hi"})
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] == "nope"

    def test_speak_without_an_id(self, rig):
        c = rig.client()
        c.send({"op": "speak"})
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] is None

    def test_duplicate_speak_id(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "speak", "id": "u1"})
        ev = c.recv()
        assert ev["ev"] == "error" and "already open" in ev["error"]

    def test_unknown_engine(self, rig):
        server = rig.server
        server._engine_factory = None            # exercise the real lookup
        c = rig.client()
        c.send({"op": "speak", "id": "u1", "engine": "nosuchengine"})
        ev = c.recv()
        assert ev["ev"] == "error" and "nosuchengine" in ev["error"]

    def test_text_after_end_is_terminal(self, rig):
        c = rig.client()
        c.send({"op": "speak", "id": "u1"})
        c.send({"op": "end", "id": "u1"})
        c.drain_until("done", "u1")
        c.send({"op": "text", "id": "u1", "text": "late"})
        ev = c.recv()
        assert ev["ev"] == "error" and ev["id"] == "u1"


# ── the canonical wire transcript ───────────────────────────────────────────

def _normalize(entry):
    """Placeholders for the two values that can't be deterministic: the
    mkstemp path and the measured audio length."""
    direction, obj = entry
    obj = dict(obj)
    if "wav" in obj:
        obj["wav"] = "<wav>"
    if "dur_ms" in obj:
        obj["dur_ms"] = "<dur_ms>"
    return {"dir": direction, "msg": obj}


class TestWireFixture:
    def test_basic_session_matches_the_canonical_transcript(self, rig):
        """Drive a lock-step session and compare it, line for line, with
        the fixture the voice package replays. Lock-step (each `text` sent
        only after the previous fragment's events arrived) is what makes
        the interleaving deterministic."""
        c = rig.client()
        c.send({"op": "speak", "id": "utt-1", "engine": "fake"})
        c.send({"op": "text", "id": "utt-1", "text": "One one one."})
        assert c.recv()["ev"] == "chunk"
        assert c.recv()["ev"] == "start"
        c.send({"op": "text", "id": "utt-1", "text": "Two two two."})
        assert c.recv()["ev"] == "chunk"
        c.send({"op": "end", "id": "utt-1"})
        assert c.recv()["ev"] == "done"

        lines = [json.dumps(_normalize(e), ensure_ascii=False, sort_keys=True)
                 for e in c.transcript]
        blob = "\n".join(lines) + "\n"

        if (os.environ.get("MARMALADE_REFRESH_FIXTURES")
                or not os.path.exists(BASIC_FIXTURE)):
            os.makedirs(FIXTURE_DIR, exist_ok=True)
            with open(BASIC_FIXTURE, "w", encoding="utf-8") as f:
                f.write(blob)
        with open(BASIC_FIXTURE, encoding="utf-8") as f:
            assert f.read() == blob

    def test_fixture_is_valid_jsonl_with_both_directions(self):
        with open(BASIC_FIXTURE, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        assert {r["dir"] for r in rows} == {"in", "out"}
        assert [r["msg"]["op"] for r in rows if r["dir"] == "in"] == \
            ["speak", "text", "text", "end"]
        assert [r["msg"]["ev"] for r in rows if r["dir"] == "out"] == \
            ["chunk", "start", "chunk", "done"]
