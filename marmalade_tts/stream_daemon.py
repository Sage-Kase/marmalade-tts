"""The streaming TTS session daemon — text in as it is written, chunk
WAVs out as they render.

The engine daemons (``daemon/*-daemon.py``) are one-shot: one JSON line
in, one complete WAV out, close. That is the wrong shape for a live
assistant turn, where the first sentence should be audible while the
model is still writing the third. This daemon is the streaming front end
for them: it owns the planning machinery (``stream_session``), the
anti-underrun playback gate (``stream_play.should_start``) and the
observed RTF stats (``perfstats``), and it drives the ordinary engine
objects — which, in daemon mode, means one socket request per chunk to
the engine daemon that holds the model.

Playback stays with the CLIENT. The daemon never plays, never
concatenates, and never deletes a WAV it has handed over.

Data flow
---------
    client ──(unix socket, newline-delimited JSON)──> _Connection.run
      op:speak  → _Utterance created (engine resolved from the request or
                  config), appended to this connection's FIFO
      op:text   → raw fragment appended to the utterance's fragment queue
      op:end    → no more fragments
      op:cancel → stop rendering this id as soon as the in-flight render
                  returns

    _Connection._work  (one worker thread per connection, ids FIFO)
      arrived text?   → stream_session.plan_fragment(...) → render items
      else pending?   → item.synthesize(engine, tmp.wav)     [engine daemon]
                      → perfstats.record(...)
                      → chunking.pad_wav_end(tmp.wav, gap)
                      → ev:chunk {id, seq, wav, text, sentence, dur_ms}
                      → gate: should_start(buffered, known-unrendered, est)
                              → ev:start (once, never revoked)
      else ended?     → ev:done   (ev:error on a render failure)

Planning runs BEFORE the next render, deliberately: a fragment that has
arrived but has not been planned is invisible to the gate, which would
then happily open on a slow engine with three sentences still queued. It
costs nothing at the start of a stream (nothing is queued ahead of the
first fragment, so time to first audio is untouched) and one
phonemization call — tens of milliseconds against a render's hundreds —
mid-stream.

Chunk WAVs are mkstemp'd, 0600 by construction, and owned by the client
from the moment its ``ev:chunk`` is written. The only files this daemon
unlinks are ones it rendered but never announced — after a cancel, or
after the connection dropped.

The wire contract is the ratified build spec in the marmalade core repo
(``docs/plans/voice-latency.md`` → "Ratified build spec (2026-08-17)").
``tests/fixtures/stream/session-basic.jsonl`` is the canonical transcript
both sides of that contract are checked against.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from collections import deque

from . import chunking, perfstats, stream_session
from .playback import wav_duration
from .stream_play import should_start

BASE_DIR = os.path.expanduser("~/.local/share/marmalade-tts")
SOCKET_PATH = os.path.join(BASE_DIR, "stream.sock")
PID_PATH = os.path.join(BASE_DIR, "stream.pid")
LOG_PATH = os.path.join(BASE_DIR, "stream.log")

# Same default as the CLI when config names no engine.
DEFAULT_ENGINE = "kitten"

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")

log = logging.getLogger("stream-daemon")


class ProtocolError(Exception):
    """A request the daemon can't act on — answered with ``ev:error``."""


# ── the start gate ──────────────────────────────────────────────────────────

def should_emit_start(*, rendered_any: bool, ended: bool, outstanding: int,
                      buffered_s: float, remaining: list, est) -> bool:
    """Should this utterance's one ``ev:start`` go out now?

    Evaluated on every chunk completion and every ``op:text``, over
    KNOWN-unrendered chunks only (``remaining``). ``outstanding`` counts
    known work that has no entry in ``remaining`` yet — fragments received
    but not planned, plus whatever the worker has in flight right now;
    without it, ``end`` arriving in the gap between "popped the fragment"
    and "planned it" would look like an utterance with nothing left to do.

    Two ways to pass:

    * everything the daemon will ever render is rendered (``end`` arrived,
      nothing outstanding, nothing pending) — including the degenerate
      utterance with no chunks at all; or
    * at least one chunk is rendered AND the ordinary anti-underrun
      inequality holds (``stream_play.should_start``).

    The second clause's "at least one chunk" is the cold-start rule: with
    no chunk rendered there is no audio to start on, and — since
    ``should_start`` returns True for an empty remaining list — no gate
    left to fail. Short fragments therefore usually degenerate to
    first-chunk-ready = start, which is the whole TTFA win.
    """
    if ended and not outstanding and not remaining:
        return True
    if not rendered_any:
        return False
    return should_start(buffered_s, remaining, est)


# ── per-utterance state ─────────────────────────────────────────────────────

class _Utterance:
    """One ``speak`` id: its engine, its fragment queue, its plan, and the
    bookkeeping the gate reads. Every mutation happens under ``lock``;
    renders happen outside it."""

    def __init__(self, uid, engine, engine_name, eng_cfg, synth_kwargs,
                 stats_key, ph_key):
        self.id = uid
        self.engine = engine
        self.engine_name = engine_name
        self.eng_cfg = eng_cfg
        self.synth_kwargs = synth_kwargs
        self.stats_key = stats_key      # perfstats key the gate reads
        self.text_key = stats_key
        self.ph_key = ph_key

        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)

        self.fragments = deque()        # raw text fragments, unplanned
        self.pending = deque()          # planned RenderItems, unrendered
        self.in_flight = False          # worker is planning/rendering one
        self.step = 0                   # ramp offset carried across them
        self.sentence = 0               # index of the next fragment

        self.seq = 0                    # next chunk sequence number
        self.buffered_s = 0.0           # audio rendered but not yet started
        self.rendered_any = False
        self.started = False
        self.ended = False
        self.cancelled = False
        self.finished = False
        self.active = False             # the worker is on this utterance

    # Gate inputs, computed under the lock by the caller.
    def _remaining(self):
        return [it.gate_item() for it in self.pending]

    def _estimate(self):
        est = perfstats.estimate(self.engine_name, self.stats_key)
        if est is not None and self.stats_key == self.ph_key:
            # Phoneme path: conditioning chars are known exactly, so the
            # gate budgets rendered audio at the size-independent marginal
            # rtf (see stream_play.should_start).
            mrtf = perfstats.estimate_marginal(self.engine_name,
                                               self.stats_key)
            if mrtf is not None:
                est = (mrtf, est[1])
        return est


# ── one client connection ───────────────────────────────────────────────────

class _Connection:
    """Reader thread (this class' ``run``) + one render worker thread.

    Utterances render FIFO in ``speak`` order — a connection is one
    speaker, and two turns must not interleave in the ear.
    """

    def __init__(self, server, sock):
        self.server = server
        self.sock = sock
        self.write_lock = threading.Lock()
        self.closed = False

        self.utterances = {}                    # id → _Utterance (live)
        self.queue = deque()                    # FIFO for the worker
        self.qcv = threading.Condition()
        self.stopping = False
        self.worker = threading.Thread(target=self._work, daemon=True,
                                       name="marmalade-stream-render")

    # ── wire I/O ─────────────────────────────────────────────────────────
    def send(self, obj: dict) -> None:
        line = (json.dumps(obj) + "\n").encode()
        with self.write_lock:
            if self.closed:
                return
            try:
                self.sock.sendall(line)
            except OSError:
                # The client vanished mid-utterance; the worker will see
                # the cancellation the reader raises on its way out.
                self.closed = True

    def run(self) -> None:
        self.worker.start()
        buf = b""
        try:
            while True:
                try:
                    data = self.sock.recv(65536)
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._dispatch(line)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """The connection is gone: cancel everything it opened, let the
        worker clean up any WAV it renders from here on, and stop it."""
        with self.write_lock:
            self.closed = True
        with self.qcv:
            self.stopping = True
        for u in list(self.utterances.values()):
            self._cancel(u)
        with self.qcv:
            self.qcv.notify_all()
        self.worker.join(timeout=30.0)
        try:
            self.sock.close()
        except OSError:
            pass

    # ── request dispatch ─────────────────────────────────────────────────
    def _dispatch(self, line: bytes) -> None:
        uid = None
        try:
            try:
                req = json.loads(line.decode("utf-8", "replace"))
            except ValueError as e:
                raise ProtocolError(f"malformed JSON: {e}") from None
            if not isinstance(req, dict):
                raise ProtocolError("request must be a JSON object")
            uid = req.get("id")
            op = req.get("op")
            if op == "speak":
                self._op_speak(req)
            elif op == "text":
                self._op_text(req)
            elif op == "end":
                self._op_end(req)
            elif op == "cancel":
                self._op_cancel(req)
            else:
                raise ProtocolError(f"unknown op {op!r}")
        except ProtocolError as e:
            log.warning("protocol error (id=%s): %s", uid, e)
            self.send({"ev": "error", "id": uid, "error": str(e)})
        except Exception as e:  # never let one bad request kill the session
            log.exception("request failed (id=%s)", uid)
            self.send({"ev": "error", "id": uid, "error": str(e)})

    def _require_id(self, req) -> str:
        uid = req.get("id")
        if not isinstance(uid, str) or not uid:
            raise ProtocolError("request needs a string 'id'")
        return uid

    def _live(self, req) -> _Utterance:
        uid = self._require_id(req)
        u = self.utterances.get(uid)
        if u is None:
            raise ProtocolError(f"unknown utterance id {uid!r}")
        return u

    def _op_speak(self, req) -> None:
        uid = self._require_id(req)
        if uid in self.utterances:
            raise ProtocolError(f"utterance id {uid!r} is already open")
        u = self.server.new_utterance(uid, req)
        self.utterances[uid] = u
        with self.qcv:
            self.queue.append(u)
            self.qcv.notify_all()

    def _op_text(self, req) -> None:
        u = self._live(req)
        text = req.get("text")
        if not isinstance(text, str):
            raise ProtocolError("'text' must be a string")
        with u.lock:
            if u.ended:
                # A client bug, not a race: one thread writes one
                # utterance. Terminal, so the id can't linger half-open.
                u.cancelled = True
                u.cv.notify_all()
                self._finish(u, error="text after end")
                return
            if u.cancelled or u.finished:
                return
            u.fragments.append(text)
            u.cv.notify_all()
            # The gate is evaluated on every op:text too — over what is
            # KNOWN unrendered, which this text is not yet part of.
            self._maybe_start(u)

    def _op_end(self, req) -> None:
        u = self._live(req)
        with u.lock:
            u.ended = True
            u.cv.notify_all()
            self._maybe_start(u)

    def _op_cancel(self, req) -> None:
        uid = self._require_id(req)
        u = self.utterances.get(uid)
        # Cancel is idempotent: cancelling an id that already finished (or
        # never existed) is a benign race with ``done`` in flight, not an
        # error the client can act on.
        if u is not None:
            self._cancel(u)

    def _cancel(self, u: _Utterance) -> None:
        with u.lock:
            if u.finished:
                return
            u.cancelled = True
            u.fragments.clear()
            u.pending.clear()
            u.cv.notify_all()
            if not u.active:
                # Nothing in flight — finish it here. An active utterance
                # is finished by the worker, after its in-flight render
                # returns and its WAV is deleted.
                self._finish(u)

    # ── events ───────────────────────────────────────────────────────────
    def _maybe_start(self, u: _Utterance) -> None:
        """Emit ``ev:start`` if the gate passes. Caller holds ``u.lock``.
        Never revoked: a late underrun lands on a fragment boundary, which
        is a natural pause."""
        if u.started or u.cancelled or u.finished:
            return
        if should_emit_start(rendered_any=u.rendered_any, ended=u.ended,
                             outstanding=len(u.fragments) + int(u.in_flight),
                             buffered_s=u.buffered_s,
                             remaining=u._remaining(), est=u._estimate()):
            u.started = True
            self.send({"ev": "start", "id": u.id})

    def _finish(self, u: _Utterance, error: str | None = None) -> None:
        """Terminal event for an id — ``done``, or ``error`` when a render
        failed. Cancelled ids get one too. Caller holds ``u.lock``."""
        if u.finished:
            return
        u.finished = True
        self.utterances.pop(u.id, None)
        if error is None:
            self.send({"ev": "done", "id": u.id})
        else:
            self.send({"ev": "error", "id": u.id, "error": error})

    # ── render worker ────────────────────────────────────────────────────
    def _work(self) -> None:
        while True:
            with self.qcv:
                while not self.queue and not self.stopping:
                    self.qcv.wait()
                if not self.queue:
                    return
                u = self.queue.popleft()
            try:
                self._run_utterance(u)
            except Exception as e:  # a bug here must not wedge the client
                log.exception("utterance %s failed", u.id)
                with u.lock:
                    self._finish(u, error=str(e))

    def _run_utterance(self, u: _Utterance) -> None:
        with u.lock:
            if u.finished:
                return
            u.active = True
        error = None
        try:
            while True:
                with u.cv:
                    while not (u.cancelled or u.pending or u.fragments
                               or u.ended):
                        u.cv.wait()
                    if u.cancelled:
                        return
                    # Plan first: unplanned text is invisible to the gate.
                    frag = u.fragments.popleft() if u.fragments else None
                    item = (u.pending.popleft()
                            if frag is None and u.pending else None)
                    done = item is None and frag is None   # ended, drained
                    u.in_flight = not done
                if done:
                    return
                try:
                    if frag is not None:
                        self._plan(u, frag)
                    else:
                        self._render(u, item)
                finally:
                    # _plan/_render clear this themselves before they
                    # evaluate the gate; this catches the throwing paths.
                    with u.lock:
                        u.in_flight = False
        except Exception as e:
            error = str(e)
            log.exception("render failed for %s", u.id)
        finally:
            with u.lock:
                u.active = False
                if error is not None:
                    self._finish(u, error=error)
                else:
                    self._maybe_start(u)
                    self._finish(u)

    def _plan(self, u: _Utterance, fragment: str) -> None:
        """Plan one arrived fragment. Runs on the worker (planning can hit
        the engine daemon for phonemization), so it never blocks reads."""
        processed = self.server.preprocess(fragment, u)
        plan = stream_session.plan_fragment(
            u.engine, processed, engine_name=u.engine_name,
            eng_cfg=u.eng_cfg, synth_kwargs=u.synth_kwargs,
            sentence=u.sentence, sentence_text=fragment,
            step=u.step, ph_key=u.ph_key)
        with u.lock:
            u.in_flight = False
            u.sentence += 1
            if u.cancelled:
                return
            u.step = plan.next_step
            u.pending.extend(plan.items)
            if plan.items:
                # Mixed-path utterances are pathological; the gate reads
                # whichever space the most recent fragment planned in, so
                # its char counts and the estimate agree.
                u.stats_key = (u.ph_key if plan.items[0].phonemes
                               else u.text_key)
            self._maybe_start(u)

    def _render(self, u: _Utterance, item) -> None:
        fd, path = tempfile.mkstemp(
            prefix=f"marmalade-stream-{_SAFE_ID.sub('', u.id)[:24]}-"
                   f"{u.seq:03d}-",
            suffix=".wav")
        os.close(fd)
        emitted = False
        try:
            t0 = time.monotonic()
            item.synthesize(u.engine, path, **u.synth_kwargs)
            dt = time.monotonic() - t0
            dur = wav_duration(path)
            key = u.ph_key if item.phonemes else u.text_key
            # Sequential renders: every chunk has the engine to itself, so
            # each is an honest (solo) speed sample. Only the first is
            # recorded as such, matching stream_play's one-per-utterance
            # uncontended measurement.
            perfstats.record(u.engine_name, key, len(item.text), dt, dur,
                             cond_chars=item.cond_chars,
                             solo=not u.rendered_any)
            if item.gap_after_ms:
                # The gap rides inside the chunk's own WAV, so the client
                # gets one duration and one file to play.
                chunking.pad_wav_end(path, item.gap_after_ms)
                dur += item.gap_after_ms / 1000.0
            with u.lock:
                u.in_flight = False
                if u.cancelled:
                    return
                seq = u.seq
                u.seq += 1
                u.rendered_any = True
                u.buffered_s += dur
                self.send({"ev": "chunk", "id": u.id, "seq": seq,
                           "wav": path, "text": item.sentence_text,
                           "sentence": item.sentence,
                           "dur_ms": int(round(dur * 1000))})
                emitted = True
                self._maybe_start(u)
        finally:
            if not emitted:
                # Rendered but never announced (cancel, dropped connection,
                # render error) — the client can't own what it never saw.
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ── the server ──────────────────────────────────────────────────────────────

class StreamServer:
    """Serves the stream socket. ``engine_factory(name, eng_cfg)`` is an
    injection point for tests; by default engines are built exactly as the
    CLI builds them, from the same config."""

    def __init__(self, socket_path: str = SOCKET_PATH, *, config=None,
                 engine_factory=None, backlog: int = 16):
        self.socket_path = socket_path
        self._config = config
        self._engine_factory = engine_factory
        self.backlog = backlog
        self.server = None
        self._engines = {}
        self._elock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()

    # ── config / engines ─────────────────────────────────────────────────
    @property
    def config(self) -> dict:
        """Loaded once — like the engine daemons, a config change needs a
        restart to take effect."""
        if self._config is None:
            from . import config as cfg_mod
            self._config = cfg_mod.load()
        return self._config

    def engine_for(self, name: str):
        """``(engine, eng_cfg)`` for an engine name.

        Engine objects are stateless config holders (voice/speed/lang ride
        on each request), so one per engine name is cached for the
        daemon's lifetime.
        """
        from . import config as cfg_mod
        eng_cfg = cfg_mod.engine_cfg(self.config, name)
        with self._elock:
            engine = self._engines.get(name)
        if engine is not None:
            return engine, eng_cfg
        if self._engine_factory is not None:
            engine = self._engine_factory(name, eng_cfg)
        else:
            from .cli import ENGINE_CLASSES
            if name not in ENGINE_CLASSES:
                raise ProtocolError(
                    f"unknown engine {name!r}; known: "
                    f"{', '.join(ENGINE_CLASSES)}")
            engine = ENGINE_CLASSES[name](eng_cfg)
        with self._elock:
            engine = self._engines.setdefault(name, engine)
        return engine, eng_cfg

    def new_utterance(self, uid: str, req: dict) -> _Utterance:
        """Build the utterance state for one ``op:speak``."""
        name = req.get("engine") or self.config.get("defaults", {}).get(
            "engine", DEFAULT_ENGINE)
        if not isinstance(name, str):
            raise ProtocolError("'engine' must be a string")
        engine, eng_cfg = self.engine_for(name)

        speed = req.get("speed")
        if speed is None:
            speed = eng_cfg.get("speed",
                                self.config.get("defaults", {}).get("speed",
                                                                    1.0))
        synth_kwargs = {"speed": float(speed)}
        if req.get("voice"):
            synth_kwargs["voice"] = str(req["voice"])
        lang = req.get("lang")
        if lang:
            synth_kwargs["lang"] = str(lang)

        mkey = perfstats.model_key(engine, eng_cfg)
        return _Utterance(uid, engine, name, eng_cfg, synth_kwargs,
                          stats_key=mkey, ph_key=f"{mkey}:ph")

    def preprocess(self, text: str, u: _Utterance) -> str:
        """Same text normalization the CLI applies (config-driven), plus
        ``--lang auto`` resolution — per fragment here, since that is the
        largest unit the daemon ever has."""
        from . import synth
        out = synth.apply_preprocessing(
            text, engine_name=u.engine_name, eng_cfg=u.eng_cfg,
            config=self.config, preprocess_mode=None, custom_rules=None)
        if u.synth_kwargs.get("lang") == "auto":
            from . import langdetect
            resolved = langdetect.resolve_auto_lang(
                u.engine, u.engine_name, out, dict(u.synth_kwargs))
            u.synth_kwargs = resolved
        return out

    # ── lifecycle ────────────────────────────────────────────────────────
    def bind(self) -> None:
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        # A poll timeout, not a deadline: closing a listening socket does
        # not wake another thread blocked in accept(), so shutdown would
        # otherwise wait for the next connection. Accepted sockets are
        # blocking regardless (Python 3.7+).
        self.server.settimeout(0.25)
        self.server.listen(self.backlog)

    def serve_forever(self) -> None:
        if self.server is None:
            self.bind()
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            c = _Connection(self, conn)
            threading.Thread(target=c.run, daemon=True,
                             name="marmalade-stream-conn").start()

    def start(self) -> None:
        """Bind and serve on a background thread (tests, embedding)."""
        self.bind()
        self._thread = threading.Thread(target=self.serve_forever,
                                        daemon=True,
                                        name="marmalade-stream-accept")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass


def main(argv=None) -> int:
    """Entry point for ``daemon/stream-daemon.py``."""
    os.makedirs(BASE_DIR, exist_ok=True)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    server = StreamServer(SOCKET_PATH)
    server.bind()
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def _shutdown(sig, _frame):
        log.info("shutting down (signal %s)", sig)
        server.close()
        try:
            os.unlink(PID_PATH)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    log.info("listening on %s", SOCKET_PATH)
    print(f"[stream-daemon] ready — socket: {SOCKET_PATH}", flush=True)
    server.serve_forever()
    return 0
