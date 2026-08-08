"""Kitten TTS engine — daemon client with subprocess fallback."""

import os

from . import Engine, run_in_venv
from .. import daemon as dmgr

KITTEN_VENV   = os.path.expanduser("~/.local/share/kittentts-venv")
KITTEN_PYTHON = os.path.join(KITTEN_VENV, "bin", "python")
DAEMON_SCRIPT = os.path.expanduser("~/.local/share/marmalade-tts/kitten-daemon.py")

MODEL_REPOS = {
    "nano":  "KittenML/kitten-tts-nano-0.8",
    "micro": "KittenML/kitten-tts-micro-0.8",
    "mini":  "KittenML/kitten-tts-mini-0.8",
}

VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]


def espeak_voice(lang) -> str | None:
    """A ``--lang`` value → the espeak voice the daemon phonemizes with.

    The detector's ISO-639-1 codes are translated (``fr`` → ``fr-fr``,
    ``pt`` → ``pt-br``) so the code a user reads in `--lang auto` output
    is also the code they can pin. Anything else goes through verbatim:
    espeak's own voice names are the engine's native vocabulary, so
    ``--lang cmn`` or ``--lang en-gb`` reach it unaltered and espeak
    rejects what it doesn't know.

    ``"auto"`` returns None rather than reaching espeak: a surviving
    sentinel means a resolution step was skipped upstream, and English is
    the right degrade (same rule as Android's
    ``KittenDirectEngine.espeakVoiceFor``).
    """
    from .. import langdetect
    if not lang or lang == "auto":
        return None
    return (langdetect.to_espeak_lang(lang)
            if lang in langdetect.DETECTED_LANGS else lang)


class KittenEngine(Engine):
    name = "kitten"
    MAX_CHARS = 500  # conservative — kitten's small CPU model degrades on long inputs

    # Kitten's model is trained on en-us IPA, but it phonemizes through
    # espeak like Kokoro does, so pointing espeak at another language is a
    # real (if accented) capability rather than a no-op — Max's 2026-08-08
    # call. The voice never changes: `--lang` moves the phonemizer only.
    SUPPORTS_LANG = True

    # The F chunking rules (Max's 2026-08-07 clause-split ear-lab pick):
    # plan in TEXT space via chunking.clause_chunks — the exact port of
    # Android's TextChunker.clauseChunks — and render each chunk plain
    # (per-chunk phonemize, no conditioning, no pad_marks, style row =
    # pre-split sentence's text length, graded 80/260 ms gaps). Max's
    # parity rule: CLI and Android must chunk and render identically so
    # a bug heard on one platform is a bug found on both.
    TEXT_CLAUSE_PLAN = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.voice = cfg.get("voice", "Kiki")
        self.model_size = cfg.get("model_size", "nano")
        # Not defaulted: None means "the daemon's own en-us", which is the
        # English path exactly as it was before `--lang` reached kitten.
        # "auto" here is resolved per utterance, same as on kokoro.
        self.lang = cfg.get("lang")
        self.use_daemon = cfg.get("daemon", True)
        # The kitten daemon runs concurrent requests (serve max_concurrency=4
        # with a phonemizer lock); the subprocess fallback must stay serial.
        self.PARALLEL_CHUNKS = self.use_daemon
        # The daemon renders a text prefix (context) and suffix (lookahead)
        # for cross-chunk prosody conditioning and cuts them out
        # sample-exactly (duration output).
        self.STREAM_CONTEXT = self.use_daemon
        self.STREAM_LOOKAHEAD = self.use_daemon
        # Phoneme-direct streaming: the client phonemizes once and plans the
        # whole stream in phoneme space (chunking.ph_stream_plan). Supersedes
        # STREAM_CONTEXT/STREAM_LOOKAHEAD on the streaming path — those stay
        # for the text path (subprocess mode, and any caller passing text).
        self.PHONEME_STREAM = self.use_daemon

    def _repo(self) -> str:
        return MODEL_REPOS.get(self.model_size, self.model_size)

    def synthesize(self, text: str, out_path: str, voice: str = None,
                   speed: float = 1.0, context: str = None,
                   lookahead: str = None, lang: str = None, **kwargs):
        v = voice or self.voice

        if self.use_daemon:
            # "model" lets the daemon verify it has this size loaded (it
            # loads one model at startup and refuses mismatches).
            request = {"text": text, "voice": v, "speed": speed,
                       "model": self._repo(), "out": out_path}
            if context:
                request["context"] = context
            if lookahead:
                request["lookahead"] = lookahead
            espeak = espeak_voice(lang or self.lang)
            if espeak:
                request["lang"] = espeak
            dmgr.synthesize("kitten", request, auto_start=True)
            return

        # Fallback: direct subprocess (slow cold start)
        cmd = [
            KITTEN_PYTHON, "-c",
            f"from kittentts import KittenTTS; "
            f"m = KittenTTS('{self._repo()}'); "
            f"m.generate_to_file({text!r}, {out_path!r}, voice={v!r}, speed={speed})",
        ]
        run_in_venv(KITTEN_PYTHON, cmd,
                    env_extra={"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
                    engine_name="kitten")

    def phonemize(self, text: str, voice: str = None, lang: str = None,
                  **kwargs) -> str:
        """espeak the whole utterance once, via the daemon (~2ms/paragraph).
        Phase 1 of the phoneme-direct streaming path.

        ``voice`` is the KITTEN speaker (Bella, Kiki, …) — it rides along
        so the daemon can validate it, and has nothing to do with espeak.
        ``lang`` is the espeak language; omitting it phonemizes English.
        """
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix="marmalade-ph-", suffix=".txt")
        os.close(fd)
        request = {"op": "phonemize", "text": text,
                   "voice": voice or self.voice,
                   "model": self._repo(), "out": tmp}
        espeak = espeak_voice(lang or self.lang)
        if espeak:
            request["lang"] = espeak
        try:
            dmgr.synthesize("kitten", request, auto_start=True)
            with open(tmp, encoding="utf-8") as f:
                return f.read().strip()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def synthesize_phonemes(self, ph_text: str, out_path: str, voice: str = None,
                            speed: float = 1.0, context: str = None,
                            lookahead: str = None, style_ref: int = None,
                            pad_marks: dict = None, lang: str = None,
                            **kwargs):
        """Synthesize from phonemes with exact-index conditioning cuts.

        ``style_ref`` pins the style-pack row (the wrapper indexes it by
        input length, and neighbouring rows are audibly different — Max's
        P11 verdict), ``pad_marks`` tops up a mark's rendered pause with
        inserted silence.

        ``lang`` is accepted and dropped: the phonemes arriving here were
        already produced in that language by :meth:`phonemize`, and this
        path never touches espeak."""
        request = {"ph_text": ph_text, "voice": voice or self.voice,
                   "speed": speed, "model": self._repo(), "out": out_path}
        if context:
            request["ph_context"] = context
        if lookahead:
            request["ph_lookahead"] = lookahead
        if style_ref is not None:
            request["style_ref"] = int(style_ref)
        if pad_marks:
            request["pad_marks"] = pad_marks
        dmgr.synthesize("kitten", request, auto_start=True)

    def list_voices(self):
        print("Language: trained on English; --lang / --lang auto point "
              "espeak at another language and the voice reads it accented")
        print(f"Kitten TTS voices: {', '.join(VOICES)}")
        # Upstream ships micro/mini only as dynamic-int8 ONNX (no fp32
        # published); fp32 nano beats them on quality AND speed.
        print("Model sizes: nano (fp32, fastest + best quality)  "
              "micro (int8)  mini (int8)")
        for size, repo in MODEL_REPOS.items():
            print(f"  {size}: {repo}")
