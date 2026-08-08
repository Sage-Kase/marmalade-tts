"""Kokoro TTS engine — daemon client with subprocess fallback.

Voice naming
------------
Each kokoro voice has a canonical upstream ID of the form
`<lang><gender>_<name>` (e.g. ``bm_george`` for "British male, George").
marmalade-tts exposes voices by their bare name as the primary form
(e.g. ``george``); the canonical IDs continue to work too.

Voice identity (the embedding) and the language used for pronunciation
are orthogonal in kokoro — the same voice embedding can speak any
supported language. Each voice has a "natural" language; if no other
language is configured, that's what we use. Override order:

  1. ``--lang`` flag on the CLI
  2. ``engines.kokoro.lang`` in config.yaml
  3. The voice's natural language
  4. Fallback to American English (``a``)

``auto`` (either source) means "detect the language from the text", once
per utterance, in :mod:`marmalade_tts.langdetect`. Detection only ever
changes the pronunciation language, never the voice; a detected English
resolves to the voice's own English variant, and an uncertain detection
falls back to the precedence above.
"""

import os

from . import Engine, run_in_venv
from .. import chunking, daemon as dmgr

# marmalade-tts owns the install: kokoro lives in its own venv and is
# invoked by explicit path, never via $PATH. A bare `kokoro` lookup would
# silently "work" against an unrelated install (or fail confusingly) — an
# explicit venv path makes a working install unambiguous and lets the
# hands-off installer self-test the engine exactly the way the CLI runs it.
KOKORO_VENV = os.path.expanduser("~/.local/share/kokoro-venv")
KOKORO_BIN = os.path.join(KOKORO_VENV, "bin", "kokoro")


# Bare name → canonical kokoro voice ID. Bare names are unique across
# languages today; if upstream adds a name that collides, the canonical
# form continues to work and the bare form will raise.
VOICE_ALIASES = {
    # American English (a)
    "heart":      "af_heart",
    "bella":      "af_bella",
    "nicole":     "af_nicole",
    "adam":       "am_adam",
    "michael":    "am_michael",
    # British English (b)
    "emma":       "bf_emma",
    "isabella":   "bf_isabella",
    "george":     "bm_george",
    "lewis":      "bm_lewis",
    # Japanese (j)
    "alpha":      "jf_alpha",
    "gongitsune": "jf_gongitsune",
    "kumo":       "jm_kumo",
    # Mandarin (z)
    "xiaobei":    "zf_xiaobei",
    "yunjian":    "zm_yunjian",
}

# Canonical voice ID → natural language code (the prefix's first letter).
VOICE_NATURAL_LANG = {canonical: canonical[0]
                      for canonical in VOICE_ALIASES.values()}

# Voice list grouped by natural language, for `--list` output.
VOICES_BY_LANG = {
    "a": ["heart", "bella", "nicole", "adam", "michael"],
    "b": ["emma", "isabella", "george", "lewis"],
    "j": ["alpha", "gongitsune", "kumo"],
    "z": ["xiaobei", "yunjian"],
}

LANG_NAMES = {
    "a": "American English",
    "b": "British English",
    "j": "Japanese",
    "z": "Mandarin",
    # Codes accepted but not currently shipped with voices upstream:
    "h": "Hindi", "e": "Spanish", "f": "French",
    "i": "Italian", "p": "Portuguese",
}

# All tokens the CLI should recognize as a positional voice argument —
# bare names and canonical IDs both count.
ALL_VOICE_TOKENS = frozenset(VOICE_ALIASES.keys()) | frozenset(VOICE_ALIASES.values())


def resolve_voice(token: str) -> str:
    """Return the canonical upstream voice ID for a user-supplied token.

    Accepts both bare names (``george``) and canonical IDs (``bm_george``).
    Unknown tokens are returned unchanged — kokoro itself will error on
    them, which is more informative than us pre-rejecting.
    """
    return VOICE_ALIASES.get(token, token)


def natural_lang(canonical_voice: str) -> str | None:
    """Natural language code for a canonical voice ID, or None if unknown."""
    return VOICE_NATURAL_LANG.get(canonical_voice)


def is_voice_token(token: str) -> bool:
    """Is `token` a valid kokoro voice in either short or canonical form?"""
    return token in ALL_VOICE_TOKENS


# Chunk-size bands for the phoneme stream — kokoro-specific. The shared
# table in chunking encodes KITTEN's cost model (marginal RTF ~0.09);
# kokoro on a desktop CPU measures render_s = 0.032 + 0.334 × audio_s at
# ~10-13 ph chars per audio second (2026-07-31, whole-utterance
# register), which changes both calibrations:
#
#   * At this speed the playback GATE dominates time-to-first-audio, not
#     the chunk-0 render. The fast ramp is the K1-4b/5 "adaptive" shape,
#     picked by simulating the real gate over the lab passages at worst
#     measured (mrtf 0.40, cps 15.77) and then MEASURED end-to-end
#     (quiet box, EMA pinned): P3 0.70 · P4 0.50 · P8 0.74 · P10 0.33s
#     TTFA, vs ~7-9s pre-K1 whole-render. Small chunk-0 target (44):
#     a chunk 0 that stays a whole short sentence — or an eager
#     pause-mark head (chunking.eager_head_cut) — renders solo and
#     banks pause-rich audio, the P10 mechanism. The dip at chunk 1
#     (40) sits where banked slack is thinnest; the 150 cap bounds late
#     deadlines (renders are sequential — one oversized late chunk can
#     stall the whole gate). The long_start column (56 head) covers a
#     LONG opening sentence with no eager mark: a mid-sentence chunk 0
#     banks wall-to-wall speech, no pause, so the gate needs a bigger
#     opening buffer (44 measured ~1.4s of extra gate hold there).
#     Whole sentences within 1.35× of target stay whole
#     (chunking.WHOLE_TOL) — sentence ends are free unconditioned
#     boundaries with the approved seam sound.
#   * The kitten table's 0.35 "slow" edge would classify this desktop as
#     slow and degrade conditioning to 2+1 — sanctioned (R14-1) ONLY for
#     a device that can't hold streaming. Kokoro at 0.36 holds fine
#     (sustained effective RTF ≈ 0.54); the slow band here starts where
#     streaming genuinely struggles.
_ADAPT_RAMP = (44, 40, 46, 52, 59, 67, 77, 88, 101, 116, 133, 150)
STREAM_BANDS = (
    ("fast", 0.50, _ADAPT_RAMP,
     chunking.CONTEXT_UNITS, chunking.LOOKAHEAD_UNITS,
     (56,) + _ADAPT_RAMP[1:]),
    ("slow", float("inf"), (90, 180, 320, 400),
     chunking.CONTEXT_UNITS, chunking.SLOW_LOOKAHEAD_UNITS),
)


class KokoroEngine(Engine):
    name = "kokoro"
    MAX_CHARS = 500
    SUPPORTS_LANG = True  # single-letter misaki codes (a/b/j/z)
    # A sentence mark inside closing quotes ends a run (K1-5, Max's
    # verdict 2026-08-01): '!"' is a real sentence end — merging it made
    # the plan condition across a boundary the model pauses at. Kokoro
    # opts in; kitten's approved renders keep the legacy merge.
    QUOTE_END_RUNS = True
    # Chunk 0 may cut at a strong pause mark (em dash / pre-quote comma)
    # for a P10-class start — see chunking.eager_head_cut (K1-5).
    EAGER_HEAD = True
    STREAM_BANDS = STREAM_BANDS

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.voice = cfg.get("voice", "heart")
        # Note: lang is intentionally not defaulted here. When None, the
        # voice's natural language is used. Set explicitly in config or
        # via --lang to force a specific pronunciation.
        self.lang = cfg.get("lang")
        self.device = cfg.get("device", "cpu")
        self.use_daemon = cfg.get("daemon", False)
        # Phoneme-direct planning (chunking.ph_stream_plan), daemon only —
        # the daemon holds the misaki G2P and the acoustic-cut synthesis.
        self.PHONEME_STREAM = self.use_daemon
        # Style rows come from PHONEME lengths: kokoro's native rule is
        # pack[len(ps)-1] over the whole segment (it does not sentence-
        # split), so the utterance-wide row is stock-faithful here.
        # Whether per-sentence rows sound better is the open K1 lab
        # question — flip to "ph-sentence" if Max's ear picks them.
        self.STYLE_ROWS = "ph-utterance"

    def _resolve_lang(self, canonical_voice: str, cli_lang: str | None) -> str:
        """Apply the language-precedence rule.

        --lang flag > config engines.kokoro.lang > voice's natural language
        > fallback "a".

        "auto" is resolved at the utterance boundary (langdetect.
        resolve_auto_lang); seeing it here means a caller bypassed that, so
        it degrades to the no-lang precedence rather than reaching the
        model.
        """
        if cli_lang and cli_lang != "auto":
            return cli_lang
        if self.lang and self.lang != "auto":
            return self.lang
        return natural_lang(canonical_voice) or "a"

    def synthesize(self, text: str, out_path: str, voice: str = None,
                   speed: float = 1.0, lang: str = None, **kwargs):
        v = resolve_voice(voice or self.voice)
        la = self._resolve_lang(v, lang)

        if self.use_daemon:
            request = {"text": text, "voice": v, "speed": speed,
                       "lang": la, "out": out_path}
            dmgr.synthesize("kokoro", request, auto_start=True)
            return

        # Subprocess fallback
        env_extra = {"HF_HUB_OFFLINE": "1"}
        if self.device == "cpu":
            env_extra["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [KOKORO_BIN, "--voice", v, "--output-file", out_path, "--text", text]
        if la:
            cmd += ["--language", la]
        if speed and speed != 1.0:
            cmd += ["--speed", str(speed)]

        run_in_venv(KOKORO_BIN, cmd, env_extra=env_extra, engine_name="kokoro")

    def phonemize(self, text: str, voice: str = None, lang: str = None,
                  **kwargs) -> str:
        """One misaki G2P pass over the whole utterance, via the daemon.
        Phase 1 of the phoneme-direct streaming path."""
        import tempfile
        v = resolve_voice(voice or self.voice)
        fd, tmp = tempfile.mkstemp(prefix="marmalade-ph-", suffix=".txt")
        os.close(fd)
        try:
            dmgr.synthesize("kokoro", {
                "op": "phonemize", "text": text,
                "lang": self._resolve_lang(v, lang), "out": tmp},
                auto_start=True)
            with open(tmp, encoding="utf-8") as f:
                return f.read().strip()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def synthesize_phonemes(self, ph_text: str, out_path: str,
                            voice: str = None, speed: float = 1.0,
                            lang: str = None, context: str = None,
                            lookahead: str = None, style_ref: int = None,
                            pad_marks: dict = None, **kwargs):
        """Synthesize from phonemes with acoustic conditioning cuts.

        ``style_ref`` pins the style-pack row (510 rows; kokoro indexes by
        phoneme length natively), ``pad_marks`` floors a mark's rendered
        pause with inserted silence."""
        v = resolve_voice(voice or self.voice)
        request = {"ph_text": ph_text, "voice": v, "speed": speed,
                   "lang": self._resolve_lang(v, lang), "out": out_path}
        if context:
            request["ph_context"] = context
        if lookahead:
            request["ph_lookahead"] = lookahead
        if style_ref is not None:
            request["style_ref"] = int(style_ref)
        if pad_marks:
            request["pad_marks"] = pad_marks
        dmgr.synthesize("kokoro", request, auto_start=True)

    def list_voices(self):
        print("Kokoro voices (use the bare name, e.g. \"george\"):\n")
        for lang_code, names in VOICES_BY_LANG.items():
            label = LANG_NAMES.get(lang_code, lang_code)
            print(f"  {label} ({lang_code}):")
            print(f"    {', '.join(names)}")
        print()
        print("Each voice uses its natural language by default. Override with")
        print("--lang or set engines.kokoro.lang in your config. The canonical")
        print("upstream form (e.g. \"bm_george\") is also accepted everywhere.")
