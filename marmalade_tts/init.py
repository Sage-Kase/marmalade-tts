"""Interactive and non-interactive init wizard for marmalade-tts."""

import os
import select
import sys

from .engines.api import VOICES as _API_VOICE_CHOICES
from .engines.kokoro import (
    VOICES_BY_LANG as _KOKORO_VOICES_BY_LANG,
    is_voice_token as _kokoro_is_voice_token,
)

# Flatten kokoro voice list into a single ordered list of bare names.
_KOKORO_VOICE_CHOICES = [v for voices in _KOKORO_VOICES_BY_LANG.values() for v in voices]

# Engine metadata used by both the TUI and non-interactive paths.
ENGINE_INFO = {
    "kitten": {
        "label": "Kitten TTS",
        "desc":  "Fast, lightweight, great quality. Ships by default.",
        "size":  "~40–80 MB (nano/micro/mini)",
        "default": True,
        "options": {
            "model_size": {
                "prompt": "Model size",
                "choices": ["nano", "micro", "mini"],
                "default": "nano",
                "help": "nano (fp32, fastest AND best quality)  micro/mini (int8-only upstream: slower, audibly worse)",
            },
        },
    },
    "piper": {
        "label": "Piper",
        "desc":  "Very fast ONNX engine. Many community voices available.",
        "size":  "~15–75 MB per voice model",
        "default": True,
        "options": {},
    },
    "kokoro": {
        "label": "Kokoro",
        "desc":  "High quality, multilingual. Needs ~500 MB + optional GPU.",
        "size":  "~500 MB",
        "default": False,
        "options": {
            "voice": {
                "prompt": "Default voice",
                "choices": _KOKORO_VOICE_CHOICES,
                "default": "heart",
                # Custom validator: accepts bare names AND canonical IDs
                # (e.g. both "george" and "bm_george"). Used by the
                # non-interactive path; the interactive picker still
                # shows only the bare names from `choices`.
                "validate": _kokoro_is_voice_token,
                "help": ("Voices grouped by natural language: American (heart, bella, "
                         "nicole, adam, michael), British (emma, isabella, george, "
                         "lewis), Japanese (alpha, gongitsune, kumo), Mandarin "
                         "(xiaobei, yunjian). Each voice defaults to its natural "
                         "language."),
            },
        },
    },
    "coqui": {
        "label": "Coqui TTS",
        "desc":  "Research-grade, many models. Largest download, slowest startup.",
        "size":  "~200 MB – 2 GB depending on model",
        "default": False,
        "options": {},
    },
    "pocket": {
        "label": "Pocket TTS",
        "desc":  "CPU-only, 100M params, ~200ms latency, voice cloning. English only.",
        "size":  "~200 MB (model auto-downloads from HuggingFace)",
        "default": False,
        "options": {
            "voice": {
                "prompt": "Default voice",
                "choices": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"],
                "default": "alba",
                "help": "Built-in voices. You can also clone any voice from a .wav file.",
            },
        },
    },
    "matcha": {
        "label": "Matcha-TTS",
        "desc":  "Fast flow-matching neural TTS. Clear, natural English. Needs espeak-ng.",
        "size":  "~73 MB model + ~50 MB vocoder (auto-download on first use)",
        "default": False,
        "options": {},
    },
    "emojivoice": {
        "label": "EmojiVoice",
        "desc":  "Emoji-controlled expressive TTS — 🤣😭😡 in the text set the emotion. English.",
        "size":  "~78 MB speaker checkpoint (manual download — see INSTALL.md)",
        "default": False,
        "options": {
            "voice": {
                "prompt": "Speaker",
                "choices": ["paige"],
                "default": "paige",
                "help": "paige — the verified EmojiVoice speaker checkpoint.",
            },
        },
    },
    "api": {
        "label": "API TTS",
        "desc":  "Hosted OpenAI-compatible TTS (Venice by default). Needs an API key + network.",
        "size":  "nothing to download",
        "default": False,
        "options": {
            "voice": {
                "prompt": "Default voice",
                "choices": _API_VOICE_CHOICES,
                "default": "af_heart",
                # Voices are provider/model-dependent (choices only lists
                # Venice's tts-kokoro set) — accept anything non-interactively.
                "validate": lambda v: bool(v),
                "help": ("Venice tts-kokoro voice IDs shown; other models/providers "
                         "have their own — run `marmalade-tts api --list`. "
                         "Key: `marmalade secret set venice/api-key` (the keyring "
                         "default), or set VENICE_API_KEY before use."),
            },
        },
    },
}

ENGINE_ORDER = ["kitten", "piper", "kokoro", "coqui", "pocket", "matcha", "emojivoice", "api"]

# ── TUI helpers (stdlib only) ────────────────────────────────────────────────

def _is_tty():
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _color_enabled():
    """ANSI styling is only emitted to a colour-capable stdout TTY."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bold(text):
    return f"\033[1m{text}\033[0m" if _color_enabled() else text


def _dim(text):
    return f"\033[2m{text}\033[0m" if _color_enabled() else text


def plain_mode():
    """True when the arrow-key TUI should be replaced by numbered prompts.

    A raw-mode menu that redraws itself in place is unreadable to a screen
    reader, so NO_COLOR and TERM=dumb opt out of it (as does `init --plain`,
    which the caller ORs in).
    """
    if os.environ.get("TERM", "") == "dumb":
        return True
    return bool(os.environ.get("NO_COLOR"))


def _read_escape_tail(fd, timeout=0.05):
    """Read the CSI tail after an ESC byte without blocking indefinitely.

    A bare ESC has no tail, so a plain two-byte read hangs the wizard until
    the user presses two more keys. Poll instead and give up on silence.
    """
    tail = ""
    while len(tail) < 2:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            break
        chunk = os.read(fd, 2 - len(tail))
        if not chunk:
            break
        tail += chunk.decode("utf-8", "replace")
    return tail


def _read_key():
    """Read a single keypress (Unix). Returns special tokens for arrows.

    Reads at the file-descriptor level so ``_read_escape_tail`` can poll with
    select() — sys.stdin's own buffering would hide bytes from it.
    """
    import tty
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode("utf-8", "replace")
        if ch == "\x1b":
            seq = _read_escape_tail(fd)
            if seq == "[A":
                return "UP"
            if seq == "[B":
                return "DOWN"
            return "ESC"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == " ":
            return "SPACE"
        if ch in ("q", "Q", "\x03"):  # Ctrl-C
            return "QUIT"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear_lines(n):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()


def _cancel():
    print("Cancelled.")
    sys.exit(0)


def _parse_numbers(resp, count):
    """Parse "1,3 4" into 1-based indices. None if anything is out of range."""
    out = []
    for tok in resp.replace(",", " ").split():
        if not tok.isdigit() or not 1 <= int(tok) <= count:
            return None
        out.append(int(tok))
    return out or None


def _multi_select_plain(items, defaults, title):
    """Numbered multi-select on plain input(). Same choices as the TUI."""
    print(f"{title}  (type numbers separated by commas, ENTER accepts the defaults)")
    print()
    for i, (key, label, desc) in enumerate(items, 1):
        mark = "selected by default" if key in defaults else "not selected"
        line = f"  {i}) {label} [{mark}]"
        if desc:
            line += f" — {desc}"
        print(line)
    print()
    default_nums = ", ".join(str(i) for i, (k, _, _) in enumerate(items, 1)
                             if k in defaults)
    while True:
        try:
            resp = input(f"  Numbers [{default_nums}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _cancel()
        if not resp:
            return [k for k, _, _ in items if k in defaults]
        picked = _parse_numbers(resp, len(items))
        if picked is None:
            print(f"  Enter numbers between 1 and {len(items)}, separated by commas.")
            continue
        chosen = {items[i - 1][0] for i in picked}
        return [k for k, _, _ in items if k in chosen]


def _single_select_plain(choices, default, prompt):
    """Numbered single select on plain input(). Same choices as the TUI."""
    print(f"{prompt}  (type a number, ENTER accepts the default)")
    print()
    for i, c in enumerate(choices, 1):
        dflt = " (default)" if c == default else ""
        print(f"  {i}) {c}{dflt}")
    print()
    default_num = choices.index(default) + 1 if default in choices else 1
    while True:
        try:
            resp = input(f"  Number [{default_num}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _cancel()
        if not resp:
            return choices[default_num - 1]
        if resp.isdigit() and 1 <= int(resp) <= len(choices):
            return choices[int(resp) - 1]
        print(f"  Enter a number between 1 and {len(choices)}.")


def _multi_select(items, defaults=None, title="Select engines", plain=False):
    """Arrow-key multi-select. Returns list of selected item keys.

    items:    list of (key, label, description)
    defaults: set of keys that start checked
    plain:    use numbered input() prompts instead of the raw-mode TUI
    """
    if defaults is None:
        defaults = set()

    if plain:
        return _multi_select_plain(items, defaults, title)

    selected = {k for k, _, _ in items if k in defaults}
    cursor = 0

    def render():
        print(f"{_bold(title)}  (↑↓ move, SPACE toggle, ENTER confirm)\n")
        for i, (key, label, desc) in enumerate(items):
            marker = "▸" if i == cursor else " "
            check = "●" if key in selected else "○"
            line = f"  {marker} {check} {label}"
            if desc:
                line += "  " + _dim(f"— {desc}")
            print(line)
        print()

    render()

    while True:
        k = _read_key()
        lines_to_clear = len(items) + 3  # title + blank + items + trailing blank
        _clear_lines(lines_to_clear)

        if k == "UP":
            cursor = (cursor - 1) % len(items)
        elif k == "DOWN":
            cursor = (cursor + 1) % len(items)
        elif k == "SPACE":
            key = items[cursor][0]
            if key in selected:
                selected.discard(key)
            else:
                selected.add(key)
        elif k == "ENTER":
            render()
            return [k for k, _, _ in items if k in selected]
        elif k in ("QUIT", "ESC"):
            _cancel()

        render()


def _single_select(choices, default=None, prompt="Choose", plain=False):
    """Arrow-key single select. Returns the chosen value.

    ``plain`` swaps the TUI for a numbered input() prompt."""
    if plain:
        return _single_select_plain(choices, default, prompt)

    cursor = 0
    if default and default in choices:
        cursor = choices.index(default)

    def render():
        print(f"{_bold(prompt)}  (↑↓ move, ENTER select)\n")
        for i, c in enumerate(choices):
            marker = "▸" if i == cursor else " "
            dflt = " (default)" if c == default else ""
            print(f"  {marker} {c}{dflt}")
        print()

    render()

    while True:
        k = _read_key()
        lines_to_clear = len(choices) + 3
        _clear_lines(lines_to_clear)

        if k == "UP":
            cursor = (cursor - 1) % len(choices)
        elif k == "DOWN":
            cursor = (cursor + 1) % len(choices)
        elif k == "ENTER":
            render()
            return choices[cursor]
        elif k in ("QUIT", "ESC"):
            _cancel()

        render()


def _ask_yn(prompt, default=True):
    """Simple y/n prompt with default."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        resp = input(f"{prompt} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not resp:
        return default
    return resp in ("y", "yes")


# ── Non-interactive init ─────────────────────────────────────────────────────

def init_non_interactive(engines, engine_options=None):
    """Configure engines without prompts. Returns config dict updates.

    Args:
        engines:        list of engine names (e.g. ["kitten", "piper"])
        engine_options: dict of {engine: {option: value}} overrides
                        e.g. {"kitten": {"model_size": "nano"}}
    """
    if engine_options is None:
        engine_options = {}

    engines_cfg = {}
    for eng in engines:
        if eng not in ENGINE_INFO:
            print(f"[init] Unknown engine: {eng}", file=sys.stderr)
            sys.exit(1)

        info = ENGINE_INFO[eng]
        cfg = {}

        # Apply defaults, then overrides
        for opt_key, opt_meta in info["options"].items():
            value = engine_options.get(eng, {}).get(opt_key, opt_meta["default"])
            # Validate: custom validator wins; otherwise fall back to choices.
            if "validate" in opt_meta:
                if not opt_meta["validate"](value):
                    print(f"[init] Invalid {opt_key} for {eng}: {value!r}",
                          file=sys.stderr)
                    sys.exit(1)
            elif "choices" in opt_meta and value not in opt_meta["choices"]:
                print(f"[init] Invalid {opt_key} for {eng}: {value!r} "
                      f"(valid: {', '.join(opt_meta['choices'])})", file=sys.stderr)
                sys.exit(1)
            cfg[opt_key] = value

        # Engine-specific defaults
        if eng == "kitten":
            cfg.setdefault("model_size", "nano")
        elif eng == "kokoro":
            cfg.setdefault("voice", "heart")
            # Note: no 'lang' default. Voice's natural language is used unless
            # the user sets one explicitly with `config set engines.kokoro.lang`
            # or --lang on the CLI.
        elif eng == "piper":
            cfg.setdefault("model", "")
        elif eng == "coqui":
            cfg.setdefault("model", "")
        elif eng == "pocket":
            cfg.setdefault("voice", "alba")
        elif eng == "matcha":
            cfg.setdefault("model", "matcha_ljspeech")
        elif eng == "emojivoice":
            cfg.setdefault("voice", "paige")
        elif eng == "api":
            cfg.setdefault("voice", "af_heart")

        cfg.setdefault("daemon", False)
        cfg.setdefault("device", "cpu")
        engines_cfg[eng] = cfg

    return engines_cfg


# ── Interactive init (TUI) ───────────────────────────────────────────────────

def init_interactive(plain=False):
    """Run the full interactive setup wizard. Returns (selected_engines, engines_cfg, default_engine).

    ``plain`` replaces the arrow-key menus with numbered input() prompts.
    """
    print()
    print(f"  🍊 {_bold('marmalade-tts setup')}")
    print("  ─────────────────────────────")
    print()
    print("  Choose which TTS engines to install.")
    print("  Kitten ships by default and is recommended for most users.")
    print("  You can change this later with: marmalade-tts config")
    print("  Screen reader or script? --plain uses numbered prompts instead of")
    print("  the arrow-key menus, and --non-interactive --engines kitten,piper")
    print("  skips the prompts entirely.")
    print()

    # Build items for multi-select
    items = []
    defaults = set()
    for eng in ENGINE_ORDER:
        info = ENGINE_INFO[eng]
        desc = f"{info['desc']}  ({info['size']})"
        items.append((eng, info["label"], desc))
        if info["default"]:
            defaults.add(eng)

    selected = _multi_select(items, defaults=defaults, title="Engines", plain=plain)

    if not selected:
        print("No engines selected. At least one is required.")
        sys.exit(1)

    print(f"\n  ✓ Selected: {', '.join(selected)}\n")

    # Per-engine options
    engines_cfg = {}
    for eng in selected:
        info = ENGINE_INFO[eng]
        cfg = {}

        if info["options"]:
            print("  " + _bold(f"{info['label']} options:"))

        for opt_key, opt_meta in info["options"].items():
            if len(opt_meta["choices"]) > 1:
                if opt_meta.get("help"):
                    print(f"  {opt_meta['help']}")
                value = _single_select(
                    opt_meta["choices"],
                    default=opt_meta["default"],
                    prompt=opt_meta["prompt"],
                    plain=plain,
                )
            else:
                value = opt_meta["default"]
            cfg[opt_key] = value

        # Engine-specific defaults
        if eng == "kitten":
            cfg.setdefault("model_size", "nano")
        elif eng == "kokoro":
            cfg.setdefault("voice", "heart")
            # Note: no 'lang' default. Voice's natural language is used unless
            # the user sets one explicitly with `config set engines.kokoro.lang`
            # or --lang on the CLI.
        elif eng == "piper":
            cfg.setdefault("model", "")
        elif eng == "coqui":
            cfg.setdefault("model", "")
        elif eng == "pocket":
            cfg.setdefault("voice", "alba")
        elif eng == "matcha":
            cfg.setdefault("model", "matcha_ljspeech")
        elif eng == "emojivoice":
            cfg.setdefault("voice", "paige")
        elif eng == "api":
            cfg.setdefault("voice", "af_heart")

        cfg.setdefault("daemon", False)
        cfg.setdefault("device", "cpu")
        engines_cfg[eng] = cfg

    # Pick default engine
    if len(selected) == 1:
        default_engine = selected[0]
    else:
        print()
        default_engine = _single_select(
            selected,
            default=selected[0],
            prompt="Default engine (used when no engine is specified)",
            plain=plain,
        )

    print(f"\n  ✓ Default engine: {default_engine}\n")

    return selected, engines_cfg, default_engine
