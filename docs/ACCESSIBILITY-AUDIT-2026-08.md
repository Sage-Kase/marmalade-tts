# Accessibility Audit — marmalade-tts CLI (2026-08-01)

Audience: blind and low-vision users driving a terminal with a screen reader
(Orca on Linux, brltty braille displays). Read-only audit; findings verified
in code. Companion audit for the Android app lives in
`marmalade-tts-android/docs/ACCESSIBILITY-AUDIT-2026-08.md`.

## Verdict

**No blockers.** The default synthesis path (`marmalade-tts "text"`) is fully
screen-reader usable end to end. Output is overwhelmingly plain, append-only,
line-oriented text; status goes to stderr; `--quiet`, `--json`, `--print-path`,
`--stdin` exist; no spinners or `\r`-rewriting progress bars in first-party
code; symbols (✓/✗) are always accompanied by words (PASS/FAIL).

The one real problem area is the `init` wizard: a raw-mode arrow-key TUI with
in-place line erasure — the single worst pattern for Orca/brltty — though the
`--non-interactive` escape hatch covers everything the TUI does.

## Findings

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Major | `marmalade_tts/init.py:132-249` | Raw-mode arrow-key TUI with in-place line erasure (`\033[A\033[2K` redraw per keypress); selection state symbol-only (`●`/`○`). No plain-prompt mode in a TTY; the `--non-interactive` escape hatch is not advertised in the wizard's opening text. Non-TTY stdin does auto-fall-back (`cli.py:218`). |
| 2 | Major | `marmalade_tts/cli.py:805` | `--json` silently ignored for `--list` (also `--list-effects` / `--list-aliases`). Structured voice data already exists (`mcp_server.py:115 list_voices_data`) but isn't reachable from the CLI. |
| 3 | Minor | `marmalade_tts/init.py:163,180,186,224,335,368` | ANSI styling (bold/dim/erase) with no `NO_COLOR` / `isatty` gate. Color is never the sole carrier of meaning anywhere, so minor. |
| 4 | Minor | `marmalade_tts/installer.py:174,342,515` | Third-party `\r` progress bars bleed through (pip, gdown, HF warm-cache downloads run with inherited stdout). Fix: `-q` flags / `HF_HUB_DISABLE_PROGRESS_BARS=1` when non-TTY or `--plain`. |
| 5 | Minor | `marmalade_tts/cli_helpers.py:223,240` | `×` (speed) and `→` (alias listing) as sole separators — some screen readers drop or garble them. Use plain words/`x`. |
| 6 | Minor | `marmalade_tts/init.py:142` | Bare ESC hangs the wizard: after `\x1b`, `sys.stdin.read(2)` blocks in raw mode until two more bytes arrive. Needs timeout/non-blocking read of the CSI tail. |
| 7 | Minor | `marmalade_tts/playback.py:34` | "No audio player found" notice printed to **stdout**, corrupting `--json` / `--print-path` machine-readable streams (JSON is printed before playback, `cli.py:982-988`). Add `file=sys.stderr`. |
| 8 | Minor | `marmalade_tts/cli.py:526` epilog | Subcommands (`daemon` / `config` / `init` / `install` / `uninstall` / `mcp`) are dispatched by intercept before argparse and never mentioned in `--help` — undiscoverable without sighted exploration of docs. Add a "Subcommands" paragraph to the epilog. |
| 9 | Info | (absent) | No speech-dispatcher/SSIP integration — see below. |

## What's already good

- Errors always textually marked (`FAILED —`, `Warning:`) and sent to stderr
  with prefixes; exit codes documented (`scripts/SCRIPTS.md:78-80`).
- Installer output is pure append-only `[install] name: step` lines — ideal.
- All listings are linear prose / `key: value` — no box-drawing characters.
- `--help` is genuinely complete for the main command: real prose per flag,
  16 worked examples, voice-syntax explained inline; errors suggest the
  recovery command (`cli_helpers.py:114-117`). Shell completion available.
- Every non-TUI prompt is plain `input()` y/n with an explicit default;
  `uninstall` honors `-y` and requires typed flags when non-TTY.

## The CLI as an assistive tool (speech-dispatcher scoping)

Already shipped and relevant to blind/low-vision users:

- `scripts/speak-selection` + `scripts/speak-clipboard` — speak-highlighted-
  text hotkey workflows with KDE binding instructions (`scripts/SCRIPTS.md`).
- Daemon mode: per-engine Unix socket, newline-delimited JSON protocol
  (`daemon.py:337-357`), systemd units. Kitten daemon's PHONEME_STREAM gives
  the low-TTFA latency profile a screen-reader backend needs.
- MCP server with structured `list_voices`.

Missing: any speech-dispatcher (Orca's speech backend) integration.

- **Cheap path:** speech-dispatcher's `sd_generic` module runs a shell command
  per utterance — `marmalade-tts --quiet "$DATA"` works today; the daemon
  keeps it fast. A sample `generic.conf` stanza in docs costs a page.
  Inherent limits: no index marks; stop = kill of the player child (works —
  `play_wav` is a killable child, `playback.py:32`).
- **Real path:** a native spd output module speaking the existing Unix-socket
  JSON protocol. Gaps: (a) rate/pitch/volume mapping (`--speed` exists; pitch
  only via sox effects) — mapping work; (b) **immediate cancel** — daemon
  synthesis is not cancellable mid-request; this is the only structural gap;
  (c) SSML/punctuation handling — mapping work.
- The README says nothing to this audience; speak-selection + daemon latency
  are already a real low-vision offering worth a docs section.

## Suggested priority order

1. `--json` for `--list` variants (#2) — unlocks scripted/assistive wrappers.
2. Plain numbered-prompt fallback for `init` + advertise `--non-interactive`
   in the wizard opening text (#1); fix the bare-ESC hang while in there (#6).
3. stdout/stderr fix in playback (#7) — one line.
4. sd_generic docs stanza + README accessibility section (#9).
5. The remaining minors (#3, #4, #5, #8) are each small and mechanical.
