"""Tests for marmalade-tts init — non-interactive + interactive TUI paths."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock, call
import yaml

from marmalade_tts.cli import main
from marmalade_tts.init import (
    init_non_interactive, ENGINE_INFO, ENGINE_ORDER, _is_tty,
)


@pytest.fixture(autouse=True)
def _mock_installer():
    """`init` now installs the selected engines via installer.install_engines.
    Stub it out everywhere so init tests exercise config writing, not real
    venv creation / downloads. The installer itself is covered by
    test_installer.py."""
    with patch("marmalade_tts.installer.install_engines", return_value=[]) as m:
        yield m


# ── Non-interactive path ─────────────────────────────────────────────────────

class TestNonInteractive:
    def test_single_engine(self):
        result = init_non_interactive(["kitten"])
        assert "kitten" in result
        assert result["kitten"]["model_size"] == "nano"  # default
        assert result["kitten"]["daemon"] is False

    def test_multiple_engines(self):
        result = init_non_interactive(["kitten", "piper", "kokoro"])
        assert len(result) == 3
        # Kokoro voice default is now the bare name, not the canonical ID.
        assert result["kokoro"]["voice"] == "heart"
        # No lang default — voice's natural language is used at synth time.
        assert "lang" not in result["kokoro"]

    def test_override_model_size(self):
        result = init_non_interactive(
            ["kitten"],
            engine_options={"kitten": {"model_size": "nano"}}
        )
        assert result["kitten"]["model_size"] == "nano"

    def test_override_kokoro_voice(self):
        # Bare name (preferred form)
        result = init_non_interactive(
            ["kokoro"],
            engine_options={"kokoro": {"voice": "adam"}}
        )
        assert result["kokoro"]["voice"] == "adam"

    def test_override_kokoro_voice_canonical_form(self):
        # Canonical form also still accepted (back-compat)
        result = init_non_interactive(
            ["kokoro"],
            engine_options={"kokoro": {"voice": "am_adam"}}
        )
        assert result["kokoro"]["voice"] == "am_adam"

    def test_invalid_engine_exits(self):
        with pytest.raises(SystemExit):
            init_non_interactive(["nonexistent_engine"])

    def test_invalid_choice_exits(self):
        with pytest.raises(SystemExit):
            init_non_interactive(
                ["kitten"],
                engine_options={"kitten": {"model_size": "gigantic"}}
            )

    def test_all_engines(self):
        result = init_non_interactive(ENGINE_ORDER)
        assert len(result) == len(ENGINE_ORDER)
        for eng in ENGINE_ORDER:
            assert eng in result
            assert "daemon" in result[eng]
            assert "device" in result[eng]

    def test_matcha_default_model(self):
        result = init_non_interactive(["matcha"])
        assert result["matcha"]["model"] == "matcha_ljspeech"

    def test_emojivoice_default_voice(self):
        result = init_non_interactive(["emojivoice"])
        assert result["emojivoice"]["voice"] == "paige"

    def test_piper_has_empty_model_default(self):
        result = init_non_interactive(["piper"])
        assert result["piper"]["model"] == ""

    def test_coqui_has_empty_model_default(self):
        result = init_non_interactive(["coqui"])
        assert result["coqui"]["model"] == ""


# ── CLI non-interactive integration ──────────────────────────────────────────

class TestCLIInitNonInteractive:
    def test_basic_init_writes_config(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["defaults"]["engine"] == "kitten"
        assert "kitten" in cfg["engines"]

    def test_set_override_via_cli(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten",
                                "--set", "kitten.model_size=nano"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["engines"]["kitten"]["model_size"] == "nano"

    def test_multiple_engines_via_cli(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten,kokoro",
                                "--default-engine", "kokoro"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["defaults"]["engine"] == "kokoro"
        assert "kitten" in cfg["engines"]
        assert "kokoro" in cfg["engines"]

    def test_missing_engines_flag_exits(self):
        with patch("sys.argv", ["marmalade-tts", "init", "--non-interactive"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code != 0

    def test_bad_set_format_exits(self):
        with patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten",
                                "--set", "bad_format"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code != 0

    def test_test_flag_attempts_synthesis(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        mock_eng = MagicMock()
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten", "--test"]), \
             patch("marmalade_tts.init._is_tty", return_value=False), \
             patch("marmalade_tts.cli.ENGINE_CLASSES",
                   {"kitten": lambda cfg: mock_eng}), \
             patch("marmalade_tts.cli.play_wav"), \
             patch("marmalade_tts.cli.make_tmp_wav", return_value="/tmp/test.wav"), \
             patch("marmalade_tts.cli.os.unlink"):
            main()
        # The --test flag should drive synthesis on the configured engine.
        mock_eng.synthesize.assert_called_once()


# ── Engine metadata ──────────────────────────────────────────────────────────

class TestEngineMetadata:
    def test_all_engines_have_info(self):
        for eng in ENGINE_ORDER:
            assert eng in ENGINE_INFO

    def test_info_has_required_keys(self):
        for eng, info in ENGINE_INFO.items():
            assert "label" in info
            assert "desc" in info
            assert "size" in info
            assert "default" in info
            assert "options" in info

    def test_defaults_are_kitten_and_piper(self):
        defaults = [eng for eng, info in ENGINE_INFO.items() if info["default"]]
        assert "kitten" in defaults
        assert "piper" in defaults

    def test_kitten_has_model_size_option(self):
        opts = ENGINE_INFO["kitten"]["options"]
        assert "model_size" in opts
        assert "micro" in opts["model_size"]["choices"]
        assert opts["model_size"]["default"] == "nano"

    def test_kokoro_has_voice_option(self):
        opts = ENGINE_INFO["kokoro"]["options"]
        assert "voice" in opts
        # Choices are bare names; default is "heart".
        assert "heart" in opts["voice"]["choices"]
        assert opts["voice"]["default"] == "heart"

    def test_engine_order_matches_info(self):
        for eng in ENGINE_ORDER:
            assert eng in ENGINE_INFO


# ── TUI helpers (unit tests, no real terminal) ───────────────────────────────

class TestTUIHelpers:
    def test_is_tty_false_in_pipe(self):
        """In test/CI, stdin is not a TTY."""
        # Could be either, but must not crash
        result = _is_tty()
        assert isinstance(result, bool)

    def test_non_interactive_falls_back_when_no_tty(self, tmp_path):
        """When stdin is not a TTY and --engines is given, should succeed."""
        cfg_path = str(tmp_path / "config.yaml")
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init", "--engines", "kitten"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "kitten" in cfg["engines"]


# ── Plain-prompt (screen-reader) path ────────────────────────────────────────

class TestPlainMode:
    def test_plain_mode_from_env(self, monkeypatch):
        from marmalade_tts.init import plain_mode
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert plain_mode() is False
        monkeypatch.setenv("TERM", "dumb")
        assert plain_mode() is True
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("NO_COLOR", "1")
        assert plain_mode() is True

    def test_color_gated_by_no_color(self, monkeypatch):
        from marmalade_tts.init import _bold, _dim
        monkeypatch.setenv("NO_COLOR", "1")
        assert _bold("hi") == "hi"
        assert _dim("hi") == "hi"

    def test_no_ansi_when_stdout_not_a_tty(self, monkeypatch):
        """capsys' stdout isn't a TTY, so styling must be plain text."""
        from marmalade_tts.init import _bold
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert "\033" not in _bold("hi")

    def test_parse_numbers(self):
        from marmalade_tts.init import _parse_numbers
        assert _parse_numbers("1,3", 5) == [1, 3]
        assert _parse_numbers("2 4", 5) == [2, 4]
        assert _parse_numbers("0", 5) is None
        assert _parse_numbers("6", 5) is None
        assert _parse_numbers("x", 5) is None
        assert _parse_numbers(",", 5) is None

    def test_multi_select_plain_defaults_on_empty_input(self, capsys):
        from marmalade_tts.init import _multi_select
        items = [("a", "A", ""), ("b", "B", ""), ("c", "C", "")]
        with patch("builtins.input", return_value=""):
            got = _multi_select(items, defaults={"a", "c"}, title="T", plain=True)
        assert got == ["a", "c"]
        out = capsys.readouterr().out
        assert "1) A" in out and "\033" not in out

    def test_multi_select_plain_picks_numbers(self):
        from marmalade_tts.init import _multi_select
        items = [("a", "A", ""), ("b", "B", ""), ("c", "C", "")]
        with patch("builtins.input", return_value="2, 3"):
            got = _multi_select(items, defaults={"a"}, title="T", plain=True)
        assert got == ["b", "c"]

    def test_multi_select_plain_reprompts_on_bad_input(self):
        from marmalade_tts.init import _multi_select
        items = [("a", "A", ""), ("b", "B", "")]
        with patch("builtins.input", side_effect=["9", "1"]):
            got = _multi_select(items, defaults=set(), title="T", plain=True)
        assert got == ["a"]

    def test_single_select_plain_default_and_pick(self):
        from marmalade_tts.init import _single_select
        with patch("builtins.input", return_value=""):
            assert _single_select(["x", "y", "z"], default="y", prompt="P",
                                  plain=True) == "y"
        with patch("builtins.input", return_value="3"):
            assert _single_select(["x", "y", "z"], default="y", prompt="P",
                                  plain=True) == "z"

    def test_single_select_plain_cancels_on_eof(self):
        from marmalade_tts.init import _single_select
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit) as exc:
                _single_select(["x", "y"], default="x", prompt="P", plain=True)
        assert exc.value.code == 0

    def test_cli_plain_flag_reaches_wizard(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init", "--plain"]), \
             patch("marmalade_tts.init._is_tty", return_value=True), \
             patch("marmalade_tts.init.init_interactive",
                   return_value=(["kitten"], {"kitten": {}}, "kitten")) as wiz, \
             patch("marmalade_tts.init._ask_yn", return_value=False):
            main()
        assert wiz.call_args.kwargs["plain"] is True


# ── ESC handling ─────────────────────────────────────────────────────────────

class TestEscapeTail:
    def test_bare_esc_does_not_block(self):
        """No bytes follow the ESC — the read must time out, not hang."""
        import os as _os
        from marmalade_tts.init import _read_escape_tail
        r, w = _os.pipe()
        try:
            assert _read_escape_tail(r, timeout=0.01) == ""
        finally:
            _os.close(r)
            _os.close(w)

    def test_arrow_tail_is_read(self):
        import os as _os
        from marmalade_tts.init import _read_escape_tail
        r, w = _os.pipe()
        try:
            _os.write(w, b"[A")
            assert _read_escape_tail(r, timeout=0.5) == "[A"
        finally:
            _os.close(r)
            _os.close(w)

    def test_esc_cancels_the_menu(self, capsys):
        """A bare ESC exits cleanly instead of being swallowed by the loop."""
        from marmalade_tts.init import _multi_select
        with patch("marmalade_tts.init._read_key", return_value="ESC"), \
             patch("marmalade_tts.init._clear_lines"):
            with pytest.raises(SystemExit) as exc:
                _multi_select([("a", "A", "")], defaults={"a"}, title="T")
        assert exc.value.code == 0
        assert "Cancelled" in capsys.readouterr().out

    def test_partial_tail_gives_up(self):
        import os as _os
        from marmalade_tts.init import _read_escape_tail
        r, w = _os.pipe()
        try:
            _os.write(w, b"[")
            assert _read_escape_tail(r, timeout=0.01) == "["
        finally:
            _os.close(r)
            _os.close(w)


# ── Config preservation ──────────────────────────────────────────────────────

class TestConfigPreservation:
    def test_init_preserves_existing_config_keys(self, tmp_path):
        """Init should merge, not overwrite existing config."""
        cfg_path = str(tmp_path / "config.yaml")
        existing = {
            "defaults": {"speed": 1.5, "play": False},
            "effects": {"defaults": {"kitten": ["reverb=20"]}},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(existing, f)

        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Original values preserved
        assert cfg["defaults"]["speed"] == 1.5
        # New values added
        assert cfg["defaults"]["engine"] == "kitten"
        # Effects untouched
        assert cfg["effects"]["defaults"]["kitten"] == ["reverb=20"]

    def test_init_overwrites_engine_config(self, tmp_path):
        """Re-running init for an engine should update its config."""
        cfg_path = str(tmp_path / "config.yaml")
        existing = {
            "defaults": {"engine": "kitten"},
            "engines": {"kitten": {"model_size": "nano", "daemon": True}},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(existing, f)

        with patch("marmalade_tts.config.CONFIG_PATH", cfg_path), \
             patch("sys.argv", ["marmalade-tts", "init",
                                "--non-interactive", "--engines", "kitten",
                                "--set", "kitten.model_size=micro"]), \
             patch("marmalade_tts.init._is_tty", return_value=False):
            main()

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["engines"]["kitten"]["model_size"] == "micro"


# ── Pocket TTS engine ────────────────────────────────────────────────────────

class TestPocketInit:
    def test_pocket_default_voice(self):
        result = init_non_interactive(["pocket"])
        assert result["pocket"]["voice"] == "alba"

    def test_pocket_override_voice(self):
        result = init_non_interactive(
            ["pocket"],
            engine_options={"pocket": {"voice": "marius"}}
        )
        assert result["pocket"]["voice"] == "marius"

    def test_pocket_invalid_voice_exits(self):
        with pytest.raises(SystemExit):
            init_non_interactive(
                ["pocket"],
                engine_options={"pocket": {"voice": "nonexistent"}}
            )

    def test_pocket_in_engine_order(self):
        assert "pocket" in ENGINE_ORDER

    def test_pocket_in_engine_info(self):
        assert "pocket" in ENGINE_INFO
        assert ENGINE_INFO["pocket"]["label"] == "Pocket TTS"
