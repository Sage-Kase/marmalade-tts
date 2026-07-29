"""Shared fixtures. Autouse: perfstats must never touch the real
~/.config perf.json from tests — mock engines would poison the observed
RTF averages the streaming gate relies on."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_perfstats(tmp_path, monkeypatch):
    from marmalade_tts import perfstats
    monkeypatch.setattr(perfstats, "STATS_PATH",
                        str(tmp_path / "perf.json"))
