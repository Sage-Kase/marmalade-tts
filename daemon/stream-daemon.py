#!/usr/bin/env python3
"""marmalade-tts stream daemon — the streaming session front end.

Unlike its siblings in this directory, this daemon holds no model: it
plans a live stream (``marmalade_tts.stream_session``), runs the playback
gate, and drives the ordinary engine objects — which in daemon mode reach
the engine daemons here, one socket request per chunk. So it needs no
venv (system python + the marmalade_tts package, like venice-daemon.py
needs none for urllib), and it auto-starts the engine daemon it fronts
through the normal ``marmalade_tts.daemon`` path.

Socket: ~/.local/share/marmalade-tts/stream.sock (0600), newline-
delimited JSON, persistent connections. The wire contract and the whole
design live in ``marmalade_tts/stream_daemon.py``.
"""

import os
import sys

# The package is not necessarily importable from the system python, so
# find it the same way the marmalade-tts entrypoint does — plus the repo
# checkout, for running this script straight out of a clone.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _lib_dir in (
    os.path.dirname(_HERE),                              # repo checkout
    "/usr/lib/marmalade-tts",                            # deb / rpm
    os.path.expanduser("~/.local/lib/marmalade-tts"),    # install.sh
):
    if (os.path.isdir(os.path.join(_lib_dir, "marmalade_tts"))
            and _lib_dir not in sys.path):
        sys.path.insert(0, _lib_dir)
        break

# Dependency venv, for user installs that put PyYAML there.
_venv_site = os.path.join(
    os.path.expanduser("~/.local/lib/marmalade-tts"), ".venv", "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages")
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

from marmalade_tts.stream_daemon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
