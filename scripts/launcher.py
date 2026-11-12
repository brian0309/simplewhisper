"""Entry point for the frozen (PyInstaller) SimpleWhisper builds.

PyInstaller runs this file instead of ``python -m simplewhisper``. Two details
need care here:

* The speaker detection runs in a child process. On Windows that child
  starts this executable again with a special argument, so
  ``multiprocessing.freeze_support()`` has to run before anything else.
* The console build runs the command line interface and the windowed build
  runs the desktop window. The two differ only in the name of the
  executable, so the name decides what starts.
"""

from __future__ import annotations

import multiprocessing
import os
import sys


def _is_windowed_build() -> bool:
    """True when this is the windowed (desktop) build."""
    if not getattr(sys, "frozen", False):
        return False
    stem = os.path.splitext(os.path.basename(sys.executable))[0].lower()
    return stem.endswith("-gui") or stem == "gui"


def main() -> int:
    # In the frozen build this is what turns the re-executed executable
    # into the speaker-detection child process. It does nothing when the
    # program is started normally.
    multiprocessing.freeze_support()

    if _is_windowed_build():
        from simplewhisper.gui import main as gui_main

        return gui_main()

    from simplewhisper.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
