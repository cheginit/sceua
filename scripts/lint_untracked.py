"""Run pre-commit on untracked, non-ignored files.

``pre-commit run --all-files`` reads only git-tracked files, so a new module passes lint
until someone stages it. ``pixi r lint`` runs this after it.
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = [name for name in listed.split("\0") if name]
    if files:
        sys.exit(subprocess.call(["pre-commit", "run", "--files", *files]))  # noqa: S603, S607
