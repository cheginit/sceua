"""Print one Markdown table row per package change in a ``pixi update --json`` report.

Usage: ``python3 .agents/skills/update-package/summarize.py <report.json>``. The change
count goes to stderr; a report with no changes prints ``0 package changes``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# A conda entry carries only its URL: name-version-build.conda
CONDA_VERSION = re.compile(r"/[^/]+?-([^-/]+)-[^-/]+\.(?:conda|tar\.bz2)$")


def version(side: dict[str, Any] | None) -> str:
    """Return the version on one side of a change, ``-`` when the package is absent."""
    if not side:
        return "-"
    if "version" in side:
        return str(side["version"])
    match = CONDA_VERSION.search(str(side.get("conda", "")))
    return match.group(1) if match else "?"


def main(report: Path) -> None:
    """Print the rows for ``report``, one per package and version pair."""
    changes: dict[tuple[str, str, str], set[str]] = {}
    environments = json.loads(report.read_text(encoding="utf-8")).get("environment", {})
    for env, platforms in environments.items():
        for packages in platforms.values():
            for package in packages:
                key = (
                    package["name"],
                    version(package.get("before")),
                    version(package.get("after")),
                )
                changes.setdefault(key, set()).add(env)
    out = sys.stdout
    for (name, old, new), envs in sorted(changes.items()):
        out.write(f"| {name} | {old} | {new} | {', '.join(sorted(envs))} |\n")
    sys.stderr.write(f"{len(changes)} package changes\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
