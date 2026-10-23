"""Run the code of the example notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"


@pytest.mark.parametrize("name", ["reliability.ipynb"])
def test_notebook(name, monkeypatch):
    path = EXAMPLES / name
    monkeypatch.chdir(path.parent)
    namespace = {"__name__": "__main__"}
    for i, cell in enumerate(json.loads(path.read_text())["cells"]):
        # Plotting cells need packages outside the test environment.
        if cell["cell_type"] != "code" or "docs-only" in cell["metadata"].get("tags", []):
            continue
        code = compile("".join(cell["source"]), f"{name} cell {i}", "exec")
        exec(code, namespace)
