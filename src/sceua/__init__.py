"""Top-level package for Seamless3dep."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from sceua.sceua import minimize

try:
    __version__ = version("sceua")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "__version__",
    "minimize",
]
