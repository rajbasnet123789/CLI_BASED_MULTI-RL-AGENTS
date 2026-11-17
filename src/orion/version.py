"""Version information for Orion AI."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = "1.0.5"

try:  # Prefer installed distribution metadata so wheels report the real version.
    __version__ = version("orion-ai")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    pass

__all__ = ["__version__"]
