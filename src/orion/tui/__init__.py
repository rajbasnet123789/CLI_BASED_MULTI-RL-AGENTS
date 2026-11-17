"""Interactive terminal UI for Orion AI."""

from __future__ import annotations

from typing import Any

__all__ = ["OrionApp", "run_tui"]


def __getattr__(name: str) -> Any:
    if name in {"OrionApp", "run_tui"}:
        from orion.tui import app

        return getattr(app, name)
    raise AttributeError(name)
