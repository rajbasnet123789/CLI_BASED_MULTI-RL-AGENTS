"""Headless smoke tests for the Textual UI."""

from __future__ import annotations

import pytest

from orion.config import OrionConfig

pytest.importorskip("textual")


@pytest.mark.asyncio
async def test_tui_renders_and_toggles_mode(config: OrionConfig) -> None:
    from orion.tui.app import OrionApp

    app = OrionApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#mode-chip")
        assert app.query_one("#input")
        assert app.mode == "build"

        await pilot.press("tab")
        await pilot.pause()
        assert app.mode == "plan"
        assert app.config.mode == "plan"

        await pilot.press("tab")
        await pilot.pause()
        assert app.mode == "build"


@pytest.mark.asyncio
async def test_tui_help_command(config: OrionConfig) -> None:
    from orion.tui.app import OrionApp

    app = OrionApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._handle_command("/help")
        await pilot.pause()
        # A help message should be mounted in the transcript.
        assert app._messages.children
        app._handle_command("/about")
        await pilot.pause()
