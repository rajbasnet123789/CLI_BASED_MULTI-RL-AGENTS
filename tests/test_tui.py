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
async def test_tui_shows_tool_calls_and_changes(config: OrionConfig) -> None:
    from orion.tui.app import OrionApp
    from orion.tui.widgets import AssistantMessage, ToolCallView

    app = OrionApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._handle_event({
            "type": "tool_call",
            "agent": "generator",
            "call_id": "call-1",
            "name": "run_command",
            "args": {"command": "py -3.10 hello.py"},
        })
        await pilot.pause()
        views = list(app.query(ToolCallView))
        assert views
        assert views[0].collapsed is False
        assert "py -3.10 hello.py" in views[0]._body_text

        app._handle_event({
            "type": "tool_result",
            "agent": "generator",
            "call_id": "call-1",
            "name": "run_command",
            "ok": True,
            "display": "$ py -3.10 hello.py (exit 0)",
            "content": "HELLO_ORION",
            "changed_paths": [],
        })
        await pilot.pause()
        assert "HELLO_ORION" in views[0]._body_text

        app._handle_event({
            "type": "done",
            "result": {
                "task": "test",
                "mode": "build",
                "final": "done",
                "changes": [{"action": "create", "path": "hello.py", "diff": ""}],
                "usage": {},
                "session_id": None,
                "duration": 1.0,
            },
        })
        await pilot.pause()
        texts = [str(msg.text) for msg in app.query(AssistantMessage)]
        assert any("Files changed in this run" in text and "hello.py" in text for text in texts)


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
