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


@pytest.mark.asyncio
async def test_tui_runs_prompt_and_shows_agent_roles(
    config: OrionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typing a prompt must start a run and surface roles and changed files."""
    from tests.test_engine import FakeLLM

    import orion.agents.nodes as nodes
    from orion.tui.app import OrionApp
    from orion.tui.widgets import AssistantMessage, DiffView, NoteMessage

    monkeypatch.setattr(nodes, "detect_verification_commands", lambda root: [])
    app = OrionApp(config)
    app.engine.llm = FakeLLM(tool_call=True)
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        app.query_one("#input").focus()
        await pilot.press(*"create a health helper")
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.2)
            if not app._run_active and app._last_result is not None:
                break
        assert app._last_result is not None
        assert app._last_result.error is None
        assert (config.workspace / "pkg" / "generated.py").is_file()
        assert list(app.query(DiffView))
        agents = [str(m.agent) for m in app.query(AssistantMessage)]
        for role in ("summarizer", "architect", "generator", "reviewer", "evaluator"):
            assert any(a.startswith(role) for a in agents), role
        assert "pkg/generated.py" in app._run_files
        notes = [str(n.render()) for n in app.query(NoteMessage)]
        assert any("generator agent working" in n for n in notes)
