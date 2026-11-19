"""Interactive Orion terminal UI built with Textual.

The TUI is a thin layer over :class:`orion.engine.OrionEngine`: the engine runs
on a worker thread, emits structured events, and the app renders them. The
agent pipeline itself never imports Textual, which keeps the library usable
headlessly.
"""

from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from orion.approval import ApprovalRequest
from orion.config import Mode, OrionConfig
from orion.engine import OrionEngine, RunResult
from orion.tui.theme import ORION_CSS, ORION_THEME
from orion.tui.widgets import (
    ApprovalModal,
    AssistantMessage,
    ChatInput,
    CommandPalette,
    DiffView,
    MessageView,
    NoteMessage,
    PlanView,
    ReviewView,
    TestView,
    ToolCallView,
)
from orion.utils import format_duration, truncate
from orion.version import __version__

LOGO = """\
   ____  _____   ___  _   _
  / __ \\|  _  | / _ \\| \\ | |
 | |  | | |_| || |_| |  \\| |
 | |__| |  _  ||  _  | |\\  |
  \\____/|_| |_||_| |_|_| \\_|
"""

WELCOME = (
    "Multi-agent coding assistant · Vertex AI + LangGraph + ChromaDB\n"
    "Works in the directory below: creates files and runs project commands there.\n"
    "Type a task and press Enter. Tab toggles plan/build. Ctrl+P opens commands."
)


# Graph node names -> the role shown to the user (matches the AGENTS sidebar).
ROLE_LABELS: dict[str, str] = {
    "context": "context",
    "summarizer": "summarizer",
    "planner": "architect",
    "architect": "architect",
    "coder": "generator",
    "generator": "generator",
    "reviewer": "reviewer",
    "evaluator": "evaluator",
    "debugger": "debugger",
    "orion": "orion",
}

ROLE_COLORS: dict[str, str] = {
    "context": "#9ca3af",
    "summarizer": "#22d3ee",
    "architect": "#a78bfa",
    "generator": "#4ade80",
    "reviewer": "#fbbf24",
    "evaluator": "#f472b6",
    "debugger": "#fb923c",
}


class OrionApp(App[None]):
    """Orion's opencode-style terminal interface."""

    TITLE = "Orion AI"
    SUB_TITLE = "multi-agent coding assistant"
    CSS = ORION_CSS

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel / quit", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+n", "new_session", "New session", show=False),
        Binding("tab", "toggle_mode", "Toggle mode", priority=True),
        Binding("escape", "focus_input", "Focus input", show=False),
        Binding("f1", "help", "Help", show=False),
    ]

    def __init__(self, config: OrionConfig) -> None:
        super().__init__()
        self.config = config
        self.engine = OrionEngine(config, approval=self._request_approval)
        self.mode: Mode = config.mode
        self._run_active = False
        self._active_message: AssistantMessage | None = None
        self._active_agent: str | None = None
        self._tool_views: dict[str, ToolCallView] = {}
        self._last_result: RunResult | None = None
        self._session_id: str | None = None
        self._usage: dict[str, int] = {}
        self._status = "ready"
        self._index_summary = ""
        self._started_at = 0.0
        self._notes_mounted = 0
        self._current_role: str | None = None
        self._run_files: dict[str, str] = {}

    @staticmethod
    def _role(agent: str) -> str:
        return ROLE_LABELS.get(agent, agent)

    def _role_model(self, role: str) -> str:
        try:
            return self.config.model_for(role)
        except Exception:
            return self.config.active_model

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(
                Text.assemble(("◆ ", "bold #f472b6"), ("ORION AI", "bold #a78bfa")),
                id="logo",
            )
            yield Static("", id="topbar-info")
        with Horizontal(id="body"):
            with Vertical(id="main"):
                yield VerticalScroll(id="messages")
                with Horizontal(id="inputbar"):
                    yield Static("", id="mode-chip")
                    yield ChatInput(
                        id="input",
                        placeholder="Describe a task…  (Enter to send, Ctrl+J for newline)",
                    )
                    yield Static("enter ↵", id="input-hint")
            with VerticalScroll(id="sidebar"):
                yield Static("", id="sidebar-content")
        yield Static("", id="statusbar")

    def on_mount(self) -> None:
        self.register_theme(ORION_THEME)
        self.theme = "orion"
        self._render_welcome()
        self._update_chip()
        self._refresh_topbar()
        self._refresh_sidebar()
        self._set_status("ready")
        self.query_one("#input", ChatInput).focus()
        if not self.config.has_credentials:
            self._note(
                "Vertex AI credentials are missing. Run "
                "'gcloud auth application-default login' and set "
                "ORION_GCLOUD_PROJECT in .env. No API key is needed.",
                "warning",
            )
        elif self.config.auto_index:
            self.run_worker(self._warm_index, thread=True, group="index")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    @property
    def _messages(self) -> VerticalScroll:
        return self.query_one("#messages", VerticalScroll)

    def _render_welcome(self) -> None:
        logo = Static(Text(LOGO, style="#a78bfa"), classes="welcome-logo")
        text = Static(
            Text.assemble(
                ("Orion AI ", "bold #f472b6"),
                (f"v{__version__}\n", "dim"),
                (WELCOME + "\n\n", ""),
                ("model  ", "dim"),
                (self.config.active_model + "\n", "#22d3ee"),
                ("mode   ", "dim"),
                (self.mode.upper() + "\n", "#c4b5fd"),
                ("dir    ", "dim"),
                (str(self.config.workspace), "#9ca3af"),
            ),
            classes="welcome",
        )
        self._messages.mount(logo)
        self._messages.mount(text)

    def _scroll_end(self) -> None:
        try:
            self._messages.scroll_end(animate=False)
        except Exception:
            pass

    def _after_mount(self, widget: Any) -> Any:
        self._messages.mount(widget)
        self.call_after_refresh(self._scroll_end)
        return widget

    def _note(self, text: str, kind: str = "info") -> NoteMessage:
        note = NoteMessage(text, kind)
        self._notes_mounted += 1
        return self._after_mount(note)  # type: ignore[return-value]

    def _update_chip(self) -> None:
        chip = self.query_one("#mode-chip", Static)
        chip.remove_class("mode-build", "mode-plan")
        if self.mode == "plan":
            chip.add_class("mode-plan")
            chip.update(Text(" PLAN ", style="bold #bae6fd"))
        else:
            chip.add_class("mode-build")
            chip.update(Text(" BUILD ", style="bold #c4b5fd"))

    def _refresh_topbar(self) -> None:
        info = self.query_one("#topbar-info", Static)
        approval = self.config.approval
        info.update(
            Text.assemble(
                ("  ", ""),
                (self.config.active_model, "#22d3ee"),
                ("  ·  ", "#374151"),
                (f"approval {approval}", "#9ca3af"),
                ("  ·  ", "#374151"),
                (self._short_workspace(), "#6b7280"),
            )
        )

    def _short_workspace(self) -> str:
        path = str(self.config.workspace)
        return path if len(path) <= 48 else "…" + path[-47:]

    def _set_status(self, message: str | None = None) -> None:
        if message is not None:
            self._status = message
        bar = self.query_one("#statusbar", Static)
        usage = self._usage or {}
        tokens = usage.get("total_tokens", 0)
        calls = usage.get("calls", 0)
        session = (self._session_id or "—")[-18:]
        index = self._index_summary or "index: ?"
        bar.update(
            Text.assemble(
                (" ", ""),
                ("● " if self._run_active else "○ ", "#f472b6" if self._run_active else "#374151"),
                (self._status, "#9ca3af"),
                ("   ", ""),
                (index, "#6b7280"),
                ("   ", ""),
                (f"{calls} calls · {tokens:,} tokens", "#6b7280"),
                ("   ", ""),
                (f"session {session}", "#4b5563"),
                ("   ", ""),
                ("tab mode · ctrl+p commands · ctrl+c cancel", "#374151"),
            )
        )

    def _refresh_sidebar(self) -> None:
        lines = Text()
        lines.append("SESSION\n", style="bold #6b7280")
        lines.append(f"mode     {self.mode}\n", style="#9ca3af")
        lines.append(f"approval {self.config.approval}\n", style="#9ca3af")
        lines.append(f"model    {self.config.active_model}\n", style="#9ca3af")
        lines.append(f"session  {(self._session_id or '—')[-16:]}\n", style="#6b7280")
        lines.append("\nAGENTS\n", style="bold #6b7280")
        for role in (
            "summarizer",
            "architect",
            "generator",
            "reviewer",
            "evaluator",
            "debugger",
        ):
            active = self._run_active and self._current_role == role
            color = ROLE_COLORS.get(role, "#6b7280")
            lines.append("▶ " if active else "  ", style=f"bold {color}")
            lines.append(
                f"{role:11} {self.config.model_for(role)}\n",
                style=f"bold {color}" if active else "#6b7280",
            )
        if self._run_files:
            lines.append("\nFILES THIS RUN\n", style="bold #6b7280")
            for path, action in list(self._run_files.items())[-12:]:
                lines.append(f"  {action:7} {path}\n", style="#4ade80")
        if self._last_result is not None:
            result = self._last_result
            lines.append("\nLAST RUN\n", style="bold #6b7280")
            lines.append(
                f"status   {'failed' if result.error else ('cancelled' if result.cancelled else 'ok')}\n",
                style="#9ca3af",
            )
            lines.append(f"changes  {len(result.changes)}\n", style="#9ca3af")
            lines.append(f"time     {format_duration(result.duration)}\n", style="#6b7280")
            if result.changes:
                lines.append("\nFILES\n", style="bold #6b7280")
                for change in result.changes[:12]:
                    lines.append(f"  {change['action']} {change['path']}\n", style="#6b7280")
        lines.append("\nINDEX\n", style="bold #6b7280")
        lines.append(self._index_summary or "  not built\n", style="#6b7280")
        lines.append("\nKEYS\n", style="bold #6b7280")
        lines.append("  enter      send\n", style="#4b5563")
        lines.append("  tab        plan / build\n", style="#4b5563")
        lines.append("  ctrl+p     commands\n", style="#4b5563")
        lines.append("  ctrl+j     newline\n", style="#4b5563")
        lines.append("  ctrl+c     cancel / quit\n", style="#4b5563")
        self.query_one("#sidebar-content", Static).update(lines)

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------
    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        input_widget = self.query_one("#input", ChatInput)
        input_widget.clear()
        input_widget.styles.height = 3
        if text.startswith("/"):
            self._handle_command(text)
            return
        if self._run_active:
            self._note("A run is already in progress. Press Ctrl+C to cancel it.", "warning")
            return
        self._messages.mount(MessageView(role="user", text=text))
        self._scroll_end()
        self._start_run(text)

    def action_focus_input(self) -> None:
        self.query_one("#input", ChatInput).focus()

    def action_toggle_mode(self) -> None:
        self.mode = "plan" if self.mode == "build" else "build"
        self.config = self.config.with_overrides(mode=self.mode)
        self.engine.config = self.config
        self._update_chip()
        self._refresh_sidebar()
        self._note(
            "Plan mode: read-only tools. Orion will investigate and propose a plan."
            if self.mode == "plan"
            else "Build mode: Orion can edit files and run approved commands.",
            "info",
        )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def _start_run(self, task: str) -> None:
        self._run_active = True
        self._active_message = None
        self._active_agent = None
        self._tool_views.clear()
        self._run_files = {}
        self._current_role = None
        self._started_at = time.monotonic()
        self._set_status(f"{self.mode} · thinking")
        self._set_busy(True)
        self.run_worker(
            lambda: self._run_task(task, self.mode),
            thread=True,
            group="run",
            exclusive=True,
        )

    def _set_busy(self, busy: bool) -> None:
        input_widget = self.query_one("#input", ChatInput)
        input_widget.disabled = busy
        hint = self.query_one("#input-hint", Static)
        hint.update(Text("esc to stop" if busy else "enter ↵", style="#4b5563"))
        if not busy:
            input_widget.focus()

    def _dispatch(self, event: dict[str, Any]) -> None:
        """Called from the engine thread; hop back to the UI thread."""
        try:
            self.call_from_thread(self._handle_event, event)
        except Exception:
            pass

    def _run_task(self, task: str, mode: Mode) -> None:
        try:
            self.engine.run(task, mode=mode, on_event=self._dispatch)
        except Exception as exc:
            self._dispatch({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            self._dispatch({"type": "done", "result": None})

    def _finalize_active(self) -> None:
        if self._active_message is not None:
            self._active_message.finalize()
            self._active_message = None
            self._active_agent = None

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        handler = getattr(self, f"_ev_{kind}", None)
        if handler is not None:
            handler(event)
        if kind not in {"token"}:
            self._scroll_end()

    # -- individual events ---------------------------------------------
    def _ev_start(self, event: dict[str, Any]) -> None:
        self._session_id = event.get("session_id")
        self._usage = {}
        self.mode = event.get("mode", self.mode)  # type: ignore[assignment]
        self._update_chip()
        self._set_status(f"{self.mode} · starting")
        self._refresh_topbar()

    def _ev_status(self, event: dict[str, Any]) -> None:
        agent = str(event.get("agent", "") or "")
        message = event.get("message", "")
        role = self._role(agent) if agent else ""
        if role and role != self._current_role:
            self._current_role = role
            self._finalize_active()
            if role != "orion":
                model = self._role_model(role)
                self._after_mount(
                    NoteMessage(f"▶ {role} agent working · model {model}", "info")
                )
            self._refresh_sidebar()
        self._set_status(f"{role} · {message}" if role else str(message))

    def _ev_usage(self, event: dict[str, Any]) -> None:
        self._usage = event.get("usage") or {}
        self._set_status()

    def _ev_token(self, event: dict[str, Any]) -> None:
        agent = self._role(str(event.get("agent", "orion")))
        text = str(event.get("text", ""))
        if not text:
            return
        if self._active_message is None or self._active_agent != agent:
            self._finalize_active()
            message = AssistantMessage(agent=agent)
            self._messages.mount(message)
            self._active_message = message
            self._active_agent = agent
        self._active_message.append_text(text)
        self._scroll_end()

    def _ev_message(self, event: dict[str, Any]) -> None:
        agent = self._role(str(event.get("agent", "orion")))
        text = str(event.get("text", ""))
        if not text.strip():
            return
        if self._active_message is not None and self._active_agent == agent:
            self._active_message.finalize()
            self._active_message = None
            self._active_agent = None
            return
        self._finalize_active()
        message = AssistantMessage(agent=agent, text=text)
        self._messages.mount(message)
        message.finalize()

    def _ev_model_output(self, event: dict[str, Any]) -> None:
        """Show the raw answer from every agent/model call in the transcript."""
        self._finalize_active()
        text = str(event.get("text", ""))
        if not text.strip():
            return
        agent = self._role(str(event.get("agent", "agent")))
        model = str(event.get("model", ""))
        label = f"{agent} · {model}" if model else agent
        message = AssistantMessage(agent=label, text=text)
        self._after_mount(message)
        message.finalize()

    def _ev_plan(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        markdown = str(event.get("markdown", ""))
        if markdown:
            self._after_mount(PlanView(markdown))

    def _ev_tool_call(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        call_id = str(event.get("call_id") or "")
        view = ToolCallView(
            str(event.get("name", "tool")),
            event.get("args") or {},
            call_id,
        )
        self._tool_views[call_id] = view
        self._after_mount(view)
        self._set_status(f"tool · {event.get('name', '')}")

    def _ev_tool_result(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id") or "")
        view = self._tool_views.get(call_id)
        if view is not None:
            view.set_result(
                bool(event.get("ok")),
                str(event.get("display") or event.get("name") or "tool"),
                str(event.get("content") or ""),
            )
        if event.get("ok"):
            name = str(event.get("name", ""))
            action = {
                "write_file": "write",
                "edit_file": "edit",
                "delete_file": "delete",
                "move_file": "move",
            }.get(name)
            for path in event.get("changed_paths") or []:
                self._run_files[str(path)] = action or "change"
                role = self._role(str(event.get("agent", "")))
                self._note(f"✎ {role} {action or 'changed'} {path}", "success")
            if event.get("changed_paths"):
                self._refresh_sidebar()
        self._set_status()

    def _ev_diff(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        diff = str(event.get("diff", ""))
        if diff:
            self._after_mount(DiffView(str(event.get("path", "")), diff))

    def _ev_evaluation(self, event: dict[str, Any]) -> None:
        verdict = str(event.get("verdict", "?"))
        summary = str(event.get("summary", ""))
        kind = "success" if verdict == "approve" else "warning"
        self._note(f"evaluator verdict: {verdict}. {summary}".strip(), kind)

    def _ev_review(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        payload = {key: value for key, value in event.items() if key != "type"}
        self._after_mount(ReviewView(payload))

    def _ev_test(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        payload = {key: value for key, value in event.items() if key != "type"}
        self._after_mount(TestView(payload))

    def _ev_context(self, event: dict[str, Any]) -> None:
        count = int(event.get("count", 0) or 0)
        if count:
            self._note(f"retrieved {count} context chunk(s) from the code index")

    def _ev_warning(self, event: dict[str, Any]) -> None:
        self._note(str(event.get("message", "warning")), "warning")

    def _ev_error(self, event: dict[str, Any]) -> None:
        self._note(str(event.get("message", "unknown error")), "error")
        self._set_status("error")

    def _ev_cancelled(self, event: dict[str, Any]) -> None:
        self._note(str(event.get("message", "cancelled")), "warning")

    def _ev_index_status(self, event: dict[str, Any]) -> None:
        phase = str(event.get("phase", ""))
        message = str(event.get("message", ""))
        if phase in {"scan", "chunk", "embed", "write"}:
            self._set_status(f"index · {message}")
        elif phase == "done":
            stats = event.get("stats") or {}
            self._index_summary = (
                f"{stats.get('vectors', 0)} vectors · "
                f"{stats.get('files_indexed', 0)} files · "
                f"{stats.get('embedder', '')}"
            )
            self._set_status("index ready")
            self._refresh_sidebar()
            self._note(message, "success")
        elif phase == "error":
            self._set_status("index failed")
            self._note(message, "error")

    def _ev_done(self, event: dict[str, Any]) -> None:
        self._finalize_active()
        result = event.get("result")
        self._run_active = False
        self._set_busy(False)
        if isinstance(result, dict):
            self._last_result = RunResult(
                task=str(result.get("task", "")),
                mode=str(result.get("mode", self.mode)),
                final=str(result.get("final", "")),
                changes=list(result.get("changes") or []),
                usage=dict(result.get("usage") or {}),
                session_id=result.get("session_id"),
                duration=float(result.get("duration") or 0.0),
                cancelled=bool(result.get("cancelled")),
                error=result.get("error"),
            )
            self._session_id = self._last_result.session_id or self._session_id
            self._usage = self._last_result.usage or self._usage
            elapsed = format_duration(self._last_result.duration)
            if self._last_result.error:
                self._set_status(f"failed in {elapsed}")
            elif self._last_result.cancelled:
                self._set_status("cancelled")
            else:
                self._set_status(f"done in {elapsed}")
        else:
            self._set_status("done")
        changes = self._last_result.changes if self._last_result is not None else []
        if changes:
            lines = ["**Files changed in this run**"]
            for change in changes:
                lines.append(
                    f"- `{change.get('action', 'update')}` `{change.get('path', '')}`"
                )
            lines.extend(["", "Use `/diff` to re-open all diffs."])
            self._after_mount(
                AssistantMessage(agent="changes", text="\n".join(lines))
            ).finalize()
        self._refresh_sidebar()
        self._refresh_topbar()

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------
    def _request_approval(self, request: ApprovalRequest) -> bool:
        """Called on a worker thread; blocks until the modal is dismissed."""
        if self.config.approval == "auto":
            return True
        if self.config.approval == "deny":
            return False
        done = threading.Event()
        holder: dict[str, Any] = {"allow": False}

        def show() -> None:
            def on_close(result: str | None) -> None:
                holder["allow"] = result in {"allow", "always"}
                if result == "always":
                    self._apply_config(approval="auto")
                done.set()

            try:
                self.push_screen(ApprovalModal(request), on_close)
            except Exception:
                done.set()

        try:
            self.call_from_thread(show)
        except Exception:
            return False
        done.wait(timeout=600)
        return bool(holder.get("allow"))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._on_palette_result)

    def _on_palette_result(self, command: str | None) -> None:
        if command:
            self._handle_command(f"/{command}")

    def _apply_config(self, **overrides: Any) -> None:
        self.config = self.config.with_overrides(**overrides)
        self.engine.config = self.config
        self.mode = self.config.mode
        self._update_chip()
        self._refresh_topbar()
        self._refresh_sidebar()

    def _handle_command(self, raw: str) -> None:
        parts = raw[1:].split()
        if not parts:
            return
        command, args = parts[0].lower(), parts[1:]
        if command in {"help", "?"}:
            self.action_help()
        elif command == "plan":
            if self.mode != "plan":
                self.action_toggle_mode()
            self._note("Plan mode enabled. Send a task to investigate.")
        elif command == "build":
            if self.mode != "build":
                self.action_toggle_mode()
            self._note("Build mode enabled. Orion can modify files.")
        elif command == "mode":
            self.action_toggle_mode()
        elif command == "index":
            self._command_index(args)
        elif command == "search":
            query = " ".join(args).strip()
            if not query:
                self._note("Usage: /search <query>", "warning")
                return
            self.run_worker(lambda: self._search_worker(query), thread=True, group="search")
            self._set_status(f"searching · {truncate(query, 40)}")
        elif command == "model":
            if not args:
                self._note(f"Current model: {self.config.active_model}", "info")
                return
            self._apply_config(model=args[0])
            self._note(f"Active model: {self.config.active_model}", "success")
        elif command == "approve":
            order = ["ask", "auto", "deny"]
            policy = order[(order.index(self.config.approval) + 1) % len(order)]
            self._apply_config(approval=policy)
            self._note(
                f"Approval policy: {policy}"
                + (" (commands run without asking)" if policy == "auto" else ""),
                "info",
            )
        elif command == "sessions":
            self._show_sessions()
        elif command == "diff":
            self._show_diff()
        elif command == "context":
            self._show_context()
        elif command in {"new", "clear"}:
            self._messages.remove_children()
            self._active_message = None
            self._last_result = None
            self._session_id = None
            self._render_welcome()
            self._refresh_sidebar()
            self._set_status("new session")
        elif command == "about":
            self._show_about()
        elif command == "quit":
            self.exit()
        else:
            self._note(f"Unknown command: /{command}. Try /help.", "warning")

    def _command_index(self, args: list[str]) -> None:
        force = "force" in {arg.lower() for arg in args}
        embeddings = None
        for arg in args:
            lowered = arg.lower()
            if lowered in {"local", "gemini", "auto"}:
                embeddings = lowered
        label = embeddings or self.config.embeddings
        self._set_status(f"index · starting ({label})")

        def work() -> None:
            try:
                stats = self.engine.index(
                    force=force,
                    embeddings=embeddings,
                    on_event=self._dispatch,
                )
                self._index_summary = (
                    f"{stats.vectors} vectors · {stats.files_indexed} files · {stats.embedder}"
                )
                self._dispatch(
                    {
                        "type": "index_status",
                        "phase": "done",
                        "message": (
                            f"Indexed {stats.files_indexed} file(s), "
                            f"{stats.chunks} chunk(s) with {stats.embedder}"
                        ),
                        "stats": stats.as_dict(),
                    }
                )
            except Exception as exc:
                self._dispatch(
                    {
                        "type": "index_status",
                        "phase": "error",
                        "message": f"Indexing failed: {exc}",
                    }
                )

        self.run_worker(work, thread=True, group="index", exclusive=True)

    def _warm_index(self) -> None:
        try:
            self.engine.prepare(emit=self._dispatch)
        except Exception as exc:
            self._dispatch({"type": "warning", "message": f"Index warm-up failed: {exc}"})

    def _search_worker(self, query: str) -> None:
        try:
            hits = self.engine.search(query, k=self.config.top_k)
        except Exception as exc:
            self._dispatch({"type": "warning", "message": f"Search failed: {exc}"})
            return
        if not hits:
            self._dispatch(
                {
                    "type": "warning",
                    "message": f"No matches for {query!r}. Try /index first.",
                }
            )
            return
        for hit in hits[:5]:
            self._dispatch(
                {
                    "type": "message",
                    "agent": "search",
                    "text": (
                        f"**{hit['path']}:{hit['start_line']}-{hit['end_line']}** "
                        f"(score {hit['score']:.2f})\n\n```{hit.get('language', 'text')}\n"
                        f"{truncate(str(hit.get('text', '')), 900)}\n```"
                    ),
                }
            )

    def _show_sessions(self) -> None:
        try:
            from orion.session import SessionStore

            metas = SessionStore(self.config).list(limit=8)
        except Exception as exc:
            self._note(f"Could not read sessions: {exc}", "warning")
            return
        if not metas:
            self._note("No sessions yet.")
            return
        lines = [
            f"- `{meta.id}` · {meta.status} · {meta.mode} · {meta.title}"
            for meta in metas
        ]
        self._after_mount(
            AssistantMessage(agent="sessions", text="**Recent sessions**\n\n" + "\n".join(lines))
        ).finalize()

    def _show_diff(self) -> None:
        if self._last_result is None or not self._last_result.changes:
            self._note("No changes in the last run.")
            return
        for change in self._last_result.changes:
            diff = change.get("diff") or ""
            if diff:
                self._after_mount(DiffView(change["path"], diff))

    def _show_context(self) -> None:
        if self._last_result is None or not self._last_result.context:
            self._note("No retrieved context in the last run.")
            return
        lines = [
            f"- `{hit['path']}:{hit['start_line']}-{hit['end_line']}` "
            f"(score {hit.get('score', 0):.2f})"
            for hit in self._last_result.context
        ]
        self._after_mount(
            AssistantMessage(
                agent="context",
                text="**Retrieved context**\n\n" + "\n".join(lines),
            )
        ).finalize()

    def _show_about(self) -> None:
        text = (
            f"**Orion AI** v{__version__}\n\n"
            f"- provider: `{self.config.llm_provider}`\n"
            f"- model: `{self.config.active_model}`\n"
            f"- embeddings: `{self.config.embeddings}` ({self.config.embedding_model})\n"
            f"- workspace: `{self.config.workspace}`\n"
            f"- data: `{self.config.data_path}`\n"
            f"- credentials: {self.config.has_credentials} ({self.config.masked_key})\n"
            f"- vertex: {self.config.vertex_enabled}"
        )
        self._after_mount(AssistantMessage(agent="orion", text=text)).finalize()

    def action_help(self) -> None:
        text = (
            "**Commands**\n\n"
            "- `/plan` · `/build` — switch mode (Tab)\n"
            "- `/index [force] [local|gemini]` — build the code index\n"
            "- `/search <query>` — hybrid code search\n"
            "- `/model <name>` — switch model (Gemini alias or local tag)\n"
            "- `/approve` — cycle ask → auto → deny\n"
            "- `/sessions` — recent sessions\n"
            "- `/diff` — changes from the last run\n"
            "- `/context` — retrieved context\n"
            "- `/new` · `/clear` — reset the conversation\n"
            "- `/about` · `/quit`\n\n"
            "**Keys**\n\n"
            "- Enter sends · Ctrl+J inserts a newline\n"
            "- Tab toggles plan/build · Ctrl+P command palette\n"
            "- Ctrl+C cancels a run (or quits when idle)"
        )
        self._after_mount(AssistantMessage(agent="help", text=text)).finalize()

    def action_new_session(self) -> None:
        self._handle_command("/new")

    def action_cancel_or_quit(self) -> None:
        if self._run_active:
            self.engine.cancel()
            self._set_status("cancelling…")
            self._note("Cancelling the current run…", "warning")
        else:
            self.exit()

    def action_quit(self) -> None:
        self.exit()


def run_tui(config: OrionConfig) -> None:
    """Launch the interactive app (blocking)."""
    OrionApp(config).run()


__all__ = ["OrionApp", "run_tui"]
