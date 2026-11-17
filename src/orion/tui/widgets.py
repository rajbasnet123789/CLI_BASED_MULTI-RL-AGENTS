"""Widgets and modal screens for the Orion TUI."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from orion.approval import ApprovalRequest
from orion.utils import truncate

COMMANDS: list[tuple[str, str]] = [
    ("plan", "Switch to plan mode (read-only)"),
    ("build", "Switch to build mode"),
    ("index", "Build or refresh the code index"),
    ("search", "Search the code index"),
    ("model", "Set the model (e.g. /model gemini-3.1-pro-preview)"),
    ("approve", "Cycle approval policy (ask / auto / deny)"),
    ("sessions", "List recent sessions"),
    ("diff", "Show changes from the last run"),
    ("context", "Show retrieved context"),
    ("new", "Start a new session"),
    ("clear", "Clear the conversation"),
    ("about", "Show version and configuration"),
    ("help", "Show keybindings and commands"),
    ("quit", "Exit Orion"),
]


class ChatInput(TextArea):
    """Multi-line input that submits on Enter and inserts a newline on Ctrl+J."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+j", "newline", "Newline", show=False),
        Binding("shift+enter", "newline", "Newline", show=False),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.show_line_numbers = False
        self.soft_wrap = True

    async def _on_key(self, event) -> None:  # type: ignore[override]
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        if event.key in {"ctrl+j", "shift+enter"}:
            event.stop()
            event.prevent_default()
            self.action_newline()
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        value = self.text.strip()
        if value:
            self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        self.insert("\n")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        lines = self.text.count("\n") + 1
        self.styles.height = max(3, min(8, lines + 1))


class MessageView(Vertical):
    """Base chat message with a header and body."""

    def __init__(
        self,
        *,
        role: str = "assistant",
        agent: str | None = None,
        text: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.agent = agent
        self._text = text
        self._body: Static | None = None
        self.finalized = False
        self.add_class(f"msg-{role}")

    def compose(self):
        yield Static(self._header(), classes="msg-header")
        self._body = Static(Text(self._text), classes="msg-body")
        yield self._body

    def _header(self) -> Text:
        if self.role == "user":
            return Text("▌ you", style="bold #22d3ee")
        label = self.agent or "orion"
        return Text(f"◆ {label}", style="bold #a78bfa")

    @property
    def text(self) -> str:
        return self._text

    def append_text(self, chunk: str) -> None:
        self._text += chunk
        if self._body is not None:
            self._body.update(Text(self._text))

    def finalize(self, text: str | None = None) -> None:
        if text is not None:
            self._text = text
        self.finalized = True
        if self._body is not None:
            self._body.update(Text(self._text))


class AssistantMessage(MessageView):
    """Assistant message that switches to Markdown once complete."""

    def __init__(self, *, agent: str | None = None, text: str = "", **kwargs: Any) -> None:
        super().__init__(role="orion" if agent == "orion" else "assistant", agent=agent, **kwargs)
        self._markdown: Markdown | None = None
        self._text = text

    def finalize(self, text: str | None = None) -> None:
        if text is not None:
            self._text = text
        self.finalized = True
        if not self._text.strip():
            return
        if self._body is not None:
            self._body.display = False
        if self._markdown is None:
            self._markdown = Markdown(self._text, classes="msg-body markdown-body")
            self.mount(self._markdown)
        else:
            self._markdown.update(self._text)


class NoteMessage(Static):
    """Small inline status note."""

    ICONS: ClassVar[dict[str, str]] = {"info": "·", "warning": "!", "error": "✗", "success": "✓"}

    def __init__(self, text: str, kind: str = "info", **kwargs: Any) -> None:
        icon = self.ICONS.get(kind, "·")
        super().__init__(Text(f"{icon} {text}"), classes=f"note note-{kind}", **kwargs)


class PlanView(Container):
    def __init__(self, markdown: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.markdown = markdown
        self.add_class("plan-view")

    def compose(self):
        yield Static(Text("PLAN", style="bold #e879f9"), classes="view-title")
        yield Markdown(self.markdown, classes="markdown-body")


class ReviewView(Container):
    def __init__(self, review: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.review = review
        verdict = str(review.get("verdict", ""))
        self.add_class("review-view")
        self.add_class("review-approve" if verdict == "approve" else "review-changes")

    def compose(self):
        verdict = str(self.review.get("verdict", "unknown"))
        label = verdict.replace("_", " ").upper()
        title = Text(f"REVIEW · {label}", style="bold")
        lines = [f"**{self.review.get('summary', '')}**"]
        issues = self.review.get("issues") or []
        if issues:
            lines.append("")
            lines.append("**Issues**")
            lines.extend(f"- {issue}" for issue in issues)
        suggestions = self.review.get("suggestions") or []
        if suggestions:
            lines.append("")
            lines.append("**Suggestions**")
            lines.extend(f"- {item}" for item in suggestions)
        yield Static(title, classes="view-title")
        yield Markdown("\n".join(lines), classes="markdown-body")


class TestView(Container):
    def __init__(self, result: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.result = result
        status = str(result.get("status", "unknown"))
        self.add_class("test-view")
        self.add_class(f"test-{status}")

    def compose(self):
        status = str(self.result.get("status", "unknown")).upper()
        yield Static(Text(f"TESTS · {status}", style="bold"), classes="view-title")
        lines = []
        if self.result.get("command"):
            lines.append(f"`{self.result['command']}`")
        if self.result.get("reason"):
            lines.append(str(self.result["reason"]))
        if self.result.get("exit_code") is not None:
            lines.append(f"Exit code: {self.result['exit_code']}")
        if status == "FAILED" and self.result.get("output"):
            lines.append("")
            lines.append("```")
            lines.append(truncate(str(self.result["output"]), 5000, "test output"))
            lines.append("```")
        yield Markdown("\n\n".join(lines) or "_No details._", classes="markdown-body")


class DiffView(Container):
    def __init__(self, path: str, diff: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.diff = diff
        self.add_class("diff-view")

    def compose(self):
        yield Static(Text(f"DIFF · {self.path}", style="bold #22d3ee"), classes="view-title")
        text = Text()
        lines = self.diff.splitlines()
        for line in lines[:500]:
            if line.startswith("+++") or line.startswith("---"):
                style = "bold #93c5fd"
            elif line.startswith("+"):
                style = "#34d399"
            elif line.startswith("-"):
                style = "#f87171"
            elif line.startswith("@@"):
                style = "#22d3ee"
            elif line.startswith("diff --git") or line.startswith("index "):
                style = "bold #9ca3af"
            else:
                style = "#9ca3af"
            text.append(line + "\n", style=style)
        if len(lines) > 500:
            text.append(f"... ({len(lines) - 500} more lines)\n", style="dim")
        yield Static(text, classes="msg-body")


class ToolCallView(Collapsible):
    """Collapsible tool invocation with its result."""

    def __init__(self, name: str, args: dict[str, Any], call_id: str, **kwargs: Any) -> None:
        self.tool_name = name
        self.call_id = call_id
        self._args = dict(args)
        preview = self._preview(name, args)
        self._body_text = preview + "\n\nrunning…"
        self._body = Static(Text(self._body_text, style="#9ca3af"), classes="tool-body")
        summary = self._summarize(name, args)
        title = Text.assemble(("⚙ ", "#a78bfa"), (name, "bold #c4b5fd"), (f"  {summary}", "#6b7280"))
        # Expanded by default: the transcript should show what each tool did,
        # not just a collapsed one-line title.
        super().__init__(self._body, title=title, collapsed=False, **kwargs)
        self.add_class("tool-view")

    @staticmethod
    def _preview(name: str, args: dict[str, Any]) -> str:
        """Show the actual command/path/content before the result arrives."""
        if name == "run_command":
            return f"$ {truncate(str(args.get('command', '')), 400, 'command')}"
        if name in {"read_file", "write_file", "edit_file", "delete_file"}:
            path = str(args.get("path", ""))
            content = str(args.get("content", ""))
            if content:
                return f"path: {path}\n\n{truncate(content, 2500, 'file content')}"
            old = str(args.get("old_string", ""))
            new = str(args.get("new_string", ""))
            if old or new:
                return (
                    f"path: {path}\n\n- {truncate(old, 1200, 'old string')}\n"
                    f"+ {truncate(new, 1200, 'new string')}"
                )
            return f"path: {path}"
        try:
            return truncate(
                json.dumps(args, indent=2, ensure_ascii=False), 2500, "tool arguments"
            )
        except (TypeError, ValueError):
            return truncate(str(args), 2500, "tool arguments")

    @staticmethod
    def _summarize(name: str, args: dict[str, Any]) -> str:
        if name in {"read_file", "write_file", "edit_file", "delete_file"}:
            return str(args.get("path", ""))
        if name == "run_command":
            return truncate(str(args.get("command", "")), 70, "command")
        if name in {"grep", "glob", "search_code"}:
            return str(args.get("pattern") or args.get("query") or "")
        if name == "list_dir":
            return str(args.get("path", "."))
        try:
            return truncate(json.dumps(args, ensure_ascii=False), 70, "args")
        except (TypeError, ValueError):
            return truncate(str(args), 70, "args")

    def set_result(self, ok: bool, display: str, content: str) -> None:
        marker = "✓" if ok else "✗"
        color = "#34d399" if ok else "#f87171"
        self.title = Text.assemble(
            (f"{marker} ", color),
            (self.tool_name, f"bold {color}"),
            (f"  {display}", "#6b7280"),
        )
        preview = self._preview(self.tool_name, self._args)
        body = truncate(content, 5000, "tool output") or "(no output)"
        self._body_text = f"{preview}\n\n→ {display}\n{body}"
        self._body.update(Text(self._body_text, style="#9ca3af"))
        self.collapsed = False


class ApprovalModal(ModalScreen[str]):
    """Blocking approval dialog used by the TUI."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "deny", "Deny"),
        Binding("a", "allow", "Allow"),
        Binding("d", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self):
        with Container(id="approval-dialog"):
            yield Static(Text(self.request.title, style="bold #fbbf24"), classes="modal-title")
            with VerticalScroll(classes="modal-detail"):
                yield Static(Text(self.request.detail), classes="modal-detail")
            with Horizontal(classes="modal-actions"):
                yield Button("Allow once", id="allow", variant="success")
                yield Button("Always allow", id="always", variant="primary")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "deny")

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_deny(self) -> None:
        self.dismiss("deny")


class CommandPalette(ModalScreen[str]):
    """Filterable slash-command palette."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_palette", "Close")]

    def __init__(self, commands: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self.commands = commands or COMMANDS

    def compose(self):
        with Container(id="palette-dialog"):
            yield Input(placeholder="Type a command…", id="palette-input")
            yield OptionList(id="palette-list")

    def on_mount(self) -> None:
        self._refresh("")
        self.query_one("#palette-input", Input).focus()

    def _refresh(self, query: str) -> None:
        options = self.query_one("#palette-list", OptionList)
        options.clear_options()
        needle = query.strip().lower()
        matches = [
            (name, description)
            for name, description in self.commands
            if not needle or needle in name.lower() or needle in description.lower()
        ]
        options.add_options(
            [Option(f"/{name:<10}  {description}", id=name) for name, description in matches]
        )
        if matches:
            options.highlighted = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        options = self.query_one("#palette-list", OptionList)
        if options.option_count:
            highlighted = options.highlighted
            option = options.get_option_at_index(highlighted if highlighted is not None else 0)
            if option.id:
                self.dismiss(option.id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(event.option.id)

    def action_dismiss_palette(self) -> None:
        self.dismiss(None)


__all__ = [
    "COMMANDS",
    "ApprovalModal",
    "AssistantMessage",
    "ChatInput",
    "CommandPalette",
    "DiffView",
    "MessageView",
    "NoteMessage",
    "PlanView",
    "ReviewView",
    "TestView",
    "ToolCallView",
]
