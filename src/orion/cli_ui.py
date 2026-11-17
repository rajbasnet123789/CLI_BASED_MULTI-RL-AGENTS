"""Rich console renderer for headless CLI runs."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from orion.utils import format_duration, truncate


class ConsoleRenderer:
    """Turn engine events into readable terminal output."""

    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self.result: dict[str, Any] | None = None
        self.session_id: str | None = None
        self._streaming = False
        self._streamed_agents: set[str] = set()
        self._plan_printed = False

    def __call__(self, event: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{event.get('type', '')}", None)
        if handler is not None:
            handler(event)

    # ------------------------------------------------------------------
    def _finish_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    def _on_start(self, event: dict[str, Any]) -> None:
        self.session_id = event.get("session_id")
        mode = str(event.get("mode", "build"))
        style = "magenta" if mode == "plan" else "cyan"
        bits = [
            Text("◆ ORION AI", style="bold magenta"),
            Text("  "),
            Text(mode.upper(), style=f"bold {style}"),
            Text(f"  {event.get('model', '')}", style="dim"),
            Text(f"  {event.get('workspace', '')}", style="dim"),
        ]
        self.console.print(Text.assemble(*bits))
        self.console.print(Rule(style="grey27"))

    def _on_status(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        agent = event.get("agent", "")
        message = event.get("message", "")
        self.console.print(f"[dim]◌ {agent}: {message}[/dim]", highlight=False)

    def _on_token(self, event: dict[str, Any]) -> None:
        agent = str(event.get("agent", ""))
        self._streamed_agents.add(agent)
        if not self._streaming:
            self.console.print(f"[bold magenta]◆ {agent}[/bold magenta]")
            self._streaming = True
        self.console.print(
            str(event.get("text", "")),
            end="",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )

    def _on_message(self, event: dict[str, Any]) -> None:
        agent = str(event.get("agent", "orion"))
        text = str(event.get("text", ""))
        if not text:
            return
        if agent in self._streamed_agents and self._streaming:
            self._finish_stream()
            return
        self._finish_stream()
        if agent == "orion":
            self.console.print(Markdown(text))
            return
        color = {
            "coder": "cyan",
            "reviewer": "yellow",
            "debugger": "red",
            "planner": "magenta",
        }.get(agent, "blue")
        self.console.print(
            Panel(
                Markdown(text),
                title=f"[bold]{agent}[/bold]",
                border_style=color,
                padding=(0, 1),
            )
        )

    def _on_plan(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        if "planner" in self._streamed_agents and self._plan_printed:
            return
        markdown = str(event.get("markdown", ""))
        if not markdown:
            return
        self._plan_printed = True
        self.console.print(
            Panel(
                Markdown(markdown),
                title="[bold magenta]plan[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        name = event.get("name", "tool")
        args = event.get("args") or {}
        summary = self._summarize_args(str(name), args)
        self.console.print(
            f"[cyan]⚙ {name}[/cyan] [dim]{summary}[/dim]", highlight=False
        )

    def _on_tool_result(self, event: dict[str, Any]) -> None:
        ok = bool(event.get("ok"))
        display = str(event.get("display") or event.get("name") or "tool")
        marker = "[green]✓[/green]" if ok else "[red]✗[/red]"
        self.console.print(f"  {marker} [dim]{display}[/dim]", highlight=False)
        if self.verbose or not ok:
            content = str(event.get("content", ""))
            if content:
                self.console.print(
                    Panel(
                        Text(truncate(content, 4000, "tool output")),
                        border_style="grey30",
                        padding=(0, 1),
                    )
                )

    def _on_diff(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        diff = str(event.get("diff", ""))
        if not diff:
            return
        path = event.get("path", "")
        self.console.print(
            Panel(
                Syntax(diff, "diff", theme="monokai", word_wrap=True),
                title=f"[bold]{path}[/bold]",
                border_style="grey30",
                padding=(0, 1),
            )
        )

    def _on_review(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        verdict = str(event.get("verdict", ""))
        good = verdict == "approve"
        color = "green" if good else "red"
        lines = [Markdown(str(event.get("summary", "")))]
        issues = event.get("issues") or []
        if issues:
            lines.append(Text("\nIssues:", style="bold red"))
            for issue in issues:
                lines.append(Text(f"• {issue}", style="red"))
        self.console.print(
            Panel(
                Group(*lines),
                title=f"[bold {color}]review: {verdict.replace('_', ' ')}[/bold {color}]",
                border_style=color,
                padding=(0, 1),
            )
        )

    def _on_test(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        status = str(event.get("status", "unknown"))
        color = {"passed": "green", "failed": "red", "skipped": "yellow"}.get(status, "white")
        command = event.get("command") or "auto-detect"
        self.console.print(
            f"[{color}]tests: {status}[/{color}] [dim]{command}[/dim]", highlight=False
        )
        if status == "failed" and event.get("output"):
            self.console.print(
                Panel(
                    Text(truncate(str(event["output"]), 6000, "test output")),
                    border_style="red",
                    padding=(0, 1),
                )
            )
        if event.get("reason"):
            self.console.print(f"[dim]  {event['reason']}[/dim]")

    def _on_context(self, event: dict[str, Any]) -> None:
        count = event.get("count", 0)
        if count:
            self.console.print(f"[dim]◈ retrieved {count} context chunk(s)[/dim]")

    def _on_index_status(self, event: dict[str, Any]) -> None:
        phase = str(event.get("phase", ""))
        message = str(event.get("message", ""))
        if phase in {"done", "error"}:
            style = "green" if phase == "done" else "red"
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            self.console.print(f"[dim]◌ {message}[/dim]")

    def _on_warning(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        self.console.print(f"[yellow]! {event.get('message', '')}[/yellow]", highlight=False)

    def _on_error(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        self.console.print(
            Panel(
                Text(str(event.get("message", "unknown error")), style="red"),
                title="[bold red]error[/bold red]",
                border_style="red",
            )
        )

    def _on_cancelled(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        self.console.print(f"[yellow]{event.get('message', 'cancelled')}[/yellow]")

    def _on_done(self, event: dict[str, Any]) -> None:
        self._finish_stream()
        result = event.get("result") or {}
        self.result = result
        duration = float(result.get("duration") or 0)
        usage = result.get("usage") or {}
        calls = usage.get("calls", 0)
        tokens = usage.get("total_tokens", 0)
        session = result.get("session_id") or self.session_id or "-"
        self.console.print(Rule(style="grey27"))
        self.console.print(
            f"[dim]done in {format_duration(duration)} · {calls} model call(s) · "
            f"{tokens:,} tokens · session {session}[/dim]"
        )
        if result.get("error"):
            self.console.print(f"[red]{result['error']}[/red]")

    # ------------------------------------------------------------------
    @staticmethod
    def _summarize_args(name: str, args: dict[str, Any]) -> str:
        if name in {"read_file", "write_file", "edit_file", "delete_file"}:
            return str(args.get("path", ""))
        if name == "run_command":
            return truncate(str(args.get("command", "")), 100, "command")
        if name in {"grep", "glob", "search_code"}:
            return str(args.get("pattern") or args.get("query") or "")
        if name == "list_dir":
            return str(args.get("path", "."))
        try:
            return truncate(json.dumps(args, ensure_ascii=False), 100, "args")
        except (TypeError, ValueError):
            return truncate(str(args), 100, "args")


__all__ = ["ConsoleRenderer"]
