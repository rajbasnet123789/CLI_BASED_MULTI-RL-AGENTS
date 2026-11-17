"""Human-in-the-loop approval for risky tool calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from orion.config import OrionConfig


@dataclass(frozen=True)
class ApprovalRequest:
    """A request shown to the user before a tool runs."""

    tool: str
    action: str
    detail: str
    risk: str = "medium"

    @property
    def title(self) -> str:
        return f"{self.action.replace('_', ' ').title()} - {self.tool}"


ApprovalCallback = Callable[[ApprovalRequest], bool]


class AutoApprove:
    def __call__(self, request: ApprovalRequest) -> bool:
        return True


class DenyAll:
    def __call__(self, request: ApprovalRequest) -> bool:
        return False


class ConsoleApproval:
    """Prompt on the terminal using Rich."""

    def __init__(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm

        self.console = Console()
        self._panel = Panel
        self._confirm = Confirm

    def __call__(self, request: ApprovalRequest) -> bool:
        body = f"[bold]{request.title}[/bold]\n\n{request.detail}"
        self.console.print(self._panel(body, border_style="yellow", title="approval required"))
        return bool(self._confirm.ask("Allow this action?", default=False))


def build_approval(config: OrionConfig, callback: ApprovalCallback | None = None) -> ApprovalCallback:
    """Combine the configured policy with an optional UI callback."""
    if config.approval == "auto":
        return AutoApprove()
    if config.approval == "deny":
        return DenyAll()
    return callback or ConsoleApproval()


__all__ = [
    "ApprovalCallback",
    "ApprovalRequest",
    "AutoApprove",
    "ConsoleApproval",
    "DenyAll",
    "build_approval",
]
