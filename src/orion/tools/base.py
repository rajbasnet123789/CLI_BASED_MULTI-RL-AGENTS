"""Tool primitives shared by the agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orion.approval import ApprovalCallback, ApprovalRequest
from orion.config import Mode, OrionConfig
from orion.errors import ToolError


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = False
    requires_approval: bool = False
    category: str = "files"

    def as_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class FileChange:
    path: str
    action: str  # create | update | delete | move
    before: str | None = None
    after: str | None = None
    diff: str = ""
    destination: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "diff": self.diff,
            "destination": self.destination,
        }


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    diff: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    display: str | None = None

    @classmethod
    def success(
        cls,
        content: str,
        *,
        data: dict[str, Any] | None = None,
        diff: str | None = None,
        changed_paths: list[str] | None = None,
        display: str | None = None,
    ) -> ToolResult:
        return cls(
            ok=True,
            content=content,
            data=data,
            diff=diff,
            changed_paths=changed_paths or [],
            display=display,
        )

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(ok=False, content=f"Error: {message}")


EmitCallback = Callable[[dict[str, Any]], None]


@dataclass
class ToolContext:
    config: OrionConfig
    mode: Mode = "build"
    approval: ApprovalCallback | None = None
    emit: EmitCallback | None = None
    dry_run: bool = False
    changes: list[FileChange] = field(default_factory=list)
    retriever: Any | None = None

    @property
    def workspace(self) -> Path:
        return self.config.workspace.resolve()

    @property
    def can_write(self) -> bool:
        return self.mode == "build"

    @property
    def apply_changes(self) -> bool:
        return self.mode == "build" and not self.dry_run

    def require_write(self) -> None:
        if self.mode != "build":
            raise ToolError(
                "Write operations are disabled in plan mode. Switch to build mode "
                "(press Tab in the TUI) to modify files."
            )

    def request_approval(
        self,
        tool: str,
        action: str,
        detail: str,
        risk: str = "medium",
    ) -> bool:
        if self.approval is None:
            return False
        return bool(
            self.approval(
                ApprovalRequest(tool=tool, action=action, detail=detail, risk=risk)
            )
        )


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


__all__ = [
    "EmitCallback",
    "FileChange",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "as_bool",
    "as_int",
    "as_str",
]
