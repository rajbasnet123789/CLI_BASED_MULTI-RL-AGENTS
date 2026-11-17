"""Tool registry with mode-aware schemas and execution."""

from __future__ import annotations

from typing import Any

from orion.config import Mode, OrionConfig
from orion.errors import OrionError, ToolError
from orion.rag.retriever import CodeRetriever
from orion.tools.base import FileChange, ToolContext, ToolResult, ToolSpec
from orion.tools.code import GitDiffTool, GitStatusTool, RunTestsTool, SearchCodeTool
from orion.tools.filesystem import (
    DeleteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from orion.tools.shell import RunCommandTool
from orion.utils import truncate

_TOOL_TYPES = (
    ReadFileTool,
    ListDirTool,
    GlobTool,
    GrepTool,
    WriteFileTool,
    EditFileTool,
    DeleteFileTool,
    MoveFileTool,
    RunCommandTool,
    RunTestsTool,
    SearchCodeTool,
    GitStatusTool,
    GitDiffTool,
)


class ToolRegistry:
    """Owns tool instances and enforces plan/build permissions."""

    def __init__(self, config: OrionConfig, retriever: CodeRetriever | None = None) -> None:
        self.config = config
        self.retriever = retriever
        self._tools: dict[str, Any] = {}
        for tool_type in _TOOL_TYPES:
            tool = tool_type()
            self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, mode: Mode = "build") -> list[ToolSpec]:
        if mode == "plan":
            return [tool.spec for tool in self._tools.values() if tool.spec.read_only]
        return [tool.spec for tool in self._tools.values()]

    def schemas(self, mode: Mode = "build") -> list[dict[str, Any]]:
        return [spec.as_schema() for spec in self.specs(mode)]

    def execute(self, name: str, args: dict[str, Any] | None, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(
                f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}"
            )
        if ctx.mode == "plan" and not tool.spec.read_only:
            return ToolResult.error(
                f"Tool '{name}' is not available in plan mode. Switch to build mode to "
                "modify the workspace."
            )
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return ToolResult.error("Tool arguments must be a JSON object")
        try:
            result = tool.run(args, ctx)
        except ToolError as exc:
            return ToolResult.error(str(exc))
        except OrionError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:
            return ToolResult.error(f"{type(exc).__name__}: {exc}")
        return result

    def make_context(
        self,
        *,
        mode: Mode,
        approval: Any | None = None,
        dry_run: bool = False,
        emit: Any | None = None,
    ) -> ToolContext:
        return ToolContext(
            config=self.config,
            mode=mode,
            approval=approval,
            dry_run=dry_run,
            emit=emit,
            retriever=self.retriever,
        )


def changes_diff(changes: list[FileChange], max_chars: int = 30_000) -> str:
    """Concatenate recorded diffs for review prompts."""
    if not changes:
        return ""
    blocks = []
    for change in changes:
        if change.action == "move":
            blocks.append(f"# {change.path} -> {change.destination}\n(rename)")
            continue
        if change.diff:
            blocks.append(f"# {change.path} ({change.action})\n{change.diff}")
        else:
            blocks.append(f"# {change.path} ({change.action})")
    return truncate("\n\n".join(blocks), max_chars, "diff")


def changes_summary(changes: list[FileChange]) -> list[dict[str, str]]:
    return [
        {
            "path": change.path,
            "action": change.action,
            "destination": change.destination or "",
            "lines": str(len(change.diff.splitlines())),
        }
        for change in changes
    ]


__all__ = ["ToolRegistry", "changes_diff", "changes_summary"]
