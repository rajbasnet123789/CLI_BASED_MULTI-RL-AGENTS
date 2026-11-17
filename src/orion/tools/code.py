"""Code-aware tools: retrieval, git inspection, and test execution."""

from __future__ import annotations

from typing import Any

from orion.tools.base import ToolContext, ToolResult, ToolSpec, as_bool, as_int, as_str
from orion.tools.shell import run_shell_command
from orion.utils import truncate
from orion.workspace import detect_test_command, safe_path


class SearchCodeTool:
    spec = ToolSpec(
        name="search_code",
        description=(
            "Semantic + lexical search over the repository index. Use this before "
            "editing to find relevant files, symbols, and patterns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code query."},
                "k": {"type": "integer", "description": "Number of results (default 8)."},
            },
            "required": ["query"],
        },
        read_only=True,
        category="search",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.retriever is None:
            return ToolResult.error("code search is unavailable (no index)")
        query = as_str(args.get("query")).strip()
        if not query:
            return ToolResult.error("query is required")
        k = max(1, min(20, as_int(args.get("k"), ctx.config.top_k)))
        hits = ctx.retriever.search(query, k=k)
        if not hits:
            return ToolResult.success(
                f"No matches for {query!r}. The index may be empty; try `orion index`.",
                display=f"search {truncate(query, 60)} (0)",
            )
        lines = [f"{len(hits)} result(s) for {query!r}:"]
        for hit in hits:
            snippet = truncate(hit.text, 1200, hit.location)
            lines.append(
                f"\n### {hit.location}  (score {hit.score:.2f}, {hit.source})\n"
                f"```{hit.language}\n{snippet}\n```"
            )
        return ToolResult.success(
            "\n".join(lines),
            data={"query": query, "hits": [hit.as_dict() for hit in hits]},
            display=f"search {truncate(query, 60)} ({len(hits)})",
        )


class GitStatusTool:
    spec = ToolSpec(
        name="git_status",
        description="Show git branch and working-tree status.",
        parameters={"type": "object", "properties": {}, "required": []},
        read_only=True,
        category="git",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code, output = run_shell_command(
            "git status --short --branch", cwd=ctx.workspace, timeout=20, max_output=8000
        )
        if code != 0:
            return ToolResult.error(output)
        return ToolResult.success(output, display="git status")


class GitDiffTool:
    spec = ToolSpec(
        name="git_diff",
        description="Show unstaged or staged git changes as a diff.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional file or directory."},
                "staged": {"type": "boolean", "description": "Show staged changes."},
            },
            "required": [],
        },
        read_only=True,
        category="git",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        staged = as_bool(args.get("staged"), False)
        command = "git diff --no-color"
        if staged:
            command += " --cached"
        raw_path = as_str(args.get("path")).strip()
        if raw_path:
            target = safe_path(ctx.workspace, raw_path)
            command += f' -- "{target}"'
        code, output = run_shell_command(command, cwd=ctx.workspace, timeout=30, max_output=25_000)
        if code != 0:
            return ToolResult.error(output)
        if not output.strip() or output.strip() == "(no output)":
            return ToolResult.success("No changes.", display="git diff (clean)")
        return ToolResult.success(output, display="git diff")


class RunTestsTool:
    spec = ToolSpec(
        name="run_tests",
        description=(
            "Detect and run the project's test suite. Requires approval. Returns the "
            "exit code and captured output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Optional command override; auto-detected when omitted.",
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            "required": [],
        },
        requires_approval=True,
        category="shell",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        command = as_str(args.get("command")).strip()
        framework = None
        if not command:
            command, framework = detect_test_command(ctx.workspace)
        if not command:
            return ToolResult.error(
                "Could not detect a test command. Pass `command` explicitly, e.g. "
                "`python -m pytest -q`."
            )
        timeout = max(10, min(1800, as_int(args.get("timeout"), 300)))
        approved = ctx.request_approval(
            self.spec.name,
            "run_tests",
            f"Run the test suite in `{ctx.workspace.name}`:\n\n  {command}",
            risk="medium",
        )
        if not approved:
            return ToolResult.error("test run denied by the user")
        code, output = run_shell_command(command, cwd=ctx.workspace, timeout=timeout)
        return ToolResult(
            ok=code == 0,
            content=f"command: {command}\nexit code: {code}\n\n{output}",
            data={
                "command": command,
                "framework": framework,
                "exit_code": code,
                "passed": code == 0,
            },
            display=f"tests: {command} (exit {code})",
        )


__all__ = ["GitDiffTool", "GitStatusTool", "RunTestsTool", "SearchCodeTool"]
