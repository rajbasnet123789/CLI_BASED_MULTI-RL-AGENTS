"""Shell execution with safety rails and approvals."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from orion.tools.base import ToolContext, ToolResult, ToolSpec, as_int, as_str
from orion.utils import truncate
from orion.workspace import relative_posix, safe_path

DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/(?:\s|$|\*)"), "recursive delete of /"),
    (re.compile(r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+~(?:\s|$|/)"), "recursive delete of the home directory"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b"), "formatting a filesystem"),
    (re.compile(r"\bdd\s+if=.*\bof=/dev/"), "raw disk writes"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "shutting down the machine"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$)"), "chmod 777 on /"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sh|bash|zsh)\b"), "piping a download into a shell"),
    (re.compile(r"\bformat\s+[a-zA-Z]:"), "formatting a drive"),
    (re.compile(r"\bdel\s+/[fsq].*[a-zA-Z]:\\\s*$", re.IGNORECASE), "recursive delete of a drive"),
    (
        re.compile(r"\bRemove-Item\b.*-Recurse.*\b(C:\\|/)\s*$", re.IGNORECASE),
        "recursive delete of a drive root",
    ),
]


def find_dangerous(command: str) -> str | None:
    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def run_shell_command(
    command: str,
    *,
    cwd: Path,
    timeout: int = 120,
    max_output: int = 30_000,
) -> tuple[int, str]:
    """Run a command and return ``(exit_code, combined_output)``."""
    env = os.environ.copy()
    env.update(
        {
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "CI": "1",
        }
    )
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = ""
        if exc.stdout:
            partial += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(
                "utf-8", errors="replace"
            )
        if exc.stderr:
            partial += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(
                "utf-8", errors="replace"
            )
        return 124, f"Command timed out after {timeout}s.\n{truncate(partial, max_output)}"
    except OSError as exc:
        return 127, f"Could not start command: {exc}"

    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append("[stderr]\n" + proc.stderr.rstrip())
    output = "\n".join(parts).strip() or "(no output)"
    return proc.returncode, truncate(output, max_output)


class RunCommandTool:
    spec = ToolSpec(
        name="run_command",
        description=(
            "Run a shell command in the workspace. Requires user approval and is "
            "disabled in plan mode. Use for builds, tests, linters, and git."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 1800).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory relative to the workspace.",
                },
            },
            "required": ["command"],
        },
        requires_approval=True,
        category="shell",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        command = as_str(args.get("command")).strip()
        if not command:
            return ToolResult.error("command is required")
        danger = find_dangerous(command)
        if danger:
            return ToolResult.error(
                f"Blocked for safety ({danger}). If this is intentional, run it yourself "
                "outside Orion."
            )
        timeout = max(1, min(1800, as_int(args.get("timeout"), ctx.config.command_timeout)))
        raw_cwd = as_str(args.get("cwd"), ".") or "."
        workdir = safe_path(ctx.workspace, raw_cwd)
        if not workdir.is_dir():
            return ToolResult.error(f"{raw_cwd} is not a directory")
        rel_cwd = relative_posix(workdir, ctx.workspace) or "."
        approved = ctx.request_approval(
            self.spec.name,
            "run_command",
            f"Run in `{rel_cwd}` with a {timeout}s timeout:\n\n  {command}",
            risk="high",
        )
        if not approved:
            return ToolResult.error("command denied by the user")
        code, output = run_shell_command(command, cwd=workdir, timeout=timeout)
        ok = code == 0
        body = f"exit code: {code}\n\n{output}"
        return ToolResult(
            ok=ok,
            content=body,
            data={"command": command, "exit_code": code, "cwd": rel_cwd},
            display=f"$ {truncate(command, 90, 'command')} (exit {code})",
        )


__all__ = ["DANGEROUS_PATTERNS", "RunCommandTool", "find_dangerous", "run_shell_command"]
