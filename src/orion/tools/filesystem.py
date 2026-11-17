"""Filesystem tools. All paths are sandboxed to the workspace."""

from __future__ import annotations

import fnmatch
import glob as glob_module
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from orion.errors import ToolError
from orion.tools.base import (
    FileChange,
    ToolContext,
    ToolResult,
    ToolSpec,
    as_bool,
    as_int,
    as_str,
)
from orion.utils import truncate
from orion.workspace import (
    IgnoreRules,
    human_size,
    is_binary,
    read_text_file,
    relative_posix,
    safe_path,
    unified_diff,
)

MAX_READ_LINES = 500
MAX_READ_CHARS = 60_000
MAX_WRITE_BYTES = 2_000_000


def _backup(ctx: ToolContext, path: Path) -> None:
    if not ctx.config.backup or not path.is_file():
        return
    rel = relative_posix(path, ctx.workspace)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = ctx.config.backup_path / stamp / rel
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    except OSError:
        pass


def _commit_write(ctx: ToolContext, path: Path, before: str, after: str, action: str) -> ToolResult:
    rel = relative_posix(path, ctx.workspace)
    diff = unified_diff(before, after, rel)
    if before == after:
        return ToolResult.success(f"{rel} is already up to date; no changes made.", display="unchanged")
    if not ctx.apply_changes:
        change = FileChange(path=rel, action=action, before=before, after=after, diff=diff)
        ctx.changes.append(change)
        return ToolResult.success(
            f"[dry run] Would {action} {rel} ({len(diff.splitlines())} diff lines).",
            diff=diff,
            changed_paths=[rel],
            display=f"[dry run] {action} {rel}",
        )
    _backup(ctx, path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"Could not write {rel}: {exc}") from exc
    change = FileChange(path=rel, action=action, before=before, after=after, diff=diff)
    ctx.changes.append(change)
    return ToolResult.success(
        f"{action.title()}d {rel} ({len(diff.splitlines())} diff lines).",
        diff=diff,
        changed_paths=[rel],
        display=f"{action} {rel}",
    )


class ReadFileTool:
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the workspace. Returns numbered lines. "
            "Use start_line/end_line for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "First line (1-based)."},
                "end_line": {"type": "integer", "description": "Last line (inclusive)."},
            },
            "required": ["path"],
        },
        read_only=True,
        category="files",
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = safe_path(ctx.workspace, as_str(args.get("path")))
        if not path.exists():
            return ToolResult.error(f"{as_str(args.get('path'))} does not exist")
        if path.is_dir():
            return ToolResult.error(f"{as_str(args.get('path'))} is a directory; use list_dir")
        if is_binary(path):
            return ToolResult.error(f"{as_str(args.get('path'))} looks like a binary file")
        text = read_text_file(path, max_bytes=ctx.config.max_file_bytes)
        lines = text.splitlines()
        total = len(lines)
        start = max(1, as_int(args.get("start_line"), 1))
        end = as_int(args.get("end_line"), 0) or total
        end = max(start, min(end, total))
        if end - start + 1 > MAX_READ_LINES:
            end = start + MAX_READ_LINES - 1
        numbered = "\n".join(
            f"{number:>5} | {lines[number - 1]}" for number in range(start, end + 1)
        )
        body = truncate(numbered, MAX_READ_CHARS, "file contents")
        rel = relative_posix(path, ctx.workspace)
        return ToolResult.success(
            f"{rel} (lines {start}-{end} of {total}):\n{body}",
            data={"path": rel, "start_line": start, "end_line": end, "total_lines": total},
            display=f"read {rel}:{start}-{end}",
        )


class ListDirTool:
    spec = ToolSpec(
        name="list_dir",
        description="List files and directories inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to the workspace."},
                "depth": {"type": "integer", "description": "Recursion depth, 1-3."},
            },
            "required": [],
        },
        read_only=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = as_str(args.get("path"), ".") or "."
        root = safe_path(ctx.workspace, raw)
        if not root.is_dir():
            return ToolResult.error(f"{raw} is not a directory")
        depth = max(1, min(3, as_int(args.get("depth"), 1)))
        rules = IgnoreRules(ctx.workspace, ctx.config.ignore)
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            level = len(current.relative_to(root).parts)
            if level >= depth:
                dirnames[:] = []
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not rules.is_ignored(
                    relative_posix(current / name, ctx.workspace), is_dir=True
                )
            ]
            prefix = "  " * level
            for name in sorted(dirnames):
                entries.append(f"{prefix}{name}/")
            for name in sorted(filenames):
                path = current / name
                rel = relative_posix(path, ctx.workspace)
                if rules.is_ignored(rel):
                    continue
                try:
                    size = human_size(path.stat().st_size)
                except OSError:
                    size = "?"
                entries.append(f"{prefix}{name}  ({size})")
            if len(entries) >= 400:
                entries.append("... (truncated)")
                break
        rel_root = relative_posix(root, ctx.workspace) or "."
        body = "\n".join(entries) or "(empty directory)"
        return ToolResult.success(
            f"{rel_root}:\n{body}",
            data={"path": rel_root, "entries": len(entries)},
            display=f"list {rel_root}",
        )


class GlobTool:
    spec = ToolSpec(
        name="glob",
        description=(
            "Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.ts'. "
            "Patterns are relative to the workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern."},
                "limit": {"type": "integer", "description": "Maximum results (default 100)."},
            },
            "required": ["pattern"],
        },
        read_only=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = as_str(args.get("pattern")).replace("\\", "/").lstrip("/")
        if not pattern or ".." in Path(pattern).parts:
            return ToolResult.error("pattern must be a relative glob without '..'")
        limit = max(1, min(500, as_int(args.get("limit"), 100)))
        matches = glob_module.glob(str(ctx.workspace / pattern), recursive=True)
        rules = IgnoreRules(ctx.workspace, ctx.config.ignore)
        files: list[str] = []
        for match in sorted(matches):
            path = Path(match)
            try:
                resolved = path.resolve()
                if resolved != ctx.workspace and ctx.workspace not in resolved.parents:
                    continue
            except OSError:
                continue
            rel = relative_posix(path, ctx.workspace)
            if rules.is_ignored(rel, is_dir=path.is_dir()):
                continue
            if path.is_file():
                files.append(rel)
            if len(files) >= limit:
                break
        body = "\n".join(files) if files else "(no matches)"
        return ToolResult.success(
            f"{len(files)} match(es) for {pattern}:\n{body}",
            data={"pattern": pattern, "matches": files},
            display=f"glob {pattern} ({len(files)})",
        )


class GrepTool:
    spec = ToolSpec(
        name="grep",
        description=(
            "Search file contents with a regular expression. Returns matching "
            "file:line entries. Use include to restrict to a glob."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "include": {"type": "string", "description": "Glob filter, e.g. '*.py'."},
                "case_sensitive": {"type": "boolean", "description": "Default false."},
                "limit": {"type": "integer", "description": "Maximum matches (default 100)."},
            },
            "required": ["pattern"],
        },
        read_only=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_pattern = as_str(args.get("pattern"))
        if not raw_pattern:
            return ToolResult.error("pattern is required")
        flags = 0 if as_bool(args.get("case_sensitive"), False) else re.IGNORECASE
        try:
            regex = re.compile(raw_pattern, flags)
        except re.error:
            regex = re.compile(re.escape(raw_pattern), flags)
        include = as_str(args.get("include")) or None
        limit = max(1, min(500, as_int(args.get("limit"), 100)))
        rules = IgnoreRules(ctx.workspace, ctx.config.ignore)
        matches: list[str] = []
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(ctx.workspace):
            current = Path(dirpath)
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not rules.is_ignored(
                    relative_posix(current / name, ctx.workspace), is_dir=True
                )
            ]
            for name in sorted(filenames):
                path = current / name
                rel = relative_posix(path, ctx.workspace)
                if rules.is_ignored(rel):
                    continue
                if include and not (
                    fnmatch.fnmatch(name, include) or fnmatch.fnmatch(rel, include)
                ):
                    continue
                try:
                    if path.stat().st_size > ctx.config.max_file_bytes:
                        continue
                except OSError:
                    continue
                if is_binary(path):
                    continue
                scanned += 1
                try:
                    text = read_text_file(path, max_bytes=ctx.config.max_file_bytes)
                except ToolError:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append(f"{rel}:{number}: {truncate(line.strip(), 300, 'line')}")
                        if len(matches) >= limit:
                            break
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break
        body = "\n".join(matches) if matches else "(no matches)"
        return ToolResult.success(
            f"{len(matches)} match(es) for /{raw_pattern}/ in {scanned} file(s):\n{body}",
            data={"pattern": raw_pattern, "matches": matches},
            display=f"grep /{raw_pattern}/ ({len(matches)})",
        )


class WriteFileTool:
    spec = ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a file with complete contents. Parent directories "
            "are created automatically. Prefer edit_file for small changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            "required": ["path", "content"],
        },
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        raw = as_str(args.get("path"))
        path = safe_path(ctx.workspace, raw)
        if path.is_dir():
            return ToolResult.error(f"{raw} is a directory")
        content = as_str(args.get("content"))
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            return ToolResult.error(
                f"content is {human_size(len(encoded))}; the limit is {human_size(MAX_WRITE_BYTES)}"
            )
        before = ""
        action = "create"
        if path.is_file():
            before = read_text_file(path, max_bytes=MAX_WRITE_BYTES)
            action = "update"
        return _commit_write(ctx, path, before, content, action)


class EditFileTool:
    spec = ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact string in an existing file. old_string must match "
            "exactly (including whitespace) and be unique unless replace_all is true."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        raw = as_str(args.get("path"))
        path = safe_path(ctx.workspace, raw)
        if not path.is_file():
            return ToolResult.error(f"{raw} does not exist; use write_file to create it")
        old_string = as_str(args.get("old_string"))
        new_string = as_str(args.get("new_string"))
        if not old_string:
            return ToolResult.error("old_string is required")
        if old_string == new_string:
            return ToolResult.error("old_string and new_string are identical")
        before = read_text_file(path, max_bytes=MAX_WRITE_BYTES)
        count = before.count(old_string)
        replace_all = as_bool(args.get("replace_all"), False)
        if count == 0:
            return ToolResult.error(
                f"old_string was not found in {raw}. Read the file again and copy the "
                "exact text, including indentation."
            )
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string occurs {count} times in {raw}. Add surrounding context to make "
                "it unique, or set replace_all=true."
            )
        after = before.replace(old_string, new_string, -1 if replace_all else 1)
        return _commit_write(ctx, path, before, after, "update")


class DeleteFileTool:
    spec = ToolSpec(
        name="delete_file",
        description="Delete a file or an empty directory. Requires approval.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "recursive": {
                    "type": "boolean",
                    "description": "Delete a directory and all of its contents.",
                },
            },
            "required": ["path"],
        },
        requires_approval=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        raw = as_str(args.get("path"))
        path = safe_path(ctx.workspace, raw)
        if path == ctx.workspace:
            return ToolResult.error("refusing to delete the workspace root")
        if not path.exists():
            return ToolResult.error(f"{raw} does not exist")
        recursive = as_bool(args.get("recursive"), False)
        rel = relative_posix(path, ctx.workspace)
        approved = ctx.request_approval(
            self.spec.name,
            "delete_file",
            f"Delete `{rel}`{' and everything inside it' if recursive else ''}.",
            risk="high",
        )
        if not approved:
            return ToolResult.error("deletion denied by the user")
        before = "" if path.is_dir() else read_text_file(path, max_bytes=MAX_WRITE_BYTES)
        if not ctx.apply_changes:
            diff = unified_diff(before, "", rel)
            ctx.changes.append(
                FileChange(path=rel, action="delete", before=before, after=None, diff=diff)
            )
            return ToolResult.success(
                f"[dry run] Would delete {rel}", diff=diff, changed_paths=[rel]
            )
        try:
            if path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            else:
                path.unlink()
        except OSError as exc:
            return ToolResult.error(f"could not delete {rel}: {exc}")
        diff = unified_diff(before, "", rel)
        ctx.changes.append(
            FileChange(path=rel, action="delete", before=before, after=None, diff=diff)
        )
        return ToolResult.success(
            f"Deleted {rel}", diff=diff, changed_paths=[rel], display=f"delete {rel}"
        )


class MoveFileTool:
    spec = ToolSpec(
        name="move_file",
        description="Move or rename a file or directory inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Current path."},
                "destination": {"type": "string", "description": "New path."},
            },
            "required": ["source", "destination"],
        },
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.require_write()
        source = safe_path(ctx.workspace, as_str(args.get("source")))
        destination = safe_path(ctx.workspace, as_str(args.get("destination")))
        if not source.exists():
            return ToolResult.error(f"{as_str(args.get('source'))} does not exist")
        if destination.exists():
            return ToolResult.error(f"{as_str(args.get('destination'))} already exists")
        if source == ctx.workspace:
            return ToolResult.error("refusing to move the workspace root")
        src_rel = relative_posix(source, ctx.workspace)
        dst_rel = relative_posix(destination, ctx.workspace)
        if not ctx.apply_changes:
            return ToolResult.success(
                f"[dry run] Would move {src_rel} -> {dst_rel}",
                changed_paths=[src_rel, dst_rel],
            )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except OSError as exc:
            return ToolResult.error(f"could not move {src_rel}: {exc}")
        ctx.changes.append(
            FileChange(path=src_rel, action="move", destination=dst_rel, diff=f"rename {src_rel} -> {dst_rel}")
        )
        return ToolResult.success(
            f"Moved {src_rel} -> {dst_rel}",
            changed_paths=[src_rel, dst_rel],
            display=f"move {src_rel} -> {dst_rel}",
        )


__all__ = [
    "DeleteFileTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ListDirTool",
    "MoveFileTool",
    "ReadFileTool",
    "WriteFileTool",
]
