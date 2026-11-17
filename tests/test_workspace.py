"""Workspace scanning, path safety, and diff helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.errors import ToolError
from orion.workspace import (
    IgnoreRules,
    build_tree,
    detect_test_command,
    git_info,
    is_binary,
    is_source_file,
    iter_source_files,
    language_for,
    project_overview,
    read_text_file,
    relative_posix,
    safe_path,
    truncate,
    unified_diff,
)


def test_safe_path_allows_inside(workspace: Path) -> None:
    target = safe_path(workspace, "pkg/app.py")
    assert target == (workspace / "pkg" / "app.py").resolve()


@pytest.mark.parametrize("candidate", ["../outside.txt", "pkg/../../outside.txt"])
def test_safe_path_rejects_escapes(workspace: Path, candidate: str) -> None:
    with pytest.raises(ToolError):
        safe_path(workspace, candidate)


def test_ignore_rules_and_negation(workspace: Path) -> None:
    (workspace / ".orionignore").write_text("*.log\n!keep.log\nbuild/\n", encoding="utf-8")
    rules = IgnoreRules(workspace, patterns=[])
    assert rules.is_ignored("debug.log")
    assert not rules.is_ignored("keep.log")
    assert rules.is_ignored("build", is_dir=True)
    assert not rules.is_ignored("build", is_dir=False)


def test_iter_source_files_skips_env_and_ignored(workspace: Path) -> None:
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "left-pad.js").write_text("module.exports=1\n", encoding="utf-8")
    (workspace / "logo.bin").write_bytes(b"\x00\x01\x02")
    files = {relative_posix(path, workspace) for path in iter_source_files(workspace)}
    assert "pkg/app.py" in files
    assert "README.md" in files
    assert ".env" not in files
    assert "node_modules/left-pad.js" not in files
    assert "logo.bin" not in files


def test_language_and_source_detection() -> None:
    assert language_for("app.py") == "python"
    assert language_for("ui.tsx") == "typescript"
    assert language_for("Dockerfile") == "dockerfile"
    assert language_for("Makefile") == "makefile"
    assert is_source_file("pyproject.toml")
    assert is_source_file("Dockerfile")
    assert not is_source_file("image.png")


def test_binary_and_text_readers(workspace: Path) -> None:
    binary = workspace / "data.bin"
    binary.write_bytes(b"\x00\x01\x02\x03")
    assert is_binary(binary)
    assert not is_binary(workspace / "pkg" / "app.py")
    text = read_text_file(workspace / "pkg" / "app.py", max_bytes=50)
    assert len(text.encode("utf-8")) <= 50


def test_unified_diff_and_truncate() -> None:
    diff = unified_diff("a\nb\n", "a\nc\n", "file.txt")
    assert "-b" in diff
    assert "+c" in diff
    assert diff.startswith("--- a/file.txt")

    long_text = "x" * 5000
    trimmed = truncate(long_text, 1000, "text")
    assert len(trimmed) < len(long_text)
    assert "omitted" in trimmed


def test_build_tree_and_overview(workspace: Path) -> None:
    tree = build_tree(["src/app.py", "src/lib/util.py", "README.md"])
    assert any("src/" in line for line in tree)
    overview = project_overview(workspace)
    assert "Repository:" in overview
    assert "pkg/" in overview
    assert "app.py" in overview
    assert "pytest" in overview.lower()


def test_detect_test_command(workspace: Path) -> None:
    command, framework = detect_test_command(workspace)
    assert command is not None
    assert framework == "pytest"


def test_detect_test_command_from_loose_test_files(tmp_path: Path) -> None:
    (tmp_path / "test_greeter.py").write_text(
        "def test_greet():\n    assert True\n", encoding="utf-8"
    )
    command, framework = detect_test_command(tmp_path)
    assert command == "python -m pytest -q"
    assert framework == "pytest"


def test_detect_test_command_from_requirements(tmp_path: Path) -> None:
    (tmp_path / "requirements-dev.txt").write_text("pytest>=8\n", encoding="utf-8")
    command, framework = detect_test_command(tmp_path)
    assert command == "python -m pytest -q"
    assert framework == "pytest"


def test_git_info_without_repository(tmp_path: Path) -> None:
    info = git_info(tmp_path)
    assert info["available"] is False
    assert info["dirty"] is False
