"""Shared fixtures for the Orion test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.config import OrionConfig

APP_SOURCE = '''"""Tiny sample application used by tests."""

DATABASE_URL = "sqlite:///example.db"


def connect() -> str:
    """Return the configured database connection string."""
    return DATABASE_URL


def add(left: int, right: int) -> int:
    return left + right
'''


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (tmp_path / "pkg" / "utils.py").write_text(
        "def slugify(value: str) -> str:\n    return value.lower().replace(' ', '-')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "from pkg.app import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Sample\n\nA tiny project.\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text("SECRET=should-not-be-indexed\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def config(workspace: Path) -> OrionConfig:
    return OrionConfig(
        _env_file=None,
        workspace=workspace,
        embeddings="local",
        auto_index=False,
        approval="auto",
        session_persist=False,
        chunk_chars=300,
        chunk_overlap=40,
    )
