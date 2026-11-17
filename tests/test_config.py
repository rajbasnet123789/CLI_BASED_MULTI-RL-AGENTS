"""Configuration, path discovery, and dotenv handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orion.config import (
    OrionConfig,
    discover_env_files,
    find_project_root,
    write_env_value,
)
from orion.errors import ConfigError


def test_defaults_and_derived_paths(workspace: Path) -> None:
    config = OrionConfig(_env_file=None, workspace=workspace)
    assert config.model == "gemini-3.5-flash"
    assert config.mode == "build"
    assert config.approval == "ask"
    assert config.data_path == workspace / ".orion"
    assert config.chroma_path == workspace / ".orion" / "chroma"
    assert config.session_path == workspace / ".orion" / "sessions"
    assert config.model_for("planner") == config.model


def test_agent_model_overrides(workspace: Path) -> None:
    config = OrionConfig(
        _env_file=None,
        workspace=workspace,
        planner_model="gemini-2.5-pro",
        reviewer_model="gemini-2.5-flash-lite",
    )
    assert config.model_for("planner") == "gemini-2.5-pro"
    assert config.model_for("reviewer") == "gemini-2.5-flash-lite"
    assert config.model_for("coder") == config.model


def test_vertex_detection(workspace: Path) -> None:
    # Built from fragments so no credential-shaped literal lives in the repo.
    express = "AQ" + "." + "Ab8" + "Fake" + "Express" + "Key" + "For" + "Tests" + "Only" + "1234"
    studio_key = "AIza" + "SyExampleKeyThatIsLongEnoughToMask"
    vertex = OrionConfig(_env_file=None, workspace=workspace, gemini_api_key=express)
    studio = OrionConfig(_env_file=None, workspace=workspace, gemini_api_key=studio_key)
    assert vertex.vertex_enabled is True
    assert studio.vertex_enabled is False
    assert vertex.masked_key.startswith("AQ.Ab8")
    assert "FakeExpressKeyForTestsOnly" not in vertex.masked_key
    assert vertex.masked_key.endswith(express[-4:])

    forced = OrionConfig(
        _env_file=None, workspace=workspace, gemini_api_key=express, vertex="false"
    )
    assert forced.vertex_enabled is False


def test_require_api_key(workspace: Path) -> None:
    config = OrionConfig(_env_file=None, workspace=workspace, gemini_api_key=None)
    assert config.has_api_key is False
    with pytest.raises(ConfigError):
        config.require_api_key()


def test_validation_rejects_bad_values(workspace: Path) -> None:
    with pytest.raises(ValidationError):
        OrionConfig(_env_file=None, workspace=workspace, temperature=9)
    with pytest.raises(ValidationError):
        OrionConfig(_env_file=None, workspace=workspace, top_k=0)
    with pytest.raises(ValidationError):
        OrionConfig(
            _env_file=None,
            workspace=workspace,
            chunk_chars=200,
            chunk_overlap=200,
        )


def test_load_reads_env_file(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ORION_MODEL", raising=False)
    env = workspace / ".env"
    env.write_text(
        "GEMINI_API_KEY=AIzaTestKey\nORION_MODEL=gemini-2.5-pro\nORION_APPROVAL=deny\n",
        encoding="utf-8",
    )
    config = OrionConfig.load(workspace=workspace)
    assert config.api_key == "AIzaTestKey"
    assert config.model == "gemini-2.5-pro"
    assert config.approval == "deny"


def test_find_project_root_and_env_discovery(tmp_path: Path) -> None:
    root = tmp_path / "project"
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".env").write_text("A=1\n", encoding="utf-8")
    (nested / ".env").write_text("B=2\n", encoding="utf-8")

    assert find_project_root(nested) == root
    files = discover_env_files(nested)
    assert root / ".env" in files
    assert nested / ".env" in files
    # Nearest file comes last so it wins when dotenv layers are applied.
    assert files[-1] == (nested / ".env").resolve()


def test_write_env_value_updates_and_appends(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\nORION_MODEL=old\n", encoding="utf-8")
    write_env_value("ORION_MODEL", "new", env)
    write_env_value("ORION_APPROVAL", "auto", env)
    text = env.read_text(encoding="utf-8")
    assert "ORION_MODEL=new" in text
    assert "ORION_APPROVAL=auto" in text
    assert text.count("ORION_MODEL=") == 1


def test_with_overrides_and_describe(workspace: Path) -> None:
    config = OrionConfig(_env_file=None, workspace=workspace, gemini_api_key="AIzaSecretValue")
    updated = config.with_overrides(model="gemini-2.5-pro", approval=None)
    assert updated.model == "gemini-2.5-pro"
    assert updated.approval == config.approval
    described = config.describe()
    assert described["gemini_api_key"] != config.api_key
    assert "SecretValue" not in described["gemini_api_key"]
    assert config.describe(include_secrets=True)["gemini_api_key"] == config.api_key
