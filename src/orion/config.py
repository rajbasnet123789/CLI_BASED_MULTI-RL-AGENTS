"""Configuration loading for Orion AI.

Settings come from (highest priority first):

1. Real process environment variables (``ORION_*`` and ``GEMINI_API_KEY``).
2. ``.env`` files, searched from the workspace up to the project root, then
   ``~/.orion/.env``.
3. Defaults declared on :class:`OrionConfig`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from orion.errors import ConfigError

Mode = Literal["build", "plan"]
ApprovalPolicy = Literal["ask", "auto", "deny"]
EmbeddingBackend = Literal["auto", "gemini", "local"]
VertexMode = Literal["auto", "true", "false"]
LLMProvider = Literal["gemini", "local"]

DEFAULT_IGNORE: list[str] = [
    ".git",
    ".hg",
    ".svn",
    ".orion",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "dist",
    "build",
    "target",
    "coverage",
    "htmlcov",
    ".next",
    ".nuxt",
    ".turbo",
    ".idea",
    ".vscode",
    "vendor",
    "Pods",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "Cargo.lock",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.whl",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.bin",
    "*.pyc",
]

PROJECT_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")


def find_project_root(start: Path | None = None) -> Path:
    """Return the closest directory that looks like a project root."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


def discover_env_files(workspace: Path | None = None) -> list[Path]:
    """Collect existing ``.env`` files, farthest first so nearest wins."""
    workspace = (workspace or Path.cwd()).resolve()
    root = find_project_root(workspace)
    files: list[Path] = []

    home_env = Path.home() / ".orion" / ".env"
    if home_env.is_file():
        files.append(home_env)

    chain: list[Path] = []
    current = workspace
    while True:
        chain.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
    for directory in reversed(chain):
        env = directory / ".env"
        if env.is_file():
            files.append(env)

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


class OrionConfig(BaseSettings):
    """Effective Orion configuration.

    Instances are cheap; create one per run. Use :meth:`load` to pick up
    ``.env`` files automatically, or construct directly in library code.
    """

    model_config = SettingsConfigDict(
        env_prefix="ORION_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Credentials ---------------------------------------------------------
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "ORION_GEMINI_API_KEY",
        ),
    )
    vertex: VertexMode = "auto"

    # LLM provider --------------------------------------------------------
    # Gemini remains the default. "local" uses an OpenAI-compatible server such
    # as Ollama and needs no API key.
    llm_provider: LLMProvider = "gemini"
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen2.5:7b-instruct"
    local_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ORION_LOCAL_API_KEY", "LOCAL_API_KEY"),
    )
    local_timeout: float = 120.0

    # Models --------------------------------------------------------------
    model: str = "gemini-3.5-flash"
    planner_model: str | None = None
    coder_model: str | None = None
    reviewer_model: str | None = None
    embedding_model: str = "gemini-embedding-001"
    temperature: float = 0.2
    max_output_tokens: int = 8192

    # Agent behaviour -----------------------------------------------------
    mode: Mode = "build"
    approval: ApprovalPolicy = "ask"
    max_iterations: int = 3
    max_tool_rounds: int = 12
    dry_run: bool = False
    auto_index: bool = True
    session_persist: bool = True

    # Retrieval -----------------------------------------------------------
    embeddings: EmbeddingBackend = "auto"
    top_k: int = 8
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    max_file_bytes: int = 512_000
    ignore: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORE))

    # Paths ---------------------------------------------------------------
    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path | None = None
    chroma_dir: Path | None = None
    backup: bool = True

    # Runtime -------------------------------------------------------------
    theme: str = "orion"
    log_level: str = "WARNING"
    command_timeout: int = 120

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, value: int) -> int:
        if not 1 <= value <= 50:
            raise ValueError("top_k must be between 1 and 50")
        return value

    @field_validator("max_output_tokens", "max_iterations", "max_tool_rounds")
    @classmethod
    def _check_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("local_timeout")
    @classmethod
    def _check_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("local_timeout must be positive")
        return value

    @model_validator(mode="after")
    def _check_chunks(self) -> OrionConfig:
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        if self.chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        return self

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        workspace: str | Path | None = None,
        env_file: str | Path | None = None,
        **overrides: Any,
    ) -> OrionConfig:
        """Build a config, reading ``.env`` files from the workspace chain."""
        workspace_path = Path(workspace).expanduser().resolve() if workspace else Path.cwd()
        if env_file is not None:
            files: tuple[Path, ...] = (Path(env_file).expanduser().resolve(),)
        else:
            files = tuple(discover_env_files(workspace_path))
        try:
            return cls(_env_file=files or None, workspace=workspace_path, **overrides)
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"Invalid Orion configuration: {exc}") from exc

    def with_overrides(self, **overrides: Any) -> OrionConfig:
        """Return a copy with selected fields replaced."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        if not clean:
            return self
        return self.model_copy(update=clean)

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str | None:
        if self.gemini_api_key is None:
            return None
        value = self.gemini_api_key.get_secret_value().strip()
        return value or None

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def is_local(self) -> bool:
        return self.llm_provider == "local"

    @property
    def local_api_key_value(self) -> str | None:
        if self.local_api_key is None:
            return None
        value = self.local_api_key.get_secret_value().strip()
        return value or None

    @property
    def has_credentials(self) -> bool:
        """Local models need no key; Gemini does."""
        return self.is_local or self.has_api_key

    @property
    def masked_key(self) -> str:
        key = self.api_key
        if not key:
            return "(not set)"
        if len(key) <= 10:
            return key[:2] + "..." + key[-2:]
        return f"{key[:6]}...{key[-4:]}"

    @property
    def vertex_enabled(self) -> bool:
        if self.vertex == "true":
            return True
        if self.vertex == "false":
            return False
        key = self.api_key or ""
        return key.startswith("AQ.")

    @property
    def project_root(self) -> Path:
        return find_project_root(self.workspace)

    @property
    def data_path(self) -> Path:
        if self.data_dir is not None:
            path = Path(self.data_dir).expanduser()
            return path if path.is_absolute() else (self.workspace / path)
        return self.workspace / ".orion"

    @property
    def chroma_path(self) -> Path:
        if self.chroma_dir is not None:
            path = Path(self.chroma_dir).expanduser()
            return path if path.is_absolute() else (self.workspace / path)
        return self.data_path / "chroma"

    @property
    def session_path(self) -> Path:
        return self.data_path / "sessions"

    @property
    def backup_path(self) -> Path:
        return self.data_path / "backups"

    @property
    def manifest_path(self) -> Path:
        return self.data_path / "index_manifest.json"

    def model_for(self, agent: str) -> str:
        """Model override for a specific agent, falling back to the global model."""
        mapping = {
            "planner": self.planner_model,
            "coder": self.coder_model,
            "reviewer": self.reviewer_model,
        }
        return mapping.get(agent) or self.model

    def ensure_dirs(self) -> None:
        for path in (self.data_path, self.session_path, self.chroma_path):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self, include_secrets: bool = False) -> dict[str, Any]:
        """Serializable view used by ``orion config`` and the server."""
        data: dict[str, Any] = {
            "llm_provider": self.llm_provider,
            "model": self.model,
            "local_base_url": self.local_base_url,
            "local_model": self.local_model,
            "has_credentials": self.has_credentials,
            "planner_model": self.model_for("planner"),
            "coder_model": self.model_for("coder"),
            "reviewer_model": self.model_for("reviewer"),
            "embedding_model": self.embedding_model,
            "mode": self.mode,
            "approval": self.approval,
            "vertex_setting": self.vertex,
            "vertex": self.vertex_enabled,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_iterations": self.max_iterations,
            "max_tool_rounds": self.max_tool_rounds,
            "embeddings": self.embeddings,
            "top_k": self.top_k,
            "auto_index": self.auto_index,
            "dry_run": self.dry_run,
            "workspace": str(self.workspace),
            "project_root": str(self.project_root),
            "data_dir": str(self.data_path),
            "chroma_dir": str(self.chroma_path),
            "gemini_api_key": self.api_key if include_secrets else self.masked_key,
        }
        return data

    def require_api_key(self) -> str:
        key = self.api_key
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Add it to .env or export it:\n"
                "  GEMINI_API_KEY=your-key   # https://aistudio.google.com/apikey\n"
                "Or run fully offline with ORION_LLM_PROVIDER=local and Ollama."
            )
        return key

    def require_credentials(self) -> str:
        """Return the identifier for the active provider, or raise a clear error."""
        if self.is_local:
            return self.local_model
        return self.require_api_key()

    def env_hint(self) -> str:
        files = discover_env_files(self.workspace)
        if files:
            return ", ".join(str(path) for path in files)
        return str(self.workspace / ".env")


def write_env_value(key: str, value: str, path: Path | None = None) -> Path:
    """Set ``KEY=value`` in a dotenv file, preserving unrelated lines."""
    target = path or (Path.cwd() / ".env")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if target.is_file():
        lines = target.read_text(encoding="utf-8").splitlines()
    needle = f"{key}="
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(needle) or line.strip().startswith(f"export {needle}"):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - Windows / restricted FS
        pass
    return target


__all__ = [
    "DEFAULT_IGNORE",
    "ApprovalPolicy",
    "EmbeddingBackend",
    "Mode",
    "OrionConfig",
    "VertexMode",
    "discover_env_files",
    "find_project_root",
    "write_env_value",
]
