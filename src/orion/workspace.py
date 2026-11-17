"""Workspace inspection helpers shared by the indexer, tools, and agents."""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from orion.config import DEFAULT_IGNORE
from orion.errors import ToolError
from orion.utils import truncate

SOURCE_EXTENSIONS: set[str] = {
    ".py", ".pyi", ".pyx", ".ipynb",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
    ".vue", ".svelte", ".astro",
    ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".clj",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx",
    ".cs", ".fs", ".vb", ".swift", ".m", ".mm",
    ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".jl", ".dart", ".ex", ".exs",
    ".erl", ".hrl", ".hs", ".lhs", ".elm", ".zig", ".nim", ".v",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto", ".thrift",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".styl",
    ".md", ".mdx", ".rst", ".txt", ".adoc",
    ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".xsl", ".csv", ".tsv", ".env.example", ".properties", ".gradle",
    ".tf", ".tfvars", ".hcl", ".dockerfile", ".gitignore", ".dockerignore",
    ".editorconfig", ".coveragerc", ".prettierrc", ".eslintrc",
}

SOURCE_NAMES: set[str] = {
    "Dockerfile", "Containerfile", "Makefile", "makefile", "GNUmakefile",
    "README", "README.md", "LICENSE", "LICENCE", "CHANGELOG", "CHANGELOG.md",
    "CONTRIBUTING.md", "CODEOWNERS", "Procfile", "Gemfile", "Rakefile",
    "CMakeLists.txt", "BUILD", "WORKSPACE", "Justfile", "Taskfile.yml",
}

LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyx": "cython", ".ipynb": "jupyter",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".vue": "vue", ".svelte": "svelte", ".astro": "astro",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".clj": "clojure",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp", ".fs": "fsharp", ".vb": "vbnet", ".swift": "swift",
    ".m": "objective-c", ".mm": "objective-c",
    ".rb": "ruby", ".php": "php", ".pl": "perl", ".pm": "perl", ".lua": "lua",
    ".r": "r", ".jl": "julia", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hrl": "erlang", ".hs": "haskell", ".lhs": "haskell",
    ".elm": "elm", ".zig": "zig", ".nim": "nim", ".v": "verilog",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "fish",
    ".ps1": "powershell", ".bat": "batch", ".cmd": "batch",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql", ".proto": "protobuf",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sass": "sass", ".less": "less", ".styl": "stylus",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".conf": "config",
    ".xml": "xml", ".xsl": "xml", ".csv": "csv", ".tsv": "csv",
    ".tf": "terraform", ".tfvars": "terraform", ".hcl": "hcl",
    ".dockerfile": "dockerfile", ".gitignore": "text", ".dockerignore": "text",
    ".editorconfig": "ini", ".properties": "properties", ".gradle": "groovy",
}

TEXT_SAMPLE_BYTES = 4096


def language_for(path: str | Path) -> str:
    """Best-effort language label for syntax highlighting and metadata."""
    p = Path(path)
    name = p.name.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"
    if name in {"makefile", "gnumakefile"}:
        return "makefile"
    if name.startswith(".") and name.count(".") == 1:
        return "text"
    return LANGUAGE_BY_EXT.get(p.suffix.lower(), "text")


def is_source_file(path: str | Path) -> bool:
    p = Path(path)
    if p.name in SOURCE_NAMES:
        return True
    if p.name.lower() in {name.lower() for name in SOURCE_NAMES}:
        return True
    return p.suffix.lower() in SOURCE_EXTENSIONS


def is_binary(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            sample = handle.read(TEXT_SAMPLE_BYTES)
    except OSError:
        return True
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        # Latin-1 files are still readable; reject only high binary ratios.
        printable = sum(byte in b"\n\r\t" or 32 <= byte < 127 for byte in sample)
        return printable / len(sample) < 0.75
    return False


def read_text_file(path: str | Path, max_bytes: int | None = None) -> str:
    """Read a text file with a few sensible fallbacks."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ToolError(f"Cannot read {p}: {exc}") from exc
    if max_bytes is not None and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class IgnoreRules:
    """Small, dependency-free matcher for ``.gitignore``-style patterns."""

    def __init__(self, root: Path, patterns: Iterable[str] | None = None) -> None:
        self.root = Path(root).resolve()
        self._rules: list[tuple[str, bool, bool, bool]] = []
        raw_patterns = list(DEFAULT_IGNORE if patterns is None else patterns)
        raw_patterns.extend(self._read_ignore_files())
        for raw in raw_patterns:
            self._add(raw)

    def _read_ignore_files(self) -> list[str]:
        patterns: list[str] = []
        for name in (".gitignore", ".orionignore"):
            path = self.root / name
            if not path.is_file():
                continue
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except OSError:
                continue
        return patterns

    def _add(self, raw: str) -> None:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            return
        negated = raw.startswith("!")
        if negated:
            raw = raw[1:].strip()
        if not raw:
            return
        dir_only = raw.endswith("/")
        raw = raw.rstrip("/")
        anchored = raw.startswith("/")
        raw = raw.lstrip("/")
        if raw in {"", ".", ".."}:
            return
        self._rules.append((raw, negated, dir_only, anchored))

    def is_ignored(self, relative_path: str | Path, is_dir: bool = False) -> bool:
        rel = str(relative_path).replace("\\", "/").strip("/")
        if not rel:
            return False
        ignored = False
        for pattern, negated, dir_only, anchored in self._rules:
            if dir_only and not is_dir:
                continue
            if self._matches(rel, pattern, anchored):
                ignored = not negated
        return ignored

    @staticmethod
    def _matches(rel: str, pattern: str, anchored: bool) -> bool:
        if "/" in pattern:
            if fnmatch.fnmatch(rel, pattern):
                return True
            return rel.startswith(pattern + "/")
        if anchored:
            return fnmatch.fnmatch(rel.split("/", 1)[0], pattern)
        if fnmatch.fnmatch(rel, pattern):
            return True
        return any(fnmatch.fnmatch(segment, pattern) for segment in rel.split("/"))


def iter_source_files(
    root: str | Path,
    ignore: Iterable[str] | None = None,
    max_bytes: int = 512_000,
    rules: IgnoreRules | None = None,
) -> Iterator[Path]:
    """Yield text-like source files under ``root``, honouring ignore rules."""
    root_path = Path(root).resolve()
    matcher = rules or IgnoreRules(root_path, ignore)
    for dirpath, dirnames, filenames in os.walk(root_path):
        current = Path(dirpath)
        rel_dir = current.relative_to(root_path)
        kept: list[str] = []
        for name in sorted(dirnames):
            rel = (rel_dir / name).as_posix()
            if matcher.is_ignored(rel, is_dir=True):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            path = current / name
            rel = (rel_dir / name).as_posix()
            if matcher.is_ignored(rel):
                continue
            if not is_source_file(path):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            if is_binary(path):
                continue
            yield path


def relative_posix(path: str | Path, root: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def safe_path(root: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and guarantee it stays inside ``root``."""
    root_path = Path(root).resolve()
    raw = Path(candidate).expanduser()
    target = raw if raw.is_absolute() else root_path / raw
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise ToolError(f"Invalid path {candidate}: {exc}") from exc
    if resolved != root_path and root_path not in resolved.parents:
        raise ToolError(
            f"Path '{candidate}' escapes the workspace ({root_path}). "
            "Orion only operates inside the project directory."
        )
    return resolved


def unified_diff(before: str, after: str, path: str, context: int = 3) -> str:
    """Return a git-style unified diff for a proposed edit."""
    import difflib

    def _lines(value: str) -> list[str]:
        lines = value.splitlines(keepends=True)
        if value and not value.endswith("\n"):
            lines = [*lines[:-1], lines[-1] + "\n"] if lines else lines
        return lines

    diff = difflib.unified_diff(
        _lines(before),
        _lines(after),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(diff)


def _run_git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_info(root: str | Path) -> dict[str, object]:
    """Collect lightweight repository metadata; never raises."""
    root_path = Path(root)
    branch = _run_git(root_path, "rev-parse", "--abbrev-ref", "HEAD")
    if not branch:
        return {"available": False, "branch": None, "commit": None, "dirty": False, "changes": 0}
    commit = _run_git(root_path, "rev-parse", "--short", "HEAD")
    status = _run_git(root_path, "status", "--porcelain")
    changes = len([line for line in status.splitlines() if line.strip()])
    return {
        "available": True,
        "branch": branch,
        "commit": commit,
        "dirty": bool(changes),
        "changes": changes,
        "status": truncate(status, 4000, "git status"),
    }


def _has_pytest_files(root_path: Path) -> bool:
    """Whether common pytest artifacts exist near the project root."""
    if (root_path / "conftest.py").is_file():
        return True
    directories = [root_path]
    for name in ("tests", "test"):
        candidate = root_path / name
        if candidate.is_dir():
            directories.append(candidate)
    for directory in directories:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or entry.suffix != ".py":
                continue
            name = entry.name
            if name.startswith("test_") or name.endswith("_test.py"):
                return True
    return False


def _declares_pytest(root_path: Path) -> bool:
    """Look for pytest in packaging and requirements files."""
    for name in (
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "dev-requirements.txt",
        "Pipfile",
    ):
        path = root_path / name
        if not path.is_file():
            continue
        try:
            if "pytest" in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def detect_test_command(root: str | Path) -> tuple[str | None, str | None]:
    """Guess how to run the project's test suite."""
    root_path = Path(root)
    pyproject = root_path / "pyproject.toml"
    if pyproject.is_file():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        if "pytest" in content or (root_path / "tests").is_dir():
            return "python -m pytest -q", "pytest"
    if (root_path / "pytest.ini").is_file() or (root_path / "tox.ini").is_file():
        return "python -m pytest -q", "pytest"
    if _has_pytest_files(root_path) or _declares_pytest(root_path):
        return "python -m pytest -q", "pytest"
    package_json = root_path / "package.json"
    if package_json.is_file():
        import json

        try:
            data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            data = {}
        scripts = data.get("scripts") if isinstance(data, dict) else None
        if isinstance(scripts, dict) and scripts.get("test"):
            return "npm test", "npm"
    makefile = root_path / "Makefile"
    if makefile.is_file():
        try:
            if any(line.strip().startswith("test:") for line in makefile.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()):
                return "make test", "make"
        except OSError:
            pass
    if (root_path / "go.mod").is_file():
        return "go test ./...", "go"
    if (root_path / "Cargo.toml").is_file():
        return "cargo test --quiet", "cargo"
    if (root_path / "pom.xml").is_file():
        return "mvn -q test", "maven"
    if (root_path / "build.gradle").is_file() or (root_path / "build.gradle.kts").is_file():
        return "./gradlew test", "gradle"
    return None, None


@dataclass(frozen=True)
class VerificationCommand:
    """A deterministic check the evaluator can run without an LLM judge."""

    name: str
    command: str
    framework: str


def detect_verification_commands(root: str | Path) -> list[VerificationCommand]:
    """Detect test, lint, and type-check commands for the evaluator.

    These are real, repeatable signals. The evaluator combines their exit codes
    with (not merely) the model's own opinion about the changes.
    """
    root_path = Path(root)
    found: list[VerificationCommand] = []
    seen: set[str] = set()

    def add(name: str, command: str | None, framework: str) -> None:
        if not command or command in seen:
            return
        seen.add(command)
        found.append(VerificationCommand(name=name, command=command, framework=framework))

    test_command, test_framework = detect_test_command(root_path)
    add("tests", test_command, test_framework or "tests")

    config_text = ""
    for config_name in ("pyproject.toml", "setup.cfg", "tox.ini", ".flake8"):
        config_path = root_path / config_name
        if config_path.is_file():
            try:
                config_text += "\n" + config_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    lowered = config_text.lower()
    has_src = (root_path / "src").is_dir()
    has_tests = (root_path / "tests").is_dir()

    if "ruff" in lowered:
        target = "src tests" if has_src and has_tests else "."
        add("lint", f"ruff check {target}", "ruff")
    elif "flake8" in lowered:
        add("lint", "python -m flake8 .", "flake8")

    if "mypy" in lowered:
        target = "src" if has_src else "."
        add("typecheck", f"python -m mypy {target}", "mypy")
    elif (root_path / "tsconfig.json").is_file():
        add("typecheck", "npx tsc --noEmit", "typescript")
    elif (root_path / "go.mod").is_file():
        add("typecheck", "go vet ./...", "go")
    elif (root_path / "Cargo.toml").is_file():
        add("typecheck", "cargo clippy --all-targets -- -D warnings", "clippy")

    if "lint" not in {command.name for command in found}:
        package_json = root_path / "package.json"
        if package_json.is_file():
            try:
                scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
            except (OSError, ValueError, AttributeError):
                scripts = {}
            if isinstance(scripts, dict) and scripts.get("lint"):
                add("lint", "npm run lint", "npm")

    return found


def build_tree(relative_paths: Iterable[str], limit: int = 120) -> list[str]:
    """Render a compact directory tree from relative POSIX paths."""
    tree: dict[str, dict] = {}
    for raw in relative_paths:
        node = tree
        parts = [part for part in raw.split("/") if part]
        for part in parts:
            node = node.setdefault(part, {})
    lines: list[str] = []

    def walk(node: dict[str, dict], prefix: str) -> None:
        if len(lines) >= limit:
            return
        items = sorted(node.items(), key=lambda item: (not bool(item[1]), item[0].lower()))
        for name, children in items:
            if len(lines) >= limit:
                return
            lines.append(f"{prefix}{name}{'/' if children else ''}")
            if children:
                walk(children, prefix + "  ")

    walk(tree, "")
    if len(lines) >= limit:
        lines.append(f"... (tree truncated at {limit} entries)")
    return lines


def project_overview(
    root: str | Path,
    ignore: Iterable[str] | None = None,
    max_files: int = 140,
    max_chars: int = 9000,
) -> str:
    """A compact markdown summary of the repository for agent prompts."""
    root_path = Path(root).resolve()
    files = list(iter_source_files(root_path, ignore=ignore, max_bytes=512_000))
    rels = sorted(relative_posix(path, root_path) for path in files)
    tree_lines = build_tree(rels, limit=max_files)
    git = git_info(root_path)
    test_command, framework = detect_test_command(root_path)

    languages: dict[str, int] = {}
    for rel in rels:
        language = language_for(rel)
        if language not in {"text", "markdown", "json", "yaml", "toml"}:
            languages[language] = languages.get(language, 0) + 1
    top_languages = sorted(languages.items(), key=lambda item: item[1], reverse=True)[:8]

    parts = [
        f"# Repository: {root_path.name}",
        f"Root: `{root_path}`",
    ]
    if git.get("available"):
        parts.append(
            f"Git: branch `{git['branch']}` at `{git['commit']}` "
            f"({'dirty, ' + str(git['changes']) + ' changed file(s)' if git['dirty'] else 'clean'})"
        )
    if top_languages:
        parts.append("Languages: " + ", ".join(f"{name} ({count})" for name, count in top_languages))
    if test_command:
        parts.append(f"Test command: `{test_command}` ({framework})")
    parts.extend(["", "## Files", "```", *tree_lines, "```"])

    important = [
        rel for rel in rels
        if Path(rel).name.lower() in {
            "readme.md", "readme", "pyproject.toml", "package.json", "cargo.toml",
            "go.mod", "dockerfile", "docker-compose.yml", "makefile",
        }
    ][:6]
    if important:
        parts.extend(["", "## Key files"])
        for rel in important:
            try:
                content = read_text_file(root_path / rel, max_bytes=1800)
            except ToolError:
                continue
            parts.append(f"### {rel}")
            parts.append("```" + language_for(rel))
            parts.append(truncate(content, 1200, rel))
            parts.append("```")

    overview = "\n".join(parts)
    return truncate(overview, max_chars, "project overview")


def changed_files(root: str | Path) -> list[str]:
    """Return git-visible changed files (empty when git is unavailable)."""
    root_path = Path(root)
    status = _run_git(root_path, "status", "--porcelain")
    results: list[str] = []
    for line in status.splitlines():
        if len(line) > 3:
            results.append(line[3:].strip())
    return results


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


__all__ = [
    "LANGUAGE_BY_EXT",
    "SOURCE_EXTENSIONS",
    "SOURCE_NAMES",
    "IgnoreRules",
    "build_tree",
    "changed_files",
    "detect_test_command",
    "detect_verification_commands",
    "git_info",
    "human_size",
    "is_binary",
    "is_source_file",
    "iter_source_files",
    "language_for",
    "project_overview",
    "read_text_file",
    "relative_posix",
    "safe_path",
    "truncate",
    "unified_diff",
]
