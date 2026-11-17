"""Orion AI command line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Keep ChromaDB quiet before it is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from orion.cli_ui import ConsoleRenderer
from orion.config import OrionConfig, discover_env_files, write_env_value
from orion.engine import OrionEngine
from orion.errors import ConfigError
from orion.utils import truncate
from orion.version import __version__


def _configure_stdio() -> None:
    """Make Rich output safe on legacy Windows code pages.

    When stdout is redirected (a pipe, a log file, CI) Rich falls back to the
    legacy Windows renderer, which encodes text with the *stream* encoding --
    cp1252 on many Windows machines. Orion's UI uses Unicode glyphs, so force
    UTF-8 with replacement instead of crashing on the final character.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_stdio()

app = typer.Typer(
    add_completion=True,
    invoke_without_command=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
    help=(
        "Orion AI - a multi-agent coding assistant powered by Gemini, LangGraph, "
        "and ChromaDB. Run without arguments to open the interactive UI."
    ),
)
config_app = typer.Typer(help="Show or update Orion configuration.")
session_app = typer.Typer(help="Inspect saved sessions.")
app.add_typer(config_app, name="config")
app.add_typer(session_app, name="sessions")

console = Console()
ENV_TEMPLATE = """# Orion AI configuration
GEMINI_API_KEY=
ORION_VERTEX=auto
ORION_MODEL=gemini-3.5-flash
ORION_EMBEDDING_MODEL=gemini-embedding-001
ORION_MODE=build
ORION_APPROVAL=ask
ORION_EMBEDDINGS=auto
ORION_AUTO_INDEX=true
"""
ORIONIGNORE_TEMPLATE = """# Directories and files Orion should never index or search
.env
.env.*
*.pem
*.key
node_modules/
dist/
build/
coverage/
"""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold magenta]orion-ai[/bold magenta] {__version__}")
        raise typer.Exit()


def _workspace(ctx: typer.Context) -> Path:
    obj = ctx.obj or {}
    workspace = obj.get("workspace") or Path.cwd()
    return Path(workspace).expanduser().resolve()


def _config(ctx: typer.Context, **overrides) -> OrionConfig:
    obj = ctx.obj or {}
    clean = {key: value for key, value in overrides.items() if value is not None}
    if obj.get("model") and "model" not in clean:
        clean["model"] = obj["model"]
    if obj.get("auto"):
        clean["approval"] = "auto"
    try:
        return OrionConfig.load(workspace=_workspace(ctx), **clean)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _engine(config: OrionConfig) -> OrionEngine:
    return OrionEngine(config)


def _read_prompt(prompt: str | None) -> str:
    if prompt and prompt.strip():
        return prompt.strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    console.print("[red]A prompt is required.[/red] Pass it as an argument or pipe it on stdin.")
    raise typer.Exit(code=2)


def _check_mode(mode: str) -> str:
    if mode not in {"build", "plan"}:
        console.print("[red]mode must be 'build' or 'plan'[/red]")
        raise typer.Exit(code=2)
    return mode


def _run_prompt(
    ctx: typer.Context,
    prompt: str | None,
    *,
    mode: str,
    dry_run: bool = False,
    json_output: bool = False,
    verbose: bool = False,
    no_index: bool = False,
) -> None:
    task = _read_prompt(prompt)
    mode = _check_mode(mode)
    config = _config(
        ctx,
        mode=mode,
        dry_run=dry_run or None,
        auto_index=False if no_index else None,
    )
    if not config.has_api_key:
        console.print(
            Panel(
                Text(
                    "GEMINI_API_KEY is not set.\n\n"
                    f"Add it to {config.workspace / '.env'}:\n"
                    "  GEMINI_API_KEY=your-key\n\n"
                    "Create a key at https://aistudio.google.com/apikey"
                ),
                title="[bold red]missing API key[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    engine = _engine(config)
    if json_output:
        result = engine.run(task, mode=mode, dry_run=dry_run or None)
        console.print_json(json.dumps(result.as_dict(), ensure_ascii=False, default=str))
        if result.error:
            raise typer.Exit(code=1)
        return

    renderer = ConsoleRenderer(console=console, verbose=verbose)
    result = engine.run(task, mode=mode, dry_run=dry_run or None, on_event=renderer)
    if result.cancelled:
        raise typer.Exit(code=130)
    if result.error:
        raise typer.Exit(code=1)


def _launch_tui(ctx: typer.Context, mode: str | None = None) -> None:
    try:
        from orion.tui.app import run_tui
    except ImportError as exc:  # pragma: no cover - textual is a required dep
        console.print(f"[red]The interactive UI is unavailable: {exc}[/red]")
        console.print("Reinstall Orion: [bold]pip install 'orion-ai'[/bold]")
        raise typer.Exit(code=1) from exc
    config = _config(ctx, mode=mode)
    run_tui(config)


# ----------------------------------------------------------------------
# Root callback
# ----------------------------------------------------------------------
@app.callback()
def cli(
    ctx: typer.Context,
    workspace: Path | None = typer.Option(
        None, "--workspace", "-C", help="Project directory to work in."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the Gemini model (aliases: flash, pro, flash-lite)."
    ),
    auto: bool = typer.Option(
        False, "--auto", "-y", help="Automatically approve shell commands."
    ),
    no_tui: bool = typer.Option(False, "--no-tui", help="Never launch the interactive UI."),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Orion AI - multi-agent coding assistant for the terminal."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "workspace": workspace,
            "model": model,
            "auto": auto,
            "no_tui": no_tui,
        }
    )
    if ctx.invoked_subcommand is not None:
        return
    if no_tui or not _is_interactive():
        console.print(
            Panel(
                Text.assemble(
                    ("◆ ORION AI\n\n", "bold magenta"),
                    ("Multi-agent coding assistant powered by Gemini + LangGraph.\n\n", ""),
                    ("orion run \"add tests for the parser\"\n", "cyan"),
                    ("orion plan \"how does auth work?\"\n", "cyan"),
                    ("orion build \"add a /health endpoint\"\n", "cyan"),
                    ("orion index\n", "cyan"),
                    ("orion\n", "cyan"),
                    ("  (no arguments opens the interactive UI)\n\n", "dim"),
                    ("orion --help", "dim"),
                ),
                border_style="magenta",
            )
        )
        return
    _launch_tui(ctx)


# ----------------------------------------------------------------------
# Run commands
# ----------------------------------------------------------------------
@app.command("run")
def run_cmd(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="What should Orion do?"),
    mode: str = typer.Option("build", "--mode", "-M", help="build or plan."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute changes and diffs without writing files."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the result as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full tool output."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip automatic indexing."),
) -> None:
    """Run one task headlessly."""
    _run_prompt(
        ctx,
        prompt,
        mode=mode,
        dry_run=dry_run,
        json_output=json_output,
        verbose=verbose,
        no_index=no_index,
    )


@app.command("plan")
def plan_cmd(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="What should Orion plan?"),
    json_output: bool = typer.Option(False, "--json", help="Print the result as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_index: bool = typer.Option(False, "--no-index"),
) -> None:
    """Plan-only mode: read, search, and propose. Never writes files."""
    _run_prompt(
        ctx,
        prompt,
        mode="plan",
        json_output=json_output,
        verbose=verbose,
        no_index=no_index,
    )


@app.command("build")
def build_cmd(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="What should Orion build?"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing."),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_index: bool = typer.Option(False, "--no-index"),
) -> None:
    """Build mode: plan, edit files, review, and test."""
    _run_prompt(
        ctx,
        prompt,
        mode="build",
        dry_run=dry_run,
        json_output=json_output,
        verbose=verbose,
        no_index=no_index,
    )


# ----------------------------------------------------------------------
# Index / search
# ----------------------------------------------------------------------
@app.command("index")
def index_cmd(
    ctx: typer.Context,
    paths: list[str] | None = typer.Argument(
        None, help="Files or directories to index (default: whole workspace)."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild the whole index."),
    embeddings: str | None = typer.Option(
        None, "--embeddings", help="gemini or local (offline)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print stats as JSON."),
) -> None:
    """Build or refresh the ChromaDB code index."""
    if embeddings and embeddings not in {"gemini", "local", "auto"}:
        console.print("[red]--embeddings must be gemini, local, or auto[/red]")
        raise typer.Exit(code=2)
    config = _config(ctx, embeddings=embeddings)
    engine = _engine(config)
    renderer = None if json_output else ConsoleRenderer(console=console)
    stats = engine.index(force=force, embeddings=embeddings, on_event=renderer)
    if json_output:
        console.print_json(json.dumps(stats.as_dict(), ensure_ascii=False))
        return
    table = Table(title="Orion index", show_header=False, border_style="magenta")
    table.add_column("key", style="dim")
    table.add_column("value")
    data = stats.as_dict()
    for key in (
        "files_total",
        "files_indexed",
        "files_unchanged",
        "files_removed",
        "chunks",
        "vectors",
        "embedder",
        "store",
        "duration",
    ):
        table.add_row(key.replace("_", " "), str(data.get(key)))
    console.print(table)
    if stats.errors:
        for error in stats.errors[:10]:
            console.print(f"[yellow]! {error}[/yellow]")


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(8, "--limit", "-k", help="Number of results."),
    embeddings: str | None = typer.Option(None, "--embeddings", help="gemini or local."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Search the code index (hybrid vector + lexical)."""
    config = _config(ctx, embeddings=embeddings)
    engine = _engine(config)
    hits = engine.search(query, k=limit)
    if json_output:
        console.print_json(json.dumps(hits, ensure_ascii=False))
        return
    if not hits:
        console.print("[yellow]No matches. Build the index with `orion index`.[/yellow]")
        return
    for hit in hits:
        score = float(hit.get("score", 0))
        console.print(
            f"\n[bold cyan]{hit['path']}[/bold cyan]"
            f"[dim]:{hit['start_line']}-{hit['end_line']}  "
            f"score {score:.2f} · {hit.get('source', 'vector')}[/dim]"
        )
        console.print(
            Syntax(
                truncate(str(hit.get("text", "")), 1600, "snippet"),
                str(hit.get("language", "text")),
                theme="monokai",
                word_wrap=True,
            )
        )


# ----------------------------------------------------------------------
# Status / doctor
# ----------------------------------------------------------------------
@app.command("status")
def status_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show configuration, index, and session status."""
    config = _config(ctx)
    engine = _engine(config)
    data = engine.status()
    if json_output:
        console.print_json(json.dumps(data, ensure_ascii=False, default=str))
        return
    cfg = data["config"]
    table = Table(title="Orion status", border_style="magenta", show_header=False)
    table.add_column("key", style="dim")
    table.add_column("value")
    labels = {"vertex": "vertex hint"}
    for key in (
        "workspace",
        "model",
        "mode",
        "approval",
        "vertex_setting",
        "vertex",
        "embeddings",
        "gemini_api_key",
        "data_dir",
    ):
        value = cfg.get(key)
        if key == "vertex":
            value = "vertex" if value else "ai-studio"
        table.add_row(labels.get(key, key.replace("_", " ")), str(value))
    index = data["index"]
    for key in ("store", "embedder", "vectors", "files", "updated_at"):
        table.add_row(f"index {key}", str(index.get(key)))
    console.print(table)
    try:
        from orion.session import SessionStore

        latest = SessionStore(config).latest()
        if latest:
            console.print(
                f"[dim]last session: {latest.id} · {latest.status} · {latest.title}[/dim]"
            )
    except Exception:
        pass


@app.command("doctor")
def doctor_cmd(
    ctx: typer.Context,
    skip_ping: bool = typer.Option(False, "--skip-ping", help="Do not call the Gemini API."),
) -> None:
    """Check the environment, dependencies, credentials, and index."""
    import importlib.util
    import platform

    config = _config(ctx)
    table = Table(title="Orion doctor", border_style="magenta", show_header=False)
    table.add_column("check", style="dim")
    table.add_column("result")
    table.add_row("orion-ai", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    table.add_row("workspace", str(config.workspace))
    table.add_row("data dir", str(config.data_path))
    table.add_row("env files", ", ".join(str(path) for path in discover_env_files(config.workspace)) or "(none)")

    key = config.api_key
    if not key:
        table.add_row("GEMINI_API_KEY", "[red]missing[/red]")
    else:
        if config.vertex == "false":
            hint = "using AI Studio (ORION_VERTEX=false)"
        elif config.vertex == "true":
            hint = "using Vertex AI (ORION_VERTEX=true)"
        elif key.startswith("AQ."):
            hint = "AQ. prefix · Vertex first, falls back to AI Studio"
        else:
            hint = "using AI Studio"
        table.add_row("GEMINI_API_KEY", f"[green]set[/green] {config.masked_key} · {hint}")
    vertex_modes = {
        "auto": "detect, then fall back",
        "true": "forced on",
        "false": "forced off",
    }
    table.add_row(
        "vertex setting",
        f"{config.vertex} · {vertex_modes.get(config.vertex, config.vertex)}",
    )

    for module, label in (
        ("langgraph", "langgraph"),
        ("chromadb", "chromadb"),
        ("textual", "textual"),
        ("google.genai", "google-genai"),
    ):
        found = importlib.util.find_spec(module) is not None
        table.add_row(label, "[green]ok[/green]" if found else "[red]missing[/red]")

    try:
        import chromadb

        table.add_row("chromadb version", getattr(chromadb, "__version__", "unknown"))
    except Exception as exc:
        table.add_row("chromadb version", f"[red]{exc}[/red]")

    try:
        engine = _engine(config)
        index = engine.status()["index"]
        table.add_row(
            "index",
            f"{index.get('vectors')} vector(s), {index.get('files')} file(s), "
            f"{index.get('embedder')}",
        )
    except Exception as exc:
        table.add_row("index", f"[yellow]{exc}[/yellow]")

    if not skip_ping and config.has_api_key:
        try:
            from orion.llm import GeminiLLM

            llm = GeminiLLM(config)
            reply = llm.ping()
            table.add_row("gemini ping", f"[green]{truncate(reply, 60, 'reply')}[/green]")
            backend = llm.active_backend
            if llm.mode_switched:
                backend += " · auto-fallback"
            table.add_row("gemini backend", backend)
        except Exception as exc:
            table.add_row("gemini ping", f"[red]{truncate(str(exc), 160, 'error')}[/red]")
    console.print(table)


# ----------------------------------------------------------------------
# Init / config / sessions
# ----------------------------------------------------------------------
@app.command("init")
def init_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Create a starter .env and .orionignore in the workspace."""
    workspace = _workspace(ctx)
    env_path = workspace / ".env"
    ignore_path = workspace / ".orionignore"
    created: list[str] = []
    if env_path.exists() and not force:
        console.print(f"[yellow]{env_path} already exists; leaving it untouched.[/yellow]")
    else:
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8", newline="\n")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        created.append(str(env_path))
    if ignore_path.exists() and not force:
        console.print(f"[dim]{ignore_path} already exists.[/dim]")
    else:
        ignore_path.write_text(ORIONIGNORE_TEMPLATE, encoding="utf-8", newline="\n")
        created.append(str(ignore_path))
    if created:
        console.print("[green]Created:[/green]")
        for path in created:
            console.print(f"  {path}")
    console.print(
        "\nNext: add your key to [bold].env[/bold], then run [bold]orion[/bold] "
        "or [bold]orion index[/bold]."
    )


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the effective configuration."""
    config = _config(ctx)
    data = config.describe()
    if json_output:
        console.print_json(json.dumps(data, ensure_ascii=False))
        return
    table = Table(title="Orion configuration", border_style="magenta", show_header=False)
    table.add_column("key", style="dim")
    table.add_column("value")
    for key, value in data.items():
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Environment variable, e.g. ORION_MODEL."),
    value: str | None = typer.Argument(None, help="New value."),
    file: Path | None = typer.Option(None, "--file", help="Dotenv file to update."),
) -> None:
    """Set a value in .env."""
    key = key.strip().upper()
    if not key.startswith("ORION_") and key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
        console.print("[red]Use an ORION_* variable or GEMINI_API_KEY.[/red]")
        raise typer.Exit(code=2)
    if value is None:
        if "KEY" in key:
            value = typer.prompt(key, hide_input=True)
        else:
            console.print("[red]A value is required.[/red]")
            raise typer.Exit(code=2)
    target = file or (_workspace(ctx) / ".env")
    path = write_env_value(key, value, target)
    console.print(f"[green]Updated[/green] {path} ([bold]{key}[/bold])")


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Show where Orion reads configuration and stores data."""
    config = _config(ctx)
    console.print(f"workspace:  {config.workspace}")
    console.print(f"env files:  {', '.join(str(path) for path in discover_env_files(config.workspace)) or '(none)'}")
    console.print(f"data dir:   {config.data_path}")
    console.print(f"chroma dir: {config.chroma_path}")
    console.print(f"sessions:   {config.session_path}")


@session_app.command("list")
def sessions_list(
    ctx: typer.Context,
    limit: int = typer.Option(15, "--limit", "-n"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List recent sessions."""
    from orion.session import SessionStore

    config = _config(ctx)
    metas = SessionStore(config).list(limit=limit)
    if json_output:
        console.print_json(json.dumps([meta.as_dict() for meta in metas], ensure_ascii=False))
        return
    if not metas:
        console.print("[dim]No sessions yet.[/dim]")
        return
    table = Table(title="Orion sessions", border_style="magenta")
    table.add_column("id", style="cyan", width=15)
    table.add_column("status", width=9)
    table.add_column("mode", width=6)
    table.add_column("updated", style="dim", width=11)
    table.add_column("title")
    budget = max(16, console.width - (15 + 9 + 6 + 11) - 16)
    for meta in metas:
        updated = meta.updated_at.replace("T", " ")[5:16]
        title = meta.title if len(meta.title) <= budget else meta.title[: budget - 1] + "…"
        table.add_row(meta.id[:15], meta.status, meta.mode, updated, title)
    console.print(table)
    console.print("[dim]Use an id prefix with `orion sessions show <id>`.[/dim]")


@session_app.command("show")
def sessions_show(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session id (prefix is fine)."),
) -> None:
    """Print a session transcript."""
    from orion.session import SessionStore

    config = _config(ctx)
    store = SessionStore(config)
    metas = store.list(limit=500)
    match = next((meta for meta in metas if meta.id == session_id), None)
    if match is None:
        match = next((meta for meta in metas if meta.id.startswith(session_id)), None)
    if match is None:
        console.print("[red]Session not found.[/red]")
        raise typer.Exit(code=1)
    console.print(
        Panel(
            Text(f"{match.title}\n{match.id}\n{match.mode} · {match.status} · {match.updated_at}"),
            title="[bold magenta]session[/bold magenta]",
            border_style="magenta",
        )
    )
    for event in store.load_events(match.id):
        kind = event.get("type")
        if kind == "message":
            console.print(
                Panel(
                    Markdown(str(event.get("text", ""))),
                    title=f"[bold]{event.get('agent', 'orion')}[/bold]",
                    border_style="grey30",
                )
            )
        elif kind == "plan":
            console.print(Panel(Markdown(str(event.get("markdown", ""))), title="plan"))
        elif kind == "tool_call":
            console.print(f"[dim]⚙ {event.get('name')} {event.get('args')}[/dim]")
        elif kind == "review":
            console.print(f"[bold]review:[/bold] {event.get('verdict')} - {event.get('summary')}")
        elif kind == "test":
            console.print(f"[bold]tests:[/bold] {event.get('status')} {event.get('command') or ''}")
        elif kind == "error":
            console.print(f"[red]{event.get('message')}[/red]")


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------
@app.command("tui")
def tui_cmd(
    ctx: typer.Context,
    mode: str | None = typer.Option(None, "--mode", "-M", help="build or plan."),
) -> None:
    """Open the interactive Orion UI."""
    if mode:
        _check_mode(mode)
    _launch_tui(ctx, mode=mode)


@app.command("graph")
def graph_cmd() -> None:
    """Print the agent graph as a Mermaid diagram."""
    from orion.agents.graph import graph_mermaid

    console.print(Syntax(graph_mermaid(), "mermaid", theme="monokai"))


@app.command("serve")
def serve_cmd(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Run the optional HTTP API (requires the 'server' extra)."""
    config = _config(ctx)
    try:
        import uvicorn

        from orion.server import create_app
    except ImportError as exc:
        console.print(
            "[red]The server extra is not installed.[/red]\n"
            "Install it with: [bold]pip install 'orion-ai[server]'[/bold]"
        )
        raise typer.Exit(code=1) from exc
    console.print(f"[magenta]Orion API[/magenta] listening on http://{host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


@app.command("version")
def version_cmd() -> None:
    """Show the Orion version."""
    console.print(f"orion-ai {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
