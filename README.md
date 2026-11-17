# Orion AI

**A multi-agent coding assistant for your terminal** — powered by **Vertex AI**
through your local **gcloud ADC** (no API key needed), **LangGraph**, **ChromaDB**
retrieval, and a Textual TUI. A team of specialized models plans, builds, reviews,
and evaluates your code using real test/lint/type-check signals.

Orion plans, edits, reviews, and tests code the way a small engineering team
would. It runs in two modes:

- **Plan mode** — read-only. Orion investigates the repository and proposes a
  plan without touching a single file.
- **Build mode** — Orion writes code, runs commands you approve, reviews its own
  work, and runs your test suite.

Everything is available from the interactive TUI, headless CLI commands, a
Python library, and an optional HTTP API.

---

## Features

| | |
| --- | --- |
| 🤖 **Multi-agent pipeline** | `context → summarizer → architect → generator → reviewer → evaluator → debugger → finalize`, orchestrated with LangGraph. Real tests/lint/type checks drive the feedback loop. |
| 🧠 **Role-based models** | Different Vertex models for architecture, generation, evaluation, review, debugging, and summarization — no single expensive model does everything. |
| 🧭 **Plan / build modes** | Plan mode exposes only read-only tools. Build mode adds writes, moves, deletes, and shell commands. |
| 🔍 **Repository RAG** | Incremental ChromaDB index with line-accurate chunks, hybrid vector + keyword search, and a JSON fallback store. |
| 🛡️ **Safe by default** | Workspace path sandboxing, a dangerous-command blocklist, approval policies (`ask` / `auto` / `deny`), and `--dry-run`. |
| 💻 **opencode-style TUI** | Streaming responses, plan/review/test/diff panels, tool-call cards, a command palette, and live status. |
| 🧪 **Headless & scriptable** | `orion run`, `orion plan`, `orion build`, `--json` output, and a stable event stream. |
| 🐍 **Library & server** | `from orion import Orion`, plus a FastAPI server behind an optional extra. |
| 🐳 **Docker-ready** | Multi-stage image running as a non-root user, with a Compose file for the CLI and API. |

---

## Quick start

### 1. Install

From a built wheel:

```bash
pip install orion_ai-1.0.7-py3-none-any.whl
# or directly from GitHub:
pip install https://raw.githubusercontent.com/rajbasnet123789/CLI_BASED_MULTI-RL-AGENTS/main/dist/orion_ai-1.0.7-py3-none-any.whl
```

If you already have an older Orion installed, use
`python -m pip install --upgrade <wheel>` so the new version replaces it (pip
skips reinstalling the same version number).

From source (editable):

```bash
git clone <your-repo> orion && cd orion
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev,server]"
```

With Docker:

```bash
cp .env.example .env
# mount your gcloud ADC credentials, then:
docker compose run --rm orion
```

### 2. Configure (Vertex AI via gcloud ADC — no API key)

One-time setup:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Then copy the example config and set your project:

```bash
cp .env.example .env
```

```dotenv
# .env
ORION_LLM_PROVIDER=gcloud
ORION_GCLOUD_PROJECT=YOUR_PROJECT_ID
ORION_GCLOUD_REGION=global
ORION_GCLOUD_ENDPOINT=aiplatform.googleapis.com
```

Orion uses the global Vertex endpoint by default. **No user API key is required.**
Make sure the Vertex AI API is enabled in the project and your account has
`roles/aiplatform.user`.

Role-to-model allocation (also in `.env.example`):

| Role | Model |
| --- | --- |
| Principal architect (analysis, planning) | `claude-opus-5-5` |
| Task decomposition / code generation | `glm-5-maas` |
| Code evaluator | `claude-opus-5-5` |
| Code reviewer | `gemini-3-flash-preview` |
| Debugging agent | `glm-5-maas` |
| Context summarization | `gemini-flash-lite` |

> Some third-party models (Claude, GLM-5) require Model Garden access or a
> self-deployed endpoint on your GCP project. Orion keeps the exact requested
> IDs in config, but if Vertex returns `404/NOT_FOUND` it automatically falls
> back to an available Gemini model on the global endpoint so a fresh install
> still works. You can change the chain in `src/orion/llm.py` →
> `GCLOUD_MODEL_FALLBACKS`.

### 3. Run

```bash
orion                  # interactive TUI (default)
orion plan "explain the authentication flow"
orion build "add a /health endpoint and a test"
orion run "find the cause of the failing test" --mode build
```

> Run `orion doctor` if anything looks wrong — it checks dependencies,
> credentials, the workspace, and the index.

---

## Interactive TUI

Launch with `orion` (or `orion tui`). The TUI is a thin layer over the same
engine used by the CLI.

**Orion works in the directory you launch it from.** `cd` into your project
first, then run `py -3.10 -m orion`. In build mode (the default) prompts create
and edit files there and run project commands with that directory as the working
directory. The welcome panel and sidebar show the active directory. Use
`-C "D:\path\to\project"` to target another directory without `cd`.

| Key | Action |
| --- | --- |
| `Enter` | Send the task |
| `Ctrl+J` / `Shift+Enter` | Insert a newline |
| `Tab` | Toggle **plan** / **build** mode |
| `Ctrl+P` | Command palette |
| `Ctrl+N` | New session |
| `Ctrl+C` | Cancel the active run (quits when idle) |
| `F1` | Help |

Slash commands: `/plan`, `/build`, `/mode`, `/index [force] [local|gemini]`,
`/search <query>`, `/model <name>`, `/approve`, `/sessions`, `/diff`,
`/context`, `/new`, `/clear`, `/about`, `/help`, `/quit`.

Approval prompts appear as modal dialogs when a tool wants to run a command.
Choose **Allow**, **Always** (switch to `auto`), or **Deny**.

---

## CLI reference

```text
orion [OPTIONS] COMMAND [ARGS]...
```

Global options: `--workspace/-C`, `--model/-m`, `--auto/-y`, `--no-tui`,
`--version/-V`.

| Command | Description |
| --- | --- |
| `orion run [PROMPT]` | Run one task headlessly (`--mode`, `--dry-run`, `--json`, `--verbose`, `--no-index`). |
| `orion plan [PROMPT]` | Read-only planning run. |
| `orion build [PROMPT]` | Build run that may edit files and run tests. |
| `orion index [PATHS...]` | Build/refresh the code index (`--force`, `--embeddings gemini\|local`, `--json`). |
| `orion search QUERY` | Hybrid search (`--limit/-k`, `--embeddings`, `--json`). |
| `orion status` | Show configuration, index, and session status (`--json`). |
| `orion doctor` | Check environment, dependencies, credentials, and index (`--skip-ping`). |
| `orion init` | Create a starter `.env` and `.orionignore` (`--force`). |
| `orion tui` | Open the interactive UI (`--mode`). |
| `orion graph` | Print the agent graph as a Mermaid diagram. |
| `orion serve` | Run the HTTP API (`--host`, `--port`; requires `[server]`). |
| `orion config show` / `set` / `path` | Inspect or update configuration. |
| `orion sessions list` / `show` | Inspect saved sessions. |
| `orion version` | Print the version. |

Examples:

```bash
orion build "add retries to the HTTP client" --dry-run
orion index --embeddings local          # offline embedder, no API key
orion search "where is the database configured" -k 5
orion run "summarise the architecture" --json > summary.json
orion config set ORION_MODEL gemini-3.5-flash
orion doctor
```

---

## Library usage

```python
from orion import Orion, OrionConfig

orion = Orion(OrionConfig.load())

result = orion.run("Add a /health endpoint", mode="build")
print(result.final)
print(result.changes)

# Stream structured events instead of waiting for completion.
for event in orion.stream("Explain the auth flow", mode="plan"):
    print(event["type"], event.get("message", ""))
```

`RunResult` exposes `task`, `mode`, `final`, `plan`, `plan_text`, `changes`,
`review`, `tests`, `usage`, `session_id`, `duration`, `cancelled`, `error`, and
`context`. See [docs/library.md](docs/library.md) and
[docs/events.md](docs/events.md).

---

## HTTP API

Install the extra and start the server:

```bash
pip install "orion-ai[server]"
orion serve --host 0.0.0.0 --port 8765
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and basic configuration. |
| `GET` | `/status` | Config + index status. |
| `GET` | `/config` | Effective configuration (secrets masked). |
| `POST` | `/run` | Run a task and return the result. |
| `POST` | `/stream` | Server-Sent Events for a run. |
| `POST` | `/index` | Build/refresh the index. |
| `GET` | `/search?q=...&k=8` | Hybrid code search. |

---

## Docker

```bash
cp .env.example .env
docker build -t orion-ai:1.0.7 .
docker compose run --rm orion                 # TUI
docker compose run --rm orion run "add tests" # headless
docker compose --profile api up api           # API on :8765
```

The project is mounted at `/workspace`; state lives in `/workspace/.orion`.
See [docs/docker.md](docs/docker.md).

---

## Architecture

```text
task ──▶ context ──▶ summarizer ──▶ architect ──▶ finalize ──▶ result   (plan mode)

task ──▶ context ──▶ summarizer ──▶ architect ──▶ generator ──▶ reviewer
                                                     │
                                                     ▼
                                                  evaluator ──▶ finalize ──▶ result
                                                     │
                                                     ▼ (needs changes)
                                                   debugger ──▶ generator
```

- **context** retrieves RAG chunks and a repository overview.
- **summarizer** compresses the context into a short briefing for the architect.
- **architect** (principal architect) produces a structured plan or markdown plan.
- **generator** runs the tool loop: read, search, write, edit, run commands.
- **reviewer** returns a verdict plus issues and suggestions.
- **evaluator** runs real test/lint/type-check commands and combines those signals
  with an independent model verdict. Failed checks always block approval.
- **debugger** turns failed evaluation/review findings into concrete fix notes.
- **finalize** assembles the final report.

Tool access is mode-aware:

| Plan mode (read-only) | Build mode (all of the above, plus) |
| --- | --- |
| `read_file`, `list_dir`, `glob`, `grep`, `search_code`, `git_status`, `git_diff` | `write_file`, `edit_file`, `delete_file`, `move_file`, `run_command`, `run_tests` |

More detail in [docs/architecture.md](docs/architecture.md).

---

## Configuration

Configuration is read from `ORION_*` environment variables and layered `.env`
files (home `~/.orion/.env`, then ancestors of the workspace, nearest last).
The most useful settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ORION_LLM_PROVIDER` | `gcloud` | Vertex AI via gcloud ADC (default, no API key). Legacy `gemini`/`local` remain for compatibility. |
| `ORION_GCLOUD_PROJECT` | — | GCP project ID (`PROJECT_ID` alias accepted). |
| `ORION_GCLOUD_REGION` | `global` | Vertex endpoint region (`REGION` alias accepted). Use `global` for generation. |
| `ORION_GCLOUD_ENDPOINT` | `aiplatform.googleapis.com` | Global Vertex endpoint (`ENDPOINT` alias accepted). |
| `ORION_ARCHITECT_MODEL` | `claude-opus-5-5` | Principal architect: repository analysis, architecture, initial plan. |
| `ORION_GENERATOR_MODEL` | `glm-5-maas` | Task decomposition and initial code generation. |
| `ORION_EVALUATOR_MODEL` | `claude-opus-5-5` | Code evaluator (uses real verification signals). |
| `ORION_REVIEWER_MODEL` | `gemini-3-flash-preview` | Code reviewer. |
| `ORION_DEBUGGING_MODEL` | `glm-5-maas` | Debugging agent. |
| `ORION_SUMMARIZER_MODEL` | `gemini-flash-lite` | Context summarization. |
| `ORION_MODEL` | `gemini-3.5-flash` | Fallback model for roles without an override. |
| `ORION_PLANNER_MODEL` / `ORION_CODER_MODEL` / `ORION_REVIEWER_MODEL` | unset | Legacy per-agent overrides (still honored as aliases). |
| `ORION_MODE` | `build` | `build` or `plan`. |
| `ORION_APPROVAL` | `ask` | `ask`, `auto`, or `deny`. |
| `ORION_EMBEDDINGS` | `local` | `local` (offline hashing), `gemini` (Vertex embeddings), or `auto`. |
| `ORION_EMBEDDING_MODEL` | `text-embedding-005` | Vertex embedding model when `ORION_EMBEDDINGS=gemini`. |
| `ORION_TOP_K` | `8` | Retrieved chunks per run. |
| `ORION_AUTO_INDEX` | `true` | Index automatically before a run. |
| `ORION_DRY_RUN` | `false` | Never write files. |
| `ORION_DATA_DIR` | `<workspace>/.orion` | Sessions, index, and manifests. |

Full list: [docs/configuration.md](docs/configuration.md).

---

## Development

```bash
make install     # pip install -e ".[dev,server]"
make test        # pytest
make lint        # ruff check src tests
make fmt         # ruff format + autofix
make build       # python -m build → dist/
make docker      # docker build -t orion-ai:1.0.7 .
```

The test suite runs without network access: agent orchestration is exercised
with a deterministic fake model, and RAG tests use the local hashing embedder.

---

## Security

- Orion only reads and writes inside the configured workspace; paths that
  escape it are rejected.
- Shell commands are gated by the approval policy and a dangerous-command
  blocklist. Use `--dry-run` or `ORION_APPROVAL=deny` in untrusted settings.
- `.env` files and `.orion/` are excluded from git and from the Docker context.
- **Never commit real API keys.** Rotate any key that has been shared or
  exposed.

---

## License

MIT — see [LICENSE](LICENSE).
