# Configuration

Orion reads configuration from, in order of increasing priority:

1. Built-in defaults.
2. `~/.orion/.env`.
3. `.env` files from the project root down to the workspace (nearest wins).
4. `ORION_*` environment variables.
5. Explicit constructor arguments or CLI flags.

Orion uses **Vertex AI via gcloud Application Default Credentials** by default.
No user API key is required. The legacy `gemini`/`local` providers remain in the
code for compatibility but are not part of the default setup.

## LLM provider

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_LLM_PROVIDER` | `gcloud` | Vertex AI via gcloud ADC (no API key). `gemini`/`local` are legacy compatibility paths. |
| `ORION_GCLOUD_PROJECT` | — | GCP project ID. Aliases: `PROJECT_ID`, `GCLOUD_PROJECT`, `GOOGLE_CLOUD_PROJECT`. |
| `ORION_GCLOUD_REGION` | `global` | Vertex endpoint region/location. Aliases: `REGION`, `GCLOUD_REGION`, `GOOGLE_CLOUD_REGION`. |
| `ORION_GCLOUD_ENDPOINT` | `aiplatform.googleapis.com` | Global Vertex endpoint. Aliases: `ENDPOINT`, `GCLOUD_ENDPOINT`. |

In `gcloud` mode Orion authenticates with local Application Default Credentials
and calls the **global** Vertex AI endpoint. No `GEMINI_API_KEY` is used. Enable
the Vertex AI API in the project and grant your account `roles/aiplatform.user`.
Use `global` for generation; regional endpoints such as `us-east5` return
`404/NOT_FOUND` for the models Orion uses on many projects.

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

```dotenv
ORION_LLM_PROVIDER=gcloud
ORION_GCLOUD_PROJECT=YOUR_PROJECT_ID
ORION_GCLOUD_REGION=global
ORION_GCLOUD_ENDPOINT=aiplatform.googleapis.com
ORION_MODEL=gemini-3.5-flash
```

`orion config`, `orion status`, the TUI, and run headers show the effective
`active_model`.

### Role-based model allocation

Each agent can use a different Vertex model. The exact requested allocation is:

| Variable | Role | Model |
| --- | --- | --- |
| `ORION_ARCHITECT_MODEL` | Principal architect (analysis, architecture, plan) | `claude-opus-5-5` |
| `ORION_GENERATOR_MODEL` | Task decomposition and initial code generation | `glm-5-maas` |
| `ORION_EVALUATOR_MODEL` | Code evaluator (with real verification signals) | `claude-opus-5-5` |
| `ORION_REVIEWER_MODEL` | Code reviewer | `gemini-3-flash-preview` |
| `ORION_DEBUGGING_MODEL` | Debugging agent | `glm-5-maas` |
| `ORION_SUMMARIZER_MODEL` | Context summarization | `gemini-flash-lite` |

If Vertex returns `404/NOT_FOUND` (e.g. Claude without Model Garden access, or
GLM-5 retired on Vertex), Orion automatically tries the next model in
`GCLOUD_MODEL_FALLBACKS` in `src/orion/llm.py`. This keeps a fresh install working
without extra keys or manual edits. `ORION_MODEL` is the fallback for roles with
no explicit override. The legacy `ORION_PLANNER_MODEL`/`ORION_CODER_MODEL`/
`ORION_REVIEWER_MODEL` variables remain as aliases.

## Credentials

| Variable | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Google AI Studio (`AIza...`) or Vertex express (`AQ....`); Gemini provider only. |
| `ORION_VERTEX` | `auto` | `auto` detects from the key prefix; `true`/`false` force the mode. |

The `GEMINI_API_KEY`/`ORION_VERTEX` settings apply only to the legacy
`ORION_LLM_PROVIDER=gemini` path. The default gcloud/Vertex path ignores them.

## Models

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_MODEL` | `gemini-3.5-flash` | Fallback model for roles without an override. Aliases: `flash`, `pro`, `flash-lite`, `3.5-flash`, `3.5-flash-lite`, `3.1-pro`. |
| `ORION_ARCHITECT_MODEL` | `claude-opus-5-5` | Principal architect. |
| `ORION_GENERATOR_MODEL` | `glm-5-maas` | Task decomposition / code generation. |
| `ORION_EVALUATOR_MODEL` | `claude-opus-5-5` | Code evaluator. |
| `ORION_REVIEWER_MODEL` | `gemini-3-flash-preview` | Code reviewer. |
| `ORION_DEBUGGING_MODEL` | `glm-5-maas` | Debugging agent. |
| `ORION_SUMMARIZER_MODEL` | `gemini-flash-lite` | Context summarization. |
| `ORION_EMBEDDING_MODEL` | `text-embedding-005` | Vertex embedding model (used when `ORION_EMBEDDINGS=gemini`). |
| `ORION_TEMPERATURE` | `0.2` | `0.0`–`2.0`. |
| `ORION_MAX_OUTPUT_TOKENS` | `8192` | Per response. |

## Behaviour

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_MODE` | `build` | `build` or `plan`. |
| `ORION_APPROVAL` | `ask` | `ask`, `auto`, `deny`. |
| `ORION_MAX_ITERATIONS` | `3` | Evaluator → debugger → generator loops. |
| `ORION_MAX_TOOL_ROUNDS` | `12` | Tool rounds per agent turn. |
| `ORION_DRY_RUN` | `false` | Record changes without writing. |
| `ORION_AUTO_INDEX` | `true` | Refresh the index before a run. |
| `ORION_SESSION_PERSIST` | `true` | Write session metadata and events. |
| `ORION_COMMAND_TIMEOUT` | `120` | Shell timeout in seconds. |

## Retrieval

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_EMBEDDINGS` | `local` | `local` (offline hashing, default), `gemini` (Vertex embeddings via ADC), or `auto`. |
| `ORION_TOP_K` | `8` | Chunks retrieved per run (`1`–`50`). |
| `ORION_CHUNK_CHARS` | `1200` | Chunk size (minimum `200`). |
| `ORION_CHUNK_OVERLAP` | `150` | Must be smaller than `ORION_CHUNK_CHARS`. |
| `ORION_MAX_FILE_BYTES` | `512000` | Skip larger files. |

The offline `local` embedder is deterministic and requires no API key. It is the
default so existing indexes (built with `local-hash-v1`) remain compatible. Set
`ORION_EMBEDDINGS=gemini` to use Vertex embeddings through the same ADC
credentials.

## Storage and paths

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_DATA_DIR` | `<workspace>/.orion` | Sessions, manifest, vector store. |
| `ORION_CHROMA_DIR` | `<data_dir>/chroma` | Override the vector store location. |
| `ORION_BACKUP` | `true` | Keep `.bak` copies when editing files. |
| `ORION_WORKSPACE` | current directory | Project root. CLI: `--workspace/-C`. |

## UI and logging

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_THEME` | `orion` | TUI theme name. |
| `ORION_LOG_LEVEL` | `WARNING` | Python logging level. |

## Managing configuration

```bash
orion config show                 # effective values (secrets masked)
orion config show --json
orion config set ORION_MODEL gemini-3.1-pro-preview
orion config set ORION_LLM_PROVIDER local
orion config set ORION_LOCAL_MODEL qwen2.5:7b-instruct
orion config set ORION_APPROVAL auto --file .env
orion config path                 # where Orion looks for .env
```

In the TUI, `/model <name>` switches models for the session and `/approve`
cycles the approval policy.

## Library overrides

```python
from orion import OrionConfig

config = OrionConfig.load(
    workspace="~/code/project",
    model="gemini-3.1-pro-preview",
    approval="auto",
    embeddings="local",
)

local = OrionConfig.load(
    workspace="~/code/project",
    llm_provider="local",
    local_model="qwen2.5:7b-instruct",
    approval="auto",
)
```

`config.with_overrides(**changes)` returns a copy without mutating the original.
