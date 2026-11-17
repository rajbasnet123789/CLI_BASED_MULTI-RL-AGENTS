# Configuration

Orion reads configuration from, in order of increasing priority:

1. Built-in defaults.
2. `~/.orion/.env`.
3. `.env` files from the project root down to the workspace (nearest wins).
4. `ORION_*` environment variables.
5. Explicit constructor arguments or CLI flags.

The key itself may be supplied as `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or
`ORION_GEMINI_API_KEY`.

## Credentials

| Variable | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Google AI Studio (`AIza...`) or Vertex express (`AQ....`). |
| `ORION_VERTEX` | `auto` | `auto` detects from the key prefix; `true`/`false` force the mode. |

In `auto` mode Orion infers the backend from the key prefix: `AQ.` keys use
`google-genai` in Vertex express mode and `AIza` keys use the standard Gemini
API. Prefixes are only a hint — if the chosen backend rejects the key (for
example, the Vertex AI API is disabled on the key's project), Orion retries the
request once with the other backend and pins the working mode for the rest of
the process. Set `ORION_VERTEX=false` to force Google AI Studio or
`ORION_VERTEX=true` to force Vertex AI when the guess is wrong or you want to
skip the extra round trip.

## Models

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_MODEL` | `gemini-3.5-flash` | Default for every agent. Aliases: `flash`, `pro`, `flash-lite`, `3.5-flash`, `3.5-flash-lite`, `3.1-pro`, `2.5-flash`, `2.5-flash-lite`, `2.5-pro`, `2.0-flash`. |
| `ORION_PLANNER_MODEL` | unset | Override for the planner. |
| `ORION_CODER_MODEL` | unset | Override for the coder. |
| `ORION_REVIEWER_MODEL` | unset | Override for the reviewer. |
| `ORION_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model. |
| `ORION_TEMPERATURE` | `0.2` | `0.0`–`2.0`. |
| `ORION_MAX_OUTPUT_TOKENS` | `8192` | Per response. |

## Behaviour

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_MODE` | `build` | `build` or `plan`. |
| `ORION_APPROVAL` | `ask` | `ask`, `auto`, `deny`. |
| `ORION_MAX_ITERATIONS` | `3` | Reviewer → debugger → coder loops. |
| `ORION_MAX_TOOL_ROUNDS` | `12` | Tool rounds per agent turn. |
| `ORION_DRY_RUN` | `false` | Record changes without writing. |
| `ORION_AUTO_INDEX` | `true` | Refresh the index before a run. |
| `ORION_SESSION_PERSIST` | `true` | Write session metadata and events. |
| `ORION_COMMAND_TIMEOUT` | `120` | Shell timeout in seconds. |

## Retrieval

| Variable | Default | Notes |
| --- | --- | --- |
| `ORION_EMBEDDINGS` | `auto` | `gemini`, `local`, or `auto` (Gemini when a key exists, otherwise local). |
| `ORION_TOP_K` | `8` | Chunks retrieved per run (`1`–`50`). |
| `ORION_CHUNK_CHARS` | `1200` | Chunk size (minimum `200`). |
| `ORION_CHUNK_OVERLAP` | `150` | Must be smaller than `ORION_CHUNK_CHARS`. |
| `ORION_MAX_FILE_BYTES` | `512000` | Skip larger files. |

The offline `local` embedder is deterministic and requires no API key, which is
useful for tests, air-gapped machines, and CI.

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
```

`config.with_overrides(**changes)` returns a copy without mutating the original.
