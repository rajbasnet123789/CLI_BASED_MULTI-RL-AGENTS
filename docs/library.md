# Library usage

Orion can be embedded in scripts, services, and notebooks. The public surface
is small and stable.

```python
from orion import Orion, OrionConfig, RunResult

orion = Orion(OrionConfig.load())
result: RunResult = orion.run("Add a /health endpoint", mode="build")
print(result.final)
```

## Constructing an instance

```python
from orion import Orion, OrionConfig

# Read .env files and environment variables.
orion = Orion.from_env(workspace="~/code/project")

# Or build a configuration explicitly (load() expands ~ and reads .env files).
config = OrionConfig.load(
    workspace="~/code/project",
    model="gemini-3.1-pro-preview",
    approval="auto",
    embeddings="local",
)
orion = Orion(config)
```

Pass `llm=` to inject a compatible client (useful for tests) and `approval=`
to inject an approval callback.

## Running tasks

```python
result = orion.run(
    "Refactor the parser and add tests",
    mode="build",          # "build" or "plan"
    model="gemini-3.1-pro-preview",# optional per-run override
    dry_run=True,          # compute diffs without writing
    on_event=print,        # optional event callback
)
```

`RunResult` fields:

| Field | Type | Description |
| --- | --- | --- |
| `task` | `str` | The original task. |
| `mode` | `str` | `build` or `plan`. |
| `final` | `str` | Final markdown report. |
| `plan` | `dict` | Structured plan (build mode). |
| `plan_text` | `str` | Rendered plan. |
| `changes` | `list[dict]` | Files created/edited/moved/deleted, with diffs. |
| `review` | `dict` | `verdict`, `summary`, `issues`, `suggestions`. |
| `tests` | `dict` | Test status and output. |
| `usage` | `dict` | Token usage. |
| `session_id` | `str \| None` | Persisted session id. |
| `duration` | `float` | Seconds. |
| `cancelled` | `bool` | Whether the run was cancelled. |
| `error` | `str \| None` | Fatal error message. |
| `context` | `list[dict]` | Retrieved chunks. |

## Streaming

```python
for event in orion.stream("Explain the auth flow", mode="plan"):
    if event["type"] == "token":
        print(event["text"], end="")
    elif event["type"] == "done":
        result = event["result"]
```

See [events.md](events.md) for the full event catalogue.

## Indexing and search

```python
stats = orion.index(force=False, embeddings="local")
print(stats["files_indexed"], stats["vectors"])

hits = orion.search("database connection", k=5)
for hit in hits:
    print(hit["path"], hit["start_line"], hit["score"])
```

## Status and cancellation

```python
print(orion.status())   # {"config": ..., "index": ...}

orion.cancel()          # request cancellation of the active run
```

Cancellation is cooperative: the current tool call finishes, then the engine
raises `CancelledError` and the run ends with `cancelled=True`.

## Async and servers

The engine is synchronous and executes one run at a time. Wrap it in a thread
or thread pool for async frameworks. The bundled FastAPI app serialises runs
with a lock and is available via `orion.server.create_app()`.
