# Events

Every run emits a stream of dictionaries with a `type` key. Pass a callback to
`Orion.run(..., on_event=...)` or iterate `Orion.stream(...)`.

The stream always starts with `start` and ends with `done` (or `error`). When a
run is cancelled it ends with `cancelled` followed by `done`.

| `type` | Payload | Meaning |
| --- | --- | --- |
| `start` | `task`, `mode`, `model`, `workspace`, `session_id`, `dry_run` | A run began. |
| `status` | `agent`, `message` | Progress update from an agent. |
| `token` | `agent`, `text` | Streamed text delta (TUI only; not persisted). |
| `message` | `agent`, `text` | A complete agent message. |
| `plan` | `plan`, `markdown` | Plan produced. In plan mode `plan` is `{"markdown": ...}`. |
| `context` | `hits`, `count`, `context_chars` | Retrieved RAG context. |
| `tool_call` | `agent`, `call_id`, `name`, `args` | The model requested a tool. |
| `tool_result` | `agent`, `call_id`, `name`, `ok`, `content`, `display`, `changed_paths` | Tool finished. |
| `diff` | `agent`, `path`, `diff` | Unified diff for a file change. |
| `changes` | `changes`, `new_changes` | Running change log after the coder. |
| `review` | `verdict`, `summary`, `issues`, `suggestions` | Reviewer verdict. |
| `test` | `status`, `command`, `framework`, `exit_code`, `output`, `reason` | Test result (`passed`, `failed`, `skipped`). |
| `usage` | `usage`, `agent` | Cumulative token usage (`prompt_tokens`, `output_tokens`, `total_tokens`, `calls`). |
| `warning` | `message` (sometimes `agent`) | Non-fatal problem. |
| `error` | `message` | The run failed. |
| `cancelled` | `message` | The run was cancelled. |
| `done` | `result` | Final `RunResult.as_dict()`, or `null` on a fatal error. |
| `index_status` | `phase`, `message`, `stats`, `current`, `total` | Indexing progress (`scan`, `chunk`, `embed`, `write`, `done`, `error`). |

## Example

```python
from orion import Orion, OrionConfig

orion = Orion(OrionConfig.load())

for event in orion.stream("Add a /health endpoint"):
    kind = event["type"]
    if kind == "tool_call":
        print("→", event["name"], event["args"])
    elif kind == "diff":
        print(event["diff"])
    elif kind == "done":
        result = event["result"]
        print("finished:", result["mode"], result["duration"])
```

## Session logs

When `ORION_SESSION_PERSIST=true`, every event except `token` is appended to
`<data_dir>/sessions/<id>.events.jsonl`. Replay a session with:

```bash
orion sessions show <session-id>
```
