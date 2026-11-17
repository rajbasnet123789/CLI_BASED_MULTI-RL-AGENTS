# Docker

Orion ships a multi-stage image that installs the package with the `server`
extra and runs as an unprivileged user (`uid 1000`).

## Build

```bash
cp .env.example .env      # add GEMINI_API_KEY
docker build -t orion-ai:1.0.1 .
```

The build context excludes `.env`, `.orion/`, `.venv/`, tests, and docs (see
`.dockerignore`), so secrets and local state are never copied into the image.

## Interactive TUI

```bash
docker compose run --rm orion
```

The Compose file mounts the project at `/workspace`, sets
`ORION_DATA_DIR=/workspace/.orion`, and keeps stdin/tty open for the TUI.

## Headless runs

```bash
docker compose run --rm orion run "add tests for the parser"
docker compose run --rm orion plan "explain the request lifecycle"
docker compose run --rm orion index --embeddings local
```

## HTTP API

```bash
docker compose --profile api up api
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/run \
  -H 'content-type: application/json' \
  -d '{"task": "summarise the architecture", "mode": "plan"}'
```

The API service exposes port `8765` and has a `/health` healthcheck.

## Passing configuration

Any `ORION_*` variable can be supplied through `env_file` or `environment`:

```yaml
services:
  orion:
    environment:
      ORION_MODEL: gemini-3.1-pro-preview
      ORION_APPROVAL: auto
```

Or mount a dotenv file read-only:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  -v "$PWD/.env:/workspace/.env:ro" \
  -w /workspace \
  orion-ai:1.0.1 run "fix the failing test"
```

## Notes

- Shell tools run inside the container. The workspace is the mounted directory;
  the image includes `git` and `ca-certificates` but not your language toolchain.
  Extend the runtime stage if you need Node, Go, etc.
- Use a named volume for `/workspace/.orion` if you want the index and sessions
  to survive container re-creation.
- Never bake API keys into the image; pass them at runtime.
