# HackVerse 2026 — 3D College Hackathon Site

A small but polished sample project used to smoke-test Orion's multi-agent
pipeline. It has:

- a **static 3D frontend** (Three.js particle field + wireframe shapes) for a
  college hackathon landing page, and
- a **simple Flask backend** with in-memory data for tracks, schedule, teams,
  and registration.

## Run it

```bash
pip install -r requirements.txt
python server.py
```

Open http://localhost:8000.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + event name. |
| `GET` | `/api/tracks` | Hackathon tracks. |
| `GET` | `/api/schedule` | Event timeline. |
| `GET` | `/api/teams` | Registered teams. |
| `POST` | `/api/register` | Register a team (JSON: `name`, `track`, `members`). |
| `GET` | `/api/stats` | Aggregate counts — **not implemented yet** (Orion task). |

The frontend shows a live "stats" strip that calls `/api/stats` and falls back
to placeholder dashes until the endpoint exists.

## Tests

```bash
python -m pytest -q
```
