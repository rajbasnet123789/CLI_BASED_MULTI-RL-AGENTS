# Task Manager

A small full-stack app generated as a test of [Orion AI](https://github.com/orion-ai/orion):
FastAPI + SQLAlchemy with SQLite on the backend, React + Vite on the frontend.

Features: create/rename/delete projects, create/edit/delete tasks, filter tasks by
status and project, move a task through `pending → in-progress → completed`, and a
responsive UI.

## Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the API from the repository root (the package uses relative imports):

```bash
uvicorn backend.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- SQLite database: `sql_app.db` (created in the repository root on startup)

Run the tests from the repository root:

```bash
python -m pytest backend/test_main.py -q
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The API base URL defaults to `http://localhost:8000`;
override it with `VITE_API_URL` in a `frontend/.env` file if needed. In this
environment port 8000 was already taken, so `frontend/.env` points the app at
`http://localhost:8011` and the API is started with `--port 8011`.

Production build:

```bash
npm run build
npm run preview
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/projects/` | Create a project |
| `GET` | `/projects/` | List projects |
| `GET` | `/projects/{id}` | Get a project |
| `PUT` | `/projects/{id}` | Rename a project |
| `DELETE` | `/projects/{id}` | Delete a project |
| `POST` | `/tasks/` | Create a task (`title`, `project_id`, optional `description`, `status`) |
| `GET` | `/tasks/?project_id=&status=` | List/filter tasks |
| `GET` | `/tasks/{id}` | Get a task |
| `PUT` | `/tasks/{id}` | Update a task |
| `DELETE` | `/tasks/{id}` | Delete a task |
