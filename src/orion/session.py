"""Session persistence: JSON metadata plus an append-only event log."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from orion.config import OrionConfig
from orion.utils import now_iso, slugify


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: str
    updated_at: str
    cwd: str
    mode: str
    model: str
    status: str = "running"
    usage: dict[str, int] = field(default_factory=dict)
    final_preview: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    meta: SessionMeta
    directory: Path

    @property
    def meta_path(self) -> Path:
        return self.directory / f"{self.meta.id}.json"

    @property
    def events_path(self) -> Path:
        return self.directory / f"{self.meta.id}.jsonl"


class SessionStore:
    """Persist sessions under ``<data_dir>/sessions``."""

    def __init__(self, config: OrionConfig) -> None:
        self.config = config
        self.directory = config.session_path
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, task: str, mode: str, model: str) -> Session:
        stamp = now_iso()
        session_id = (
            f"{stamp.replace(':', '').replace('-', '')[:15]}-"
            f"{slugify(task, 32)}-{uuid.uuid4().hex[:4]}"
        )
        meta = SessionMeta(
            id=session_id,
            title=task.strip().splitlines()[0][:100] if task.strip() else "Untitled session",
            created_at=stamp,
            updated_at=stamp,
            cwd=str(self.config.workspace),
            mode=mode,
            model=model,
        )
        session = Session(meta=meta, directory=self.directory)
        self._write_meta(session)
        session.events_path.touch()
        return session

    def _write_meta(self, session: Session) -> None:
        temp = session.meta_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(session.meta.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp.replace(session.meta_path)

    def append(self, session: Session, event: dict[str, Any]) -> None:
        if event.get("type") == "token":
            return
        payload = dict(event)
        if "result" in payload and isinstance(payload["result"], dict):
            payload["result"] = {
                key: value
                for key, value in payload["result"].items()
                if key in {"task", "mode", "final", "usage", "session_id", "duration", "error", "cancelled"}
            }
        try:
            with session.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def finish(self, session: Session, *, status: str, usage: dict[str, int], final: str) -> None:
        session.meta.updated_at = now_iso()
        session.meta.status = status
        session.meta.usage = usage
        session.meta.final_preview = final.strip()[:400]
        try:
            self._write_meta(session)
        except OSError:
            pass

    def list(self, limit: int = 20) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metas.append(SessionMeta(**data))
            except (OSError, ValueError, TypeError):
                continue
        metas.sort(key=lambda meta: meta.updated_at, reverse=True)
        return metas[:limit]

    def latest(self) -> SessionMeta | None:
        metas = self.list(limit=1)
        return metas[0] if metas else None

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self.directory / f"{session_id}.jsonl"
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            return []
        return events

    def load_meta(self, session_id: str) -> SessionMeta | None:
        path = self.directory / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            return SessionMeta(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def delete(self, session_id: str) -> bool:
        removed = False
        for suffix in (".json", ".jsonl"):
            path = self.directory / f"{session_id}{suffix}"
            if path.is_file():
                try:
                    path.unlink()
                    removed = True
                except OSError:
                    pass
        return removed


__all__ = ["Session", "SessionMeta", "SessionStore"]
