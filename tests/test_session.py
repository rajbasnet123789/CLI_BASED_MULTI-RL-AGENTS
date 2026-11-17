"""Session persistence."""

from __future__ import annotations

from orion.config import OrionConfig
from orion.session import SessionStore


def test_session_lifecycle(config: OrionConfig) -> None:
    store = SessionStore(config)
    session = store.create("Add a health endpoint", "build", "gemini-3.5-flash")
    assert session.meta.id
    assert session.meta.title == "Add a health endpoint"
    assert session.meta_path.is_file()
    assert session.events_path.is_file()

    store.append(session, {"type": "start", "task": "Add a health endpoint"})
    store.append(session, {"type": "token", "text": "ignored"})
    store.append(session, {"type": "message", "agent": "coder", "text": "done"})
    store.finish(
        session,
        status="completed",
        usage={"total_tokens": 42, "calls": 2},
        final="All done.",
    )

    metas = store.list()
    assert len(metas) == 1
    assert metas[0].status == "completed"
    assert metas[0].usage["total_tokens"] == 42
    assert metas[0].final_preview == "All done."
    assert store.latest() is not None

    events = store.load_events(session.meta.id)
    assert [event["type"] for event in events] == ["start", "message"]
    assert store.load_meta(session.meta.id).title == "Add a health endpoint"

    assert store.delete(session.meta.id) is True
    assert store.list() == []
    assert store.load_events(session.meta.id) == []


def test_session_store_handles_corrupt_metadata(config: OrionConfig) -> None:
    store = SessionStore(config)
    bad = store.directory / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    assert store.list() == []
    assert store.load_meta("broken") is None
    assert store.delete("missing") is False
