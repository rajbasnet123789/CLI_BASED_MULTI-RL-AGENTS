"""Public library API for Orion AI."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from orion.config import Mode, OrionConfig
from orion.engine import OrionEngine, RunResult


class Orion:
    """Programmatic entry point.

    Example::

        orion = Orion(OrionConfig.load())
        result = orion.run("Add a health endpoint", mode="build")
        print(result.final)

    Events are dictionaries with a ``type`` key; see ``docs/events.md`` for the
    full list.
    """

    def __init__(
        self,
        config: OrionConfig | None = None,
        *,
        llm: Any | None = None,
        approval: Any | None = None,
    ) -> None:
        self.config = config or OrionConfig.load()
        self.engine = OrionEngine(self.config, llm=llm, approval=approval)

    @classmethod
    def from_env(cls, workspace: str | None = None) -> Orion:
        return cls(OrionConfig.load(workspace=workspace))

    def run(
        self,
        task: str,
        *,
        mode: Mode | None = None,
        model: str | None = None,
        dry_run: bool | None = None,
        on_event: Any | None = None,
    ) -> RunResult:
        """Run a task to completion and return the result."""
        return self.engine.run(
            task,
            mode=mode,
            model=model,
            dry_run=dry_run,
            on_event=on_event,
        )

    def stream(
        self,
        task: str,
        *,
        mode: Mode | None = None,
        model: str | None = None,
        dry_run: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream structured events while the task runs."""
        return self.engine.stream(task, mode=mode, model=model, dry_run=dry_run)

    def index(
        self,
        *,
        force: bool = False,
        embeddings: str | None = None,
        on_event: Any | None = None,
    ) -> dict[str, Any]:
        """Build or refresh the code index."""
        return self.engine.index(force=force, embeddings=embeddings, on_event=on_event).as_dict()

    def search(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        """Search the code index (hybrid vector + lexical)."""
        return self.engine.search(query, k=k)

    def status(self) -> dict[str, Any]:
        """Return effective configuration and index status."""
        return self.engine.status()

    def cancel(self) -> None:
        """Request cancellation of the active run."""
        self.engine.cancel()

    def prepare(self, on_event: Any | None = None) -> None:
        """Warm up the index and tool registry without running an agent."""
        self.engine.prepare(emit=on_event)


__all__ = ["Orion", "RunResult"]
