"""Run orchestration: config -> RAG -> LangGraph -> events."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from orion.agents.graph import build_graph
from orion.agents.nodes import AgentRuntime
from orion.approval import ApprovalCallback, build_approval
from orion.config import Mode, OrionConfig
from orion.errors import CancelledError
from orion.llm import LLM, build_llm
from orion.rag.embeddings import build_embedder
from orion.rag.indexer import IndexStats
from orion.rag.retriever import CodeRetriever
from orion.session import Session, SessionStore
from orion.tools.registry import ToolRegistry


@dataclass
class RunResult:
    """Everything a run produced."""

    task: str
    mode: str
    final: str
    plan: dict[str, Any] = field(default_factory=dict)
    plan_text: str = ""
    changes: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    tests: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    session_id: str | None = None
    duration: float = 0.0
    cancelled: bool = False
    error: str | None = None
    context: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 2)
        return data


class OrionEngine:
    """Stateful facade around the agent graph.

    The engine is safe to reuse across runs in one process. A run is executed
    on a background thread by :meth:`stream`, or synchronously by :meth:`run`.
    """

    def __init__(
        self,
        config: OrionConfig,
        llm: LLM | None = None,
        approval: ApprovalCallback | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or build_llm(config)
        self.approval = approval
        self.retriever: CodeRetriever | None = None
        self.registry: ToolRegistry | None = None
        self._index_checked = False
        self._cancel = threading.Event()
        self._last_result: RunResult | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def prepare(
        self,
        config: OrionConfig | None = None,
        emit: Any | None = None,
        index: bool = True,
    ) -> None:
        config = config or self.config
        emit = emit or (lambda event: None)
        if self.retriever is None:
            self.retriever = CodeRetriever(
                config,
                on_warning=lambda message: emit({"type": "warning", "message": message}),
            )
        if self.registry is None:
            self.registry = ToolRegistry(config, self.retriever)
        if index and config.auto_index and not self._index_checked:
            self._index_checked = True
            try:
                self.retriever.ensure_index(on_event=emit)
            except Exception as exc:
                emit(
                    {
                        "type": "warning",
                        "message": (
                            f"Automatic indexing failed: {exc}. "
                            "Continuing with lexical search; run `orion index` to retry."
                        ),
                    }
                )

    def index(
        self,
        force: bool = False,
        embeddings: str | None = None,
        on_event: Any | None = None,
    ) -> IndexStats:
        config = self.config
        if embeddings:
            config = config.with_overrides(embeddings=embeddings)
        embedder = build_embedder(config, self.llm) if embeddings else None
        retriever = CodeRetriever(
            config,
            embedder=embedder,
            on_warning=lambda message: on_event({"type": "warning", "message": message})
            if on_event
            else None,
        )
        stats = retriever.index(force=force, on_event=on_event)
        self.retriever = retriever
        self.registry = ToolRegistry(config, retriever)
        self._index_checked = True
        return stats

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        self.prepare(index=False)
        assert self.retriever is not None
        return [hit.as_dict() for hit in self.retriever.search(query, k=k)]

    def status(self) -> dict[str, Any]:
        self.prepare(index=False)
        assert self.retriever is not None
        return {
            "config": self.config.describe(),
            "index": self.retriever.status(),
        }

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        *,
        mode: Mode | None = None,
        model: str | None = None,
        dry_run: bool | None = None,
        on_event: Any | None = None,
    ) -> RunResult:
        return self._execute(
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
        """Yield live events, ending with a ``done`` event containing the result."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def worker() -> None:
            try:
                self._execute(
                    task,
                    mode=mode,
                    model=model,
                    dry_run=dry_run,
                    on_event=events.put,
                )
            except Exception as exc:
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                events.put({"type": "done", "result": None})
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, name="orion-run", daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event
        thread.join(timeout=1)

    # ------------------------------------------------------------------
    def _execute(
        self,
        task: str,
        *,
        mode: Mode | None = None,
        model: str | None = None,
        dry_run: bool | None = None,
        on_event: Any | None = None,
    ) -> RunResult:
        task = (task or "").strip()
        emit = on_event or (lambda event: None)
        if not task:
            result = RunResult(task=task, mode=mode or self.config.mode, final="Nothing to do.")
            emit({"type": "done", "result": result.as_dict()})
            return result

        started = time.monotonic()
        run_config = self.config.with_overrides(model=model) if model else self.config
        run_mode: Mode = mode or run_config.mode
        run_dry = run_config.dry_run if dry_run is None else dry_run
        self._cancel = threading.Event()

        session_store: SessionStore | None = None
        session: Session | None = None
        if run_config.session_persist:
            try:
                session_store = SessionStore(run_config)
                session = session_store.create(task, run_mode, run_config.active_model)
            except Exception as exc:
                emit({"type": "warning", "message": f"Session persistence disabled: {exc}"})

        def handle(event: dict[str, Any]) -> None:
            emit(event)
            if session_store is not None and session is not None:
                session_store.append(session, event)

        result = RunResult(
            task=task,
            mode=run_mode,
            final="",
            session_id=session.meta.id if session else None,
        )
        handle(
            {
                "type": "start",
                "task": task,
                "mode": run_mode,
                "model": run_config.active_model,
                "workspace": str(run_config.workspace),
                "session_id": result.session_id,
                "dry_run": run_dry,
            }
        )

        runtime: AgentRuntime | None = None
        try:
            self.prepare(run_config, emit=handle)
            if self.registry is None:  # pragma: no cover - prepare always sets it
                raise RuntimeError("Tool registry was not initialized")
            runtime = AgentRuntime(
                config=run_config,
                llm=self.llm,
                registry=self.registry,
                retriever=self.retriever,
                emit=handle,
                approval=build_approval(run_config, self.approval),
                cancel=self._cancel,
                dry_run=run_dry,
            )
            graph = build_graph(runtime)
            state = graph.invoke(
                {"task": task, "mode": run_mode, "iterations": 0},
                config={"recursion_limit": 80},
            )
            result.plan = state.get("plan") or {}
            result.plan_text = state.get("plan_text") or ""
            result.context = state.get("context") or []
            result.changes = [change.as_dict() for change in runtime.changes]
            result.review = state.get("review") or {}
            result.evaluation = state.get("evaluation") or {}
            result.tests = state.get("test_results") or {}
            result.final = (
                state.get("final")
                or state.get("coder_summary")
                or "The run finished without a summary."
            )
        except CancelledError:
            result.cancelled = True
            result.final = "Run cancelled."
            handle({"type": "cancelled", "message": result.final})
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.final = f"Run failed: {exc}"
            handle({"type": "error", "message": str(exc)})
        finally:
            if runtime is not None:
                result.usage = runtime.usage.as_dict()
                result.changes = [change.as_dict() for change in runtime.changes]
            result.duration = time.monotonic() - started
            if session_store is not None and session is not None:
                status = "cancelled" if result.cancelled else ("failed" if result.error else "completed")
                session_store.finish(
                    session,
                    status=status,
                    usage=result.usage,
                    final=result.final,
                )
            self._last_result = result
            handle({"type": "done", "result": result.as_dict()})

        return result


__all__ = ["OrionEngine", "RunResult"]
