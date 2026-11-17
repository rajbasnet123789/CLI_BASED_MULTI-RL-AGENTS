"""Optional HTTP API for Orion AI (install with ``orion-ai[server]``)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orion.config import OrionConfig
from orion.engine import OrionEngine
from orion.version import __version__


class RunRequest(BaseModel):
    task: str = Field(min_length=1, description="What Orion should do.")
    mode: Literal["build", "plan"] | None = None
    model: str | None = None
    dry_run: bool | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[dict[str, Any]]


def create_app(config: OrionConfig | None = None) -> FastAPI:
    """Build the FastAPI app.

    A single engine is reused so the index is opened once, but runs, indexing,
    and streams are serialised with a lock: one engine executes one task at a
    time.
    """
    config = config or OrionConfig.load()
    engine = OrionEngine(config)
    run_lock = threading.Lock()

    app = FastAPI(
        title="Orion AI",
        version=__version__,
        description=(
            "Multi-agent coding assistant API. Every endpoint is a thin wrapper "
            "around the same engine used by the CLI and TUI."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": config.llm_provider,
            "model": config.active_model,
            "workspace": str(config.workspace),
            "api_key": config.has_api_key,
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        return engine.status()

    @app.get("/config")
    def get_config() -> dict[str, Any]:
        return config.describe()

    @app.post("/run")
    def run(request: RunRequest) -> dict[str, Any]:
        try:
            with run_lock:
                result = engine.run(
                    request.task,
                    mode=request.mode,
                    model=request.model,
                    dry_run=request.dry_run,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.as_dict()

    @app.post("/stream")
    def stream(request: RunRequest) -> StreamingResponse:
        def events() -> Iterator[str]:
            try:
                with run_lock:
                    for event in engine.stream(
                        request.task,
                        mode=request.mode,
                        model=request.model,
                        dry_run=request.dry_run,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except GeneratorExit:
                engine.cancel()
                raise
            except Exception as exc:
                payload = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/index")
    def index(force: bool = False, embeddings: str | None = None) -> dict[str, Any]:
        try:
            with run_lock:
                stats = engine.index(force=force, embeddings=embeddings)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return stats.as_dict()

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(min_length=1, description="Search query."),
        k: int = Query(8, ge=1, le=50),
    ) -> SearchResponse:
        return SearchResponse(query=q, hits=engine.search(q, k=k))

    return app


__all__ = ["create_app"]
