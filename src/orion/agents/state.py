"""LangGraph state shared by the Orion agents."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    task: str
    mode: str
    repo_overview: str
    context: list[dict[str, Any]]
    context_text: str
    plan: dict[str, Any]
    plan_text: str
    fix_notes: str
    coder_summary: str
    changes: list[dict[str, Any]]
    review: dict[str, Any]
    issues: list[str]
    test_results: dict[str, Any]
    iterations: int
    final: str
    error: str | None


__all__ = ["AgentState"]
