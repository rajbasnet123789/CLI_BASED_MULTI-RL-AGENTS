"""LangGraph wiring for the Orion agent team."""

from __future__ import annotations

from functools import partial
from typing import Any

from orion.agents.nodes import (
    AgentRuntime,
    coder_node,
    context_node,
    debugger_node,
    finalizer_node,
    planner_node,
    reviewer_node,
    tester_node,
)
from orion.agents.state import AgentState


def build_graph(runtime: AgentRuntime):
    """Compile the planner -> builder -> reviewer -> tester state machine."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError(
            "langgraph is not installed. Reinstall Orion: pip install 'orion-ai'"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("context", partial(context_node, runtime))
    graph.add_node("planner", partial(planner_node, runtime))
    graph.add_node("coder", partial(coder_node, runtime))
    graph.add_node("reviewer", partial(reviewer_node, runtime))
    graph.add_node("debugger", partial(debugger_node, runtime))
    graph.add_node("tester", partial(tester_node, runtime))
    graph.add_node("finalize", partial(finalizer_node, runtime))

    def after_plan(state: AgentState) -> str:
        return "finalize" if state.get("mode") == "plan" else "coder"

    def after_review(state: AgentState) -> str:
        review: dict[str, Any] = state.get("review") or {}
        iterations = int(state.get("iterations", 0) or 0)
        if review.get("verdict") == "needs_changes" and iterations < runtime.config.max_iterations:
            return "debugger"
        return "tester"

    graph.add_edge(START, "context")
    graph.add_edge("context", "planner")
    graph.add_conditional_edges(
        "planner",
        after_plan,
        {"finalize": "finalize", "coder": "coder"},
    )
    graph.add_edge("coder", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        after_review,
        {"debugger": "debugger", "tester": "tester"},
    )
    graph.add_edge("debugger", "coder")
    graph.add_edge("tester", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def graph_mermaid() -> str:
    """Return a Mermaid diagram of the agent graph (used in the docs)."""
    return """flowchart TD
    START([start]) --> context[context / RAG]
    context --> planner[planner]
    planner -->|plan mode| finalize[finalize]
    planner -->|build mode| coder[coder + tools]
    coder --> reviewer[reviewer]
    reviewer -->|needs changes| debugger[debugger]
    debugger --> coder
    reviewer -->|approved| tester[tester]
    tester --> finalize
    finalize --> END([end])"""


__all__ = ["build_graph", "graph_mermaid"]
