"""LangGraph wiring for the Orion agent team."""

from __future__ import annotations

from functools import partial
from typing import Any

from orion.agents.nodes import (
    AgentRuntime,
    coder_node,
    context_node,
    debugger_node,
    evaluator_node,
    finalizer_node,
    planner_node,
    reviewer_node,
    summarizer_node,
)
from orion.agents.state import AgentState


def build_graph(runtime: AgentRuntime):
    """Compile the architect -> generator -> reviewer -> evaluator state machine."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError(
            f"langgraph could not be imported: {exc}. "
            "Reinstall Orion: pip install --upgrade 'orion-ai'"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("context", partial(context_node, runtime))
    graph.add_node("summarizer", partial(summarizer_node, runtime))
    graph.add_node("planner", partial(planner_node, runtime))
    graph.add_node("coder", partial(coder_node, runtime))
    graph.add_node("reviewer", partial(reviewer_node, runtime))
    graph.add_node("evaluator", partial(evaluator_node, runtime))
    graph.add_node("debugger", partial(debugger_node, runtime))
    graph.add_node("finalize", partial(finalizer_node, runtime))

    def after_plan(state: AgentState) -> str:
        return "finalize" if state.get("mode") == "plan" else "coder"

    def after_evaluation(state: AgentState) -> str:
        evaluation: dict[str, Any] = state.get("evaluation") or {}
        iterations = int(state.get("iterations", 0) or 0)
        if (
            evaluation.get("verdict") == "needs_changes"
            and iterations < runtime.config.max_iterations
        ):
            return "debugger"
        return "finalize"

    graph.add_edge(START, "context")
    graph.add_edge("context", "summarizer")
    graph.add_edge("summarizer", "planner")
    graph.add_conditional_edges(
        "planner",
        after_plan,
        {"finalize": "finalize", "coder": "coder"},
    )
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        after_evaluation,
        {"debugger": "debugger", "finalize": "finalize"},
    )
    graph.add_edge("debugger", "coder")
    graph.add_edge("finalize", END)

    return graph.compile()


def graph_mermaid() -> str:
    """Return a Mermaid diagram of the agent graph (used in the docs)."""
    return """flowchart TD
    START([start]) --> context[context / RAG]
    context --> summarizer[context summarizer]
    summarizer --> architect[principal architect / planner]
    architect -->|plan mode| finalize[finalize]
    architect -->|build mode| generator[code generator + tools]
    generator --> reviewer[code reviewer]
    reviewer --> evaluator[code evaluator + real checks]
    evaluator -->|needs changes| debugger[debugging agent]
    debugger --> generator
    evaluator -->|approved| finalize
    finalize --> END([end])"""


__all__ = ["build_graph", "graph_mermaid"]
