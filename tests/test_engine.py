"""End-to-end graph orchestration with a fake model (no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from orion.config import OrionConfig
from orion.engine import OrionEngine
from orion.llm import FunctionCall, LLMResponse


class FakeLLM:
    """Deterministic stand-in for GeminiLLM."""

    def __init__(self, *, tool_call: bool = False) -> None:
        self.tool_call = tool_call
        self.coder_calls = 0
        self.systems: list[str] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: Any | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system = system or ""
        self.systems.append(system)
        if "SUMMARIZER" in system:
            return LLMResponse(
                text="- Relevant files: pkg/app.py\n- Test command: pytest",
                prompt_tokens=4,
                output_tokens=4,
                model=model or "fake",
            )
        if "EVALUATOR" in system:
            return LLMResponse(
                text=json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Deterministic checks and diff look good.",
                        "issues": [],
                        "evidence": ["no changes" if not self.tool_call else "diff reviewed"],
                    }
                ),
                prompt_tokens=10,
                output_tokens=5,
                model=model or "fake",
            )
        if "PLANNER" in system:
            if tools is not None:
                # Plan mode uses the tool loop and expects a markdown plan.
                return LLMResponse(
                    text="## Plan\n\n1. Add the route in `pkg/app.py`.\n\n"
                    "No files were changed.",
                    prompt_tokens=10,
                    output_tokens=5,
                    model=model or "fake",
                )
            return LLMResponse(
                text=json.dumps(
                    {
                        "goal": "Add a health endpoint",
                        "steps": [
                            {
                                "title": "Add the route",
                                "detail": "in app.py",
                                "files": ["pkg/app.py"],
                            }
                        ],
                        "risks": [],
                        "test_plan": "pytest",
                    }
                ),
                prompt_tokens=10,
                output_tokens=5,
                model=model or "fake",
            )
        if "BUILDER" in system:
            self.coder_calls += 1
            if self.tool_call and self.coder_calls == 1:
                return LLMResponse(
                    text="",
                    function_calls=[
                        FunctionCall(
                            name="write_file",
                            args={
                                "path": "pkg/generated.py",
                                "content": "def health():\n    return {'status': 'ok'}\n",
                            },
                            id="call-1",
                        )
                    ],
                    prompt_tokens=10,
                    output_tokens=5,
                    model=model or "fake",
                )
            return LLMResponse(
                text="Implemented the change and verified it.",
                prompt_tokens=10,
                output_tokens=5,
                model=model or "fake",
            )
        if "REVIEWER" in system:
            return LLMResponse(
                text=json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Looks good.",
                        "issues": [],
                        "suggestions": [],
                    }
                ),
                prompt_tokens=10,
                output_tokens=5,
                model=model or "fake",
            )
        if "DEBUGGER" in system:
            return LLMResponse(
                text="1. Fix the edge case.",
                prompt_tokens=10,
                output_tokens=5,
                model=model or "fake",
            )
        return LLMResponse(text="ok", prompt_tokens=1, output_tokens=1, model="fake")


@pytest.fixture
def no_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    import orion.agents.nodes as nodes

    monkeypatch.setattr(nodes, "detect_verification_commands", lambda root: [])


def test_build_run_end_to_end(config: OrionConfig, no_tests: None) -> None:
    engine = OrionEngine(config, llm=FakeLLM())
    events: list[dict[str, Any]] = []
    result = engine.run("Add a health endpoint", on_event=events.append)

    assert result.error is None
    assert result.mode == "build"
    assert "Run complete" in result.final
    assert result.plan["goal"] == "Add a health endpoint"
    assert result.review["verdict"] == "approve"
    assert result.evaluation["verdict"] == "approve"
    assert result.tests["status"] == "skipped"
    assert result.changes == []
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start"
    assert "summary" in kinds
    assert "plan" in kinds
    assert "review" in kinds
    assert "test" in kinds
    assert "evaluation" in kinds
    assert kinds[-1] == "done"


def test_local_run_start_event_uses_active_model(config: OrionConfig, no_tests: None) -> None:
    local = config.with_overrides(llm_provider="local", local_model="qwen2.5:test")
    engine = OrionEngine(local, llm=FakeLLM())
    events: list[dict[str, Any]] = []
    result = engine.run("Explain the auth flow", mode="plan", on_event=events.append)

    assert result.error is None
    assert events[0]["type"] == "start"
    assert events[0]["model"] == "qwen2.5:test"


def test_plan_mode_never_changes_files(config: OrionConfig, no_tests: None) -> None:
    engine = OrionEngine(config, llm=FakeLLM())
    result = engine.run("Explain the auth flow", mode="plan")
    assert result.error is None
    assert result.mode == "plan"
    assert result.changes == []
    assert result.final
    assert "No files" in result.final or "Plan" in result.final


def test_tool_calls_produce_changes(config: OrionConfig, no_tests: None) -> None:
    engine = OrionEngine(config, llm=FakeLLM(tool_call=True))
    events: list[dict[str, Any]] = []
    result = engine.run("Create a health helper", on_event=events.append)

    assert result.error is None
    assert result.changes
    assert result.changes[0]["path"] == "pkg/generated.py"
    assert result.changes[0]["action"] == "create"
    assert (config.workspace / "pkg" / "generated.py").is_file()
    assert any(event["type"] == "tool_call" for event in events)
    assert any(event["type"] == "tool_result" for event in events)
    assert any(event["type"] == "diff" for event in events)


def test_stream_yields_done_event(config: OrionConfig, no_tests: None) -> None:
    engine = OrionEngine(config, llm=FakeLLM())
    events = list(engine.stream("Add docs"))
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["mode"] == "build"


def test_graph_mermaid_mentions_nodes() -> None:
    from orion.agents.graph import graph_mermaid

    diagram = graph_mermaid()
    for node in (
        "context",
        "summarizer",
        "architect",
        "generator",
        "reviewer",
        "evaluator",
        "debugger",
        "finalize",
    ):
        assert node in diagram
