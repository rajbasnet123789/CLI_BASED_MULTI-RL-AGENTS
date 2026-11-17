"""LangGraph nodes implementing Orion's agent team."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from orion.agents.prompts import (
    CODER_SYSTEM,
    DEBUGGER_SYSTEM,
    EVALUATOR_SYSTEM,
    PLANNER_BUILD_SYSTEM,
    PLANNER_PLAN_SYSTEM,
    REVIEWER_SYSTEM,
    SUMMARIZER_SYSTEM,
    coder_user_prompt,
    common_rules,
    debugger_user_prompt,
    evaluator_user_prompt,
    normalize_plan,
    planner_user_prompt,
    render_plan,
    reviewer_user_prompt,
    summarizer_user_prompt,
)
from orion.agents.state import AgentState
from orion.approval import ApprovalCallback
from orion.config import Mode, OrionConfig
from orion.errors import CancelledError, LLMError
from orion.llm import LLM, LLMResponse, Usage
from orion.rag.retriever import CodeRetriever
from orion.tools.base import FileChange, ToolContext
from orion.tools.registry import ToolRegistry, changes_diff
from orion.utils import extract_json, truncate
from orion.workspace import (
    VerificationCommand,
    detect_verification_commands,
    project_overview,
)

EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class AgentRuntime:
    """Mutable per-run state shared by the nodes."""

    config: OrionConfig
    llm: LLM
    registry: ToolRegistry
    retriever: CodeRetriever | None = None
    emit: EventCallback = lambda event: None
    approval: ApprovalCallback | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    dry_run: bool = False
    changes: list[FileChange] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    _overview: str | None = None

    # ------------------------------------------------------------------
    def event(self, event_type: str, **payload: Any) -> None:
        payload = {"type": event_type, **payload}
        try:
            self.emit(payload)
        except Exception:  # pragma: no cover - UI callbacks must never break a run
            pass

    def status(self, agent: str, message: str) -> None:
        self.event("status", agent=agent, message=message)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise CancelledError("Run cancelled")

    def repo_overview(self) -> str:
        if self._overview is None:
            self._overview = project_overview(
                self.config.workspace,
                ignore=self.config.ignore,
                max_files=140,
                max_chars=9000,
            )
        return self._overview

    def make_context(self, mode: Mode) -> ToolContext:
        return ToolContext(
            config=self.config,
            mode=mode,
            approval=self.approval,
            dry_run=self.dry_run,
            emit=self.emit,
            changes=self.changes,
            retriever=self.retriever,
        )

    # ------------------------------------------------------------------
    def ask(
        self,
        agent: str,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.check_cancel()
        self.status(agent, f"{agent} is thinking")
        response = self.llm.generate(
            messages,
            system=system,
            tools=tools,
            model=model or self.config.model_for(agent),
            response_mime_type="application/json" if json_mode else None,
            temperature=temperature,
        )
        self.usage.add(response.prompt_tokens, response.output_tokens)
        self.event("usage", usage=self.usage.as_dict(), agent=agent)
        if (response.text or "").strip():
            self.event(
                "model_output",
                agent=agent,
                model=response.model or model or self.config.model_for(agent),
                text=truncate(response.text, 12_000, "model output"),
                call=self.usage.calls,
                prompt_tokens=response.prompt_tokens,
                output_tokens=response.output_tokens,
            )
        return response

    def stream(
        self,
        agent: str,
        messages: list[dict[str, Any]],
        *,
        system: str,
        model: str | None = None,
    ) -> str:
        self.check_cancel()
        self.status(agent, f"{agent} is writing")
        before = (
            self.llm.usage.prompt_tokens,
            self.llm.usage.output_tokens,
            self.llm.usage.calls,
        )
        chunks: list[str] = []
        for chunk in self.llm.stream(
            messages,
            system=system,
            model=model or self.config.model_for(agent),
        ):
            self.check_cancel()
            chunks.append(chunk)
            self.event("token", agent=agent, text=chunk)
        after = (
            self.llm.usage.prompt_tokens,
            self.llm.usage.output_tokens,
            self.llm.usage.calls,
        )
        self.usage.add(
            after[0] - before[0],
            after[1] - before[1],
            calls=max(0, after[2] - before[2]),
        )
        self.event("usage", usage=self.usage.as_dict(), agent=agent)
        text = "".join(chunks).strip()
        if text:
            self.event(
                "model_output",
                agent=agent,
                model=model or self.config.model_for(agent),
                text=truncate(text, 12_000, "model output"),
                call=self.usage.calls,
            )
        return text

    def run_tool_loop(
        self,
        *,
        agent: str,
        system: str,
        user_prompt: str,
        mode: Mode,
        max_rounds: int,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> tuple[str, ToolContext]:
        """ReAct-style loop: model -> tools -> model until a final answer."""
        context = self.make_context(mode)
        schemas = tools if tools is not None else self.registry.schemas(mode)
        messages: list[dict[str, Any]] = [{"role": "user", "text": user_prompt}]
        collected: list[str] = []

        for round_index in range(max_rounds):
            self.check_cancel()
            response = self.ask(
                agent,
                messages,
                system=system,
                tools=schemas,
                model=model,
                temperature=temperature,
            )
            if response.text:
                collected.append(response.text)
                self.event("message", agent=agent, text=response.text)
            if not response.function_calls:
                break

            calls: list[dict[str, Any]] = []
            for index, call in enumerate(response.function_calls):
                calls.append(
                    {
                        "name": call.name,
                        "args": call.args or {},
                        "id": call.id or f"{agent}-{round_index}-{index}",
                        "thought_signature": call.thought_signature,
                    }
                )
            messages.append(
                {"role": "model", "text": response.text, "function_calls": calls}
            )
            tool_responses: list[dict[str, Any]] = []
            for call in calls:
                self.check_cancel()
                self.event(
                    "tool_call",
                    agent=agent,
                    call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                )
                result = self.registry.execute(call["name"], call["args"], context)
                self.event(
                    "tool_result",
                    agent=agent,
                    call_id=call["id"],
                    name=call["name"],
                    ok=result.ok,
                    content=truncate(result.content, 12_000, "tool result"),
                    display=result.display or call["name"],
                    changed_paths=result.changed_paths,
                )
                if result.diff:
                    path = (
                        result.changed_paths[0]
                        if result.changed_paths
                        else str(call["args"].get("path", ""))
                    )
                    self.event("diff", agent=agent, path=path, diff=result.diff)
                tool_responses.append(
                    {
                        "name": call["name"],
                        "id": call["id"],
                        "response": {
                            "ok": result.ok,
                            "result": truncate(result.content, 12_000, "tool result"),
                        },
                    }
                )
            messages.append({"role": "tool", "function_responses": tool_responses})
            if round_index == max_rounds - 1:
                self.event(
                    "warning",
                    agent=agent,
                    message=f"Tool budget reached ({max_rounds} rounds); wrapping up.",
                )
        return "\n\n".join(collected).strip(), context


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
def context_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    runtime.status("context", "Gathering repository context")
    overview = runtime.repo_overview()
    hits = []
    if runtime.retriever is not None:
        try:
            hits = runtime.retriever.search(state.get("task", ""), k=runtime.config.top_k)
        except Exception as exc:
            runtime.event("warning", agent="context", message=f"Retrieval failed: {exc}")
    context_text = ""
    if hits and runtime.retriever is not None:
        context_text = runtime.retriever.build_context(hits)
    runtime.event(
        "context",
        hits=[hit.as_dict() for hit in hits],
        count=len(hits),
        context_chars=len(context_text),
    )
    return {
        "repo_overview": overview,
        "context_hits": [hit.as_dict() for hit in hits],
        "context_text": context_text,
    }


def summarizer_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    """Compress repository context into a briefing for the architect/builder."""
    runtime.check_cancel()
    overview = state.get("repo_overview", "") or ""
    context = state.get("context_text", "") or ""
    task = state.get("task", "")
    if not overview and not context:
        return {"context_summary": ""}

    runtime.status("summarizer", "Summarizing repository context")
    response = runtime.ask(
        "summarizer",
        [{"role": "user", "text": summarizer_user_prompt(task, overview, context)}],
        system=SUMMARIZER_SYSTEM,
        model=runtime.config.model_for("summarizer"),
    )
    summary = truncate((response.text or "").strip(), 4000, "context summary")
    if not summary:
        summary = truncate(f"{overview}\n\n{context}".strip(), 4000, "context summary")
    runtime.event("summary", agent="summarizer", text=summary)
    return {"context_summary": summary}


def planner_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    mode: Mode = state.get("mode", runtime.config.mode)  # type: ignore[assignment]
    prompt = planner_user_prompt(
        state.get("task", ""),
        state.get("repo_overview", ""),
        state.get("context_summary") or state.get("context_text", ""),
        mode,
    )
    system = f"{PLANNER_PLAN_SYSTEM}\n{common_rules(mode)}"

    if mode == "plan":
        text, _ = runtime.run_tool_loop(
            agent="planner",
            system=system,
            user_prompt=prompt,
            mode="plan",
            max_rounds=min(6, runtime.config.max_tool_rounds),
            model=runtime.config.model_for("planner"),
        )
        if not text:
            text = "## Plan\nNo plan was produced. Try rephrasing the request."
        runtime.event("plan", markdown=text, plan={"markdown": text})
        return {"plan": {"markdown": text}, "plan_text": text}

    response = runtime.ask(
        "planner",
        [{"role": "user", "text": prompt}],
        system=f"{PLANNER_BUILD_SYSTEM}\n{common_rules(mode)}",
        json_mode=True,
        model=runtime.config.model_for("planner"),
    )
    parsed = extract_json(response.text)
    if parsed is None:
        plan = normalize_plan({"goal": state.get("task", ""), "steps": [response.text]})
    else:
        plan = normalize_plan(parsed)
    if not plan.get("steps"):
        plan["steps"] = [
            {
                "title": "Inspect and implement the task",
                "detail": state.get("task", ""),
                "files": [],
            }
        ]
    markdown = render_plan(plan)
    runtime.event("plan", plan=plan, markdown=markdown)
    return {"plan": plan, "plan_text": markdown}


def coder_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    before = len(runtime.changes)
    prompt = coder_user_prompt(
        state.get("task", ""),
        state.get("plan_text", ""),
        state.get("repo_overview", ""),
        state.get("context_summary") or state.get("context_text", ""),
        state.get("fix_notes", ""),
        _change_summary(runtime.changes),
    )
    text, _ = runtime.run_tool_loop(
        agent="coder",
        system=f"{CODER_SYSTEM}\n{common_rules('build')}",
        user_prompt=prompt,
        mode="build",
        max_rounds=runtime.config.max_tool_rounds,
        model=runtime.config.model_for("coder"),
    )
    if not text:
        if runtime.changes:
            paths = ", ".join(sorted({change.path for change in runtime.changes}))
            text = f"Applied {len(runtime.changes)} change(s) to: {paths}"
        else:
            text = "No changes were necessary for this request."
        runtime.event("message", agent="coder", text=text)
    runtime.event(
        "changes",
        changes=[change.as_dict() for change in runtime.changes],
        new_changes=len(runtime.changes) - before,
    )
    return {
        "coder_summary": text,
        "changes": [change.as_dict() for change in runtime.changes],
    }


def reviewer_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    if not runtime.changes:
        review = {
            "verdict": "approve",
            "summary": "No files were changed; there is nothing to review.",
            "issues": [],
            "suggestions": [],
        }
        runtime.event("review", **review)
        return {"review": review, "issues": []}

    diff = changes_diff(runtime.changes)
    prompt = reviewer_user_prompt(
        state.get("task", ""),
        state.get("plan_text", ""),
        diff,
        "",
    )
    try:
        response = runtime.ask(
            "reviewer",
            [{"role": "user", "text": prompt}],
            system=f"{REVIEWER_SYSTEM}\n{common_rules('build')}",
            json_mode=True,
            model=runtime.config.model_for("reviewer"),
        )
    except LLMError as exc:
        review = {
            "verdict": "approve",
            "summary": f"Reviewer unavailable ({exc}); changes were not independently reviewed.",
            "issues": [],
            "suggestions": [],
        }
        runtime.event("review", **review)
        return {"review": review, "issues": []}

    parsed = extract_json(response.text) or {}
    verdict = str(parsed.get("verdict", "")).lower().strip()
    if verdict not in {"approve", "needs_changes"}:
        verdict = "needs_changes" if parsed.get("issues") else "approve"
    issues = parsed.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(issue) for issue in issues]
    suggestions = parsed.get("suggestions") or []
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if verdict == "needs_changes" and not issues:
        issues = ["The reviewer requested changes without listing specific issues."]
    review = {
        "verdict": verdict,
        "summary": str(parsed.get("summary") or response.text or ""),
        "issues": issues,
        "suggestions": [str(item) for item in suggestions],
    }
    runtime.event("review", **review)
    return {"review": review, "issues": issues}


def _run_check(runtime: AgentRuntime, check: VerificationCommand) -> dict[str, Any]:
    """Run one deterministic check through the approved tool layer."""
    context = runtime.make_context("build")
    tool_name = "run_tests" if check.name == "tests" else "run_command"
    result = runtime.registry.execute(tool_name, {"command": check.command}, context)
    content = (result.content or "").lower()
    denied = not result.ok and ("denied" in content or "not available in plan mode" in content)
    status = "skipped" if denied else ("passed" if result.ok else "failed")
    data = result.data or {}
    return {
        "name": check.name,
        "framework": check.framework,
        "command": check.command,
        "status": status,
        "exit_code": data.get("exit_code"),
        "detail": truncate(result.content, 3000, f"{check.name} output"),
    }


def evaluator_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    """Combine real test/lint/type signals with an independent model verdict.

    Deterministic failures are blocking: the evaluator never approves a change
    set that failed tests, lint, or type checking, regardless of model opinion.
    """
    runtime.check_cancel()

    signals: list[dict[str, Any]] = []
    if runtime.changes:
        runtime.status("evaluator", "Running deterministic verification")
        for check in detect_verification_commands(runtime.config.workspace):
            runtime.check_cancel()
            signal = _run_check(runtime, check)
            signals.append(signal)
            runtime.event(
                "verification",
                agent="evaluator",
                name=signal["name"],
                status=signal["status"],
                command=signal["command"],
            )

    test_signal = next((signal for signal in signals if signal["name"] == "tests"), None)
    test_results: dict[str, Any] = (
        {
            "status": test_signal["status"],
            "command": test_signal["command"],
            "framework": test_signal["framework"],
            "exit_code": test_signal["exit_code"],
            "output": test_signal["detail"],
            "reason": "Approval denied." if test_signal["status"] == "skipped" else None,
        }
        if test_signal
        else {
            "status": "skipped",
            "reason": "No changes to verify." if not runtime.changes else "No test command detected.",
            "command": None,
            "framework": None,
            "exit_code": None,
        }
    )
    runtime.event("test", **test_results)

    failures = [
        f"{signal['name']} failed ({signal['command']}): {signal['detail'][:400]}"
        for signal in signals
        if signal["status"] == "failed"
    ]

    if not runtime.changes and not failures:
        evaluation = {
            "verdict": "approve",
            "summary": "No files were changed; there is nothing to evaluate.",
            "issues": [],
            "signals": signals,
            "failures": [],
        }
        runtime.event("evaluation", agent="evaluator", **evaluation)
        return {
            "evaluation": evaluation,
            "test_results": test_results,
            "issues": [],
        }

    diff = changes_diff(runtime.changes)
    prompt = evaluator_user_prompt(
        state.get("task", ""),
        state.get("plan_text", ""),
        diff,
        state.get("review") or {},
        signals,
    )
    try:
        response = runtime.ask(
            "evaluator",
            [{"role": "user", "text": prompt}],
            system=f"{EVALUATOR_SYSTEM}\n{common_rules('build')}",
            json_mode=True,
            model=runtime.config.model_for("evaluator"),
        )
        parsed = extract_json(response.text) or {}
    except LLMError:
        parsed = {}

    verdict = str(parsed.get("verdict", "")).lower().strip()
    if verdict not in {"approve", "needs_changes"}:
        verdict = "approve" if not failures else "needs_changes"
    issues = parsed.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(issue) for issue in issues]
    evidence = parsed.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    if failures:
        verdict = "needs_changes"
        issues = failures + issues
        evidence = [f"deterministic signal: {item}" for item in failures] + [
            str(item) for item in evidence
        ]

    evaluation = {
        "verdict": verdict,
        "summary": str(parsed.get("summary") or ("Real checks failed." if failures else "Change set evaluated.")),
        "issues": issues,
        "evidence": [str(item) for item in evidence],
        "signals": signals,
        "failures": failures,
    }
    runtime.event("evaluation", agent="evaluator", **evaluation)
    return {
        "evaluation": evaluation,
        "test_results": test_results,
        "issues": issues,
    }


def tester_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    """Backwards-compatible alias for the evaluator node."""
    return evaluator_node(runtime, state)


def debugger_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    issues = state.get("issues") or []
    if not issues:
        return {"iterations": int(state.get("iterations", 0) or 0) + 1}
    runtime.status("debugger", "Analyzing review findings")
    diff = changes_diff(runtime.changes, max_chars=20_000)
    prompt = debugger_user_prompt(state.get("task", ""), diff, issues)
    response = runtime.ask(
        "debugger",
        [{"role": "user", "text": prompt}],
        system=f"{DEBUGGER_SYSTEM}\n{common_rules('build')}",
        model=runtime.config.model_for("debugger"),
    )
    notes = response.text or "\n".join(f"- {issue}" for issue in issues)
    runtime.event("message", agent="debugger", text=notes)
    return {
        "fix_notes": notes,
        "iterations": int(state.get("iterations", 0) or 0) + 1,
    }


def finalizer_node(runtime: AgentRuntime, state: AgentState) -> dict[str, Any]:
    runtime.check_cancel()
    mode = state.get("mode", "build")
    if mode == "plan":
        final = state.get("plan_text") or "No plan was produced."
        runtime.event(
            "message",
            agent="orion",
            text=(
                "Plan complete. No files were changed.\n\n"
                "Press Tab to switch to build mode and execute this plan."
            ),
        )
        return {"final": final}

    changes = runtime.changes
    lines = ["## Run complete", f"**Task:** {state.get('task', '')}", ""]
    if state.get("plan_text"):
        lines.extend([state["plan_text"], ""])
    if changes:
        lines.append("### Files changed")
        for change in changes:
            if change.action == "move":
                lines.append(f"- `{change.path}` -> `{change.destination}` (move)")
            else:
                added = sum(
                    1 for line in change.diff.splitlines() if line.startswith("+") and not line.startswith("+++")
                )
                removed = sum(
                    1 for line in change.diff.splitlines() if line.startswith("-") and not line.startswith("---")
                )
                lines.append(
                    f"- `{change.path}` ({change.action}, +{added} / -{removed})"
                )
    else:
        lines.append("_No files were changed._")

    review = state.get("review") or {}
    if review:
        lines.extend(
            [
                "",
                "### Review",
                f"**{str(review.get('verdict', 'unknown')).replace('_', ' ').title()}** - "
                f"{review.get('summary', '')}",
            ]
        )
        for issue in review.get("issues") or []:
            lines.append(f"- {issue}")

    evaluation = state.get("evaluation") or {}
    if evaluation:
        lines.extend(
            [
                "",
                "### Evaluation",
                f"- Verdict: **{str(evaluation.get('verdict', 'unknown')).replace('_', ' ').title()}**",
                f"- {evaluation.get('summary', '')}",
            ]
        )
        for signal in evaluation.get("signals") or []:
            lines.append(
                f"- {signal.get('name')}: {signal.get('status')} "
                f"(`{signal.get('command')}`)"
            )
        for issue in evaluation.get("issues") or []:
            lines.append(f"- {issue}")

    tests = state.get("test_results") or {}
    if tests:
        lines.extend(
            [
                "",
                "### Tests",
                f"- Status: **{tests.get('status', 'unknown')}**",
            ]
        )
        if tests.get("command"):
            lines.append(f"- Command: `{tests['command']}`")
        if tests.get("reason"):
            lines.append(f"- Note: {tests['reason']}")
        if tests.get("exit_code") is not None:
            lines.append(f"- Exit code: {tests['exit_code']}")

    if runtime.usage.calls:
        lines.extend(
            [
                "",
                "### Usage",
                f"- {runtime.usage.calls} model call(s), "
                f"{runtime.usage.prompt_tokens + runtime.usage.output_tokens:,} tokens",
            ]
        )
    if runtime.dry_run:
        lines.extend(["", "> Dry run: changes were computed but not written to disk."])
    if runtime.config.mode == "build" and runtime.config.dry_run:
        lines.extend(["", "> Dry run is enabled in configuration."])

    final = "\n".join(lines).strip()
    runtime.event("message", agent="orion", text=final)
    return {"final": final}


def _change_summary(changes: list[FileChange]) -> str:
    if not changes:
        return ""
    return "\n".join(f"- {change.action}: {change.path}" for change in changes)


__all__ = [
    "AgentRuntime",
    "coder_node",
    "context_node",
    "debugger_node",
    "evaluator_node",
    "finalizer_node",
    "planner_node",
    "reviewer_node",
    "summarizer_node",
    "tester_node",
]
