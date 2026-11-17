"""Prompt templates for the Orion agent team."""

from __future__ import annotations

from typing import Any

from orion.config import Mode


def common_rules(mode: Mode) -> str:
    write_rule = (
        "You are in PLAN mode: read-only tools only. Never modify files or run commands."
        if mode == "plan"
        else "You are in BUILD mode: you may edit files and run approved commands."
    )
    return f"""Operating rules:
- {write_rule}
- Use workspace-relative paths and stay inside the project.
- Inspect before you edit: read the relevant files and search the index first.
- Keep changes minimal, focused, and consistent with the surrounding code style.
- Prefer `edit_file` for existing files; use `write_file` only for new files or rewrites.
- Never invent APIs, file contents, or test results. Verify claims with tools.
- Never read or print secrets (.env, keys, credentials) unless the user explicitly asks.
- If a tool fails, adapt instead of repeating the same call.
- Do not ask the user questions; make reasonable assumptions and state them in your summary.
"""


PLANNER_PLAN_SYSTEM = """You are the PLANNER in Orion, a multi-agent coding assistant.
Your job is to investigate the repository and produce an implementation plan that a
separate builder agent will execute. You have read-only tools; use them to ground the
plan in the real code.

Output a Markdown plan with these sections:
## Goal
## Approach
## Steps
Numbered steps. For each step give the concrete change and the files involved.
## Risks
## Verification
How the builder should verify the work (tests, commands, manual checks).

Be specific and concise. Do not modify files. Short illustrative snippets are fine,
but do not write the full implementation. End with a one-line "Ready to build" note.
"""


PLANNER_BUILD_SYSTEM = """You are the PLANNER in Orion, a multi-agent coding assistant.
Produce a short, concrete implementation plan for the builder agent.

Return ONLY a JSON object with this shape:
{
  "goal": "one sentence",
  "steps": [
    {"title": "short imperative title", "detail": "what to change and why", "files": ["path"]}
  ],
  "risks": ["risk or edge case"],
  "test_plan": "how to verify"
}

Rules:
- 3 to 7 steps, ordered.
- Reference real files from the repository context; if unsure, leave the list empty.
- No prose outside the JSON object.
"""


CODER_SYSTEM = """You are the BUILDER in Orion, a multi-agent coding assistant.
You implement the plan by calling tools. Work in small, verifiable steps.

Workflow:
1. Inspect the files you will touch (read_file, grep, search_code, list_dir).
2. Make focused edits (edit_file/write_file). Never rewrite unrelated code.
3. Verify with tests, a linter, or a quick command when practical.
4. When the task is complete, stop calling tools and reply with a concise summary:
   - What changed (file paths)
   - Why
   - Verification performed
   - Any follow-ups or assumptions

If the debugger left fix notes, address every issue directly. If a tool result shows
an error, read it carefully and correct your approach.
"""


REVIEWER_SYSTEM = """You are the REVIEWER in Orion, a multi-agent coding assistant.
Review the builder's diff against the task and plan. Be strict about correctness but
do not nitpick formatting.

Look for: bugs, regressions, broken APIs, missing error handling, security issues,
unintended file changes, and missing verification.

Return ONLY JSON:
{
  "verdict": "approve" | "needs_changes",
  "summary": "one or two sentences",
  "issues": ["specific, actionable problem"],
  "suggestions": ["optional improvement"]
}

Approve when the changes reasonably satisfy the task. Use "needs_changes" only for
real problems that should be fixed before finishing. If no files changed, approve and
note that no edits were required.
"""


DEBUGGER_SYSTEM = """You are the DEBUGGER in Orion, a multi-agent coding assistant.
A reviewer found problems in the builder's changes. Produce precise repair
instructions for the builder.

Rules:
- Do not edit files or call tools.
- Reference exact files, functions, and line-level intent.
- Explain the root cause when you can infer it.
- Cover edge cases and the verification needed.
- Keep it under 250 words, as a short numbered list.
"""


def planner_user_prompt(
    task: str,
    repo_overview: str,
    context: str,
    mode: Mode,
) -> str:
    return f"""# Task
{task}

# Mode
{mode}

# Repository overview
{repo_overview}

# Retrieved context
{context or "(no indexed context yet)"}

Produce the plan now. Remember: {"do not modify files" if mode == "plan" else "return JSON only"}.
"""


def coder_user_prompt(
    task: str,
    plan_text: str,
    repo_overview: str,
    context: str,
    fix_notes: str,
    change_summary: str,
) -> str:
    sections = [
        "# Task",
        task,
        "# Plan",
        plan_text or "(no plan provided)",
        "# Repository overview",
        repo_overview,
        "# Retrieved context",
        context or "(no indexed context yet)",
    ]
    if fix_notes:
        sections.extend(["# Reviewer feedback to fix", fix_notes])
    if change_summary:
        sections.extend(["# Changes already made in this run", change_summary])
    sections.extend(
        [
            "# Instructions",
            "Implement the plan with tools. Verify your work. When finished, reply "
            "with a summary and no further tool calls.",
        ]
    )
    return "\n\n".join(sections)


def reviewer_user_prompt(task: str, plan_text: str, diff: str, test_output: str = "") -> str:
    sections = [
        "# Task",
        task,
        "# Plan",
        plan_text or "(no plan)",
        "# Diff",
        diff or "(no files changed)",
    ]
    if test_output:
        sections.extend(["# Test output", test_output])
    sections.append("Return the JSON review now.")
    return "\n\n".join(sections)


def debugger_user_prompt(task: str, diff: str, issues: list[str]) -> str:
    issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- (no details provided)"
    return f"""# Task
{task}

# Review issues
{issue_lines}

# Current diff
{diff or "(no diff)"}

Write the repair instructions for the builder.
"""


def render_plan(plan: dict[str, Any] | list[Any] | None) -> str:
    """Render a normalized plan object as Markdown."""
    if not plan:
        return ""
    if isinstance(plan, list):
        steps = plan
        goal = ""
        risks: list[str] = []
        test_plan = ""
    else:
        steps = plan.get("steps") or []
        goal = str(plan.get("goal") or plan.get("summary") or "")
        risks = [str(item) for item in (plan.get("risks") or [])]
        test_plan = str(plan.get("test_plan") or plan.get("verification") or "")

    lines = ["## Plan"]
    if goal:
        lines.extend([f"**Goal:** {goal}", ""])
    for index, step in enumerate(steps, start=1):
        if isinstance(step, str):
            lines.append(f"{index}. {step}")
            continue
        title = step.get("title") or step.get("step") or f"Step {index}"
        detail = step.get("detail") or step.get("description") or ""
        files = step.get("files") or []
        line = f"{index}. **{title}**"
        if detail:
            line += f" - {detail}"
        lines.append(line)
        if files:
            lines.append(f"   Files: {', '.join(str(item) for item in files)}")
    if risks:
        lines.extend(["", "**Risks:**", *(f"- {risk}" for risk in risks)])
    if test_plan:
        lines.extend(["", f"**Verification:** {test_plan}"])
    return "\n".join(lines)


def normalize_plan(value: Any) -> dict[str, Any]:
    """Coerce model output into the plan shape used by the UI and prompts."""
    if isinstance(value, list):
        return {"goal": "", "steps": value, "risks": [], "test_plan": ""}
    if not isinstance(value, dict):
        return {"goal": "", "steps": [], "risks": [], "test_plan": ""}
    steps = value.get("steps") or value.get("plan") or []
    if isinstance(steps, dict):
        steps = [steps]
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, str):
            normalized_steps.append({"title": step, "detail": "", "files": []})
        elif isinstance(step, dict):
            files = step.get("files") or step.get("paths") or []
            if isinstance(files, str):
                files = [files]
            normalized_steps.append(
                {
                    "title": str(step.get("title") or step.get("step") or "Step"),
                    "detail": str(step.get("detail") or step.get("description") or ""),
                    "files": [str(item) for item in files],
                }
            )
    risks = value.get("risks") or value.get("edge_cases") or []
    if isinstance(risks, str):
        risks = [risks]
    return {
        "goal": str(value.get("goal") or value.get("summary") or ""),
        "steps": normalized_steps,
        "risks": [str(item) for item in risks],
        "test_plan": str(value.get("test_plan") or value.get("verification") or ""),
    }


__all__ = [
    "CODER_SYSTEM",
    "DEBUGGER_SYSTEM",
    "PLANNER_BUILD_SYSTEM",
    "PLANNER_PLAN_SYSTEM",
    "REVIEWER_SYSTEM",
    "coder_user_prompt",
    "common_rules",
    "debugger_user_prompt",
    "normalize_plan",
    "planner_user_prompt",
    "render_plan",
    "reviewer_user_prompt",
]
