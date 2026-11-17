"""Orion AI - a multi-agent coding assistant for the terminal.

Orion combines a LangGraph multi-agent pipeline (planner, coder, reviewer,
debugger, tester) with Gemini models and a ChromaDB-backed code index.

Quick start::

    from orion import Orion, OrionConfig

    orion = Orion(OrionConfig.load())
    result = orion.run("Add a /health endpoint", mode="build")
    print(result.final)

Or stream structured events instead::

    for event in orion.stream("Explain the auth flow", mode="plan"):
        print(event["type"], event.get("text", ""))
"""

from __future__ import annotations

from orion.api import Orion, RunResult
from orion.config import ApprovalPolicy, Mode, OrionConfig
from orion.errors import (
    ApprovalDenied,
    ConfigError,
    LLMError,
    OrionError,
    RAGError,
    ToolError,
)
from orion.version import __version__

__all__ = [
    "ApprovalDenied",
    "ApprovalPolicy",
    "ConfigError",
    "LLMError",
    "Mode",
    "Orion",
    "OrionConfig",
    "OrionError",
    "RAGError",
    "RunResult",
    "ToolError",
    "__version__",
]
