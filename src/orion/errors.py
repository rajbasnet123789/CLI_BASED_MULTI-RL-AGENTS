"""Exception hierarchy used across Orion."""

from __future__ import annotations


class OrionError(Exception):
    """Base class for all Orion errors."""


class ConfigError(OrionError):
    """Invalid or missing configuration."""


class LLMError(OrionError):
    """The language model call failed."""


class RAGError(OrionError):
    """The retrieval pipeline failed."""


class ToolError(OrionError):
    """A tool could not complete its work."""


class ApprovalDenied(ToolError):
    """The user (or policy) denied a tool invocation."""


class CancelledError(OrionError):
    """The current run was cancelled by the user."""


__all__ = [
    "ApprovalDenied",
    "CancelledError",
    "ConfigError",
    "LLMError",
    "OrionError",
    "RAGError",
    "ToolError",
]
