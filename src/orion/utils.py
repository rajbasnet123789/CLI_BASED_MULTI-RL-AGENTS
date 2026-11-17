"""Small helpers shared across Orion."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC was added in Python 3.11; support 3.10 too.
from typing import Any

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+.-]*\s*\n(.*?)\n```\s*$", re.DOTALL)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json(text: str) -> Any | None:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    candidates = [text.strip(), strip_code_fence(text)]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            pass
        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except ValueError:
                continue
    return None


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_length].rstrip("-") or "session")


def truncate(text: str, limit: int = 20_000, label: str = "output") -> str:
    """Trim long text while keeping both ends useful."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head - 80
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n... [{omitted:,} characters of {label} omitted] ...\n\n"
        + text[-tail:]
    )


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder}s"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


__all__ = [
    "clamp",
    "dedupe",
    "extract_json",
    "format_duration",
    "now_iso",
    "slugify",
    "strip_code_fence",
    "truncate",
]
