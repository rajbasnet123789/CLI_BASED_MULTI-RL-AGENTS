"""Chunker and plan/JSON helpers."""

from __future__ import annotations

from orion.agents.prompts import normalize_plan, render_plan
from orion.rag.chunker import Chunk, chunk_text, embedding_text
from orion.utils import extract_json, format_duration, slugify, strip_code_fence, truncate


def test_chunk_text_line_ranges() -> None:
    source = "\n".join(f"line {index}" for index in range(1, 101))
    chunks = chunk_text(source, "pkg/app.py", max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 100
    for chunk in chunks:
        assert chunk.path == "pkg/app.py"
        assert chunk.language == "python"
        assert chunk.start_line <= chunk.end_line
        assert chunk.text
        assert chunk.hash


def test_chunk_overlap_repeats_lines() -> None:
    source = "\n".join(f"line {index}" for index in range(1, 41))
    chunks = chunk_text(source, "file.py", max_chars=250, overlap=80)
    assert len(chunks) >= 2
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunk_handles_empty_and_long_lines() -> None:
    assert chunk_text("", "file.py") == []
    assert chunk_text("   \n\n", "file.py") == []
    long_line = "x" * 1000
    chunks = chunk_text(long_line, "min.js", max_chars=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_chunk_ids_and_metadata() -> None:
    chunk = Chunk(path="a.py", start_line=1, end_line=2, text="x = 1\n", hash="abc")
    assert chunk.id == "a.py:1-2:abc"
    assert chunk.location == "a.py:1-2"
    assert chunk.as_metadata()["path"] == "a.py"
    assert embedding_text(chunk).startswith("File: a.py")


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here you go: {"a": 3} thanks') == {"a": 3}
    assert extract_json("not json") is None
    assert strip_code_fence("```python\nx = 1\n```") == "x = 1"


def test_plan_normalization_and_rendering() -> None:
    plan = normalize_plan(
        {
            "goal": "Add health check",
            "steps": [
                {"title": "Create endpoint", "detail": "In app.py", "files": "app.py"},
                "Add a test",
            ],
            "risks": "Port conflicts",
            "test_plan": "pytest",
        }
    )
    assert plan["goal"] == "Add health check"
    assert plan["steps"][0]["files"] == ["app.py"]
    assert plan["risks"] == ["Port conflicts"]
    markdown = render_plan(plan)
    assert "**Create endpoint**" in markdown
    assert "Verification" in markdown

    assert normalize_plan(None)["steps"] == []
    assert normalize_plan(["one", "two"])["steps"] == ["one", "two"]


def test_misc_utils() -> None:
    assert slugify("Add a /health endpoint!") == "add-a-health-endpoint"
    assert format_duration(0.25) == "250ms"
    assert format_duration(65) == "1m 5s"
    assert truncate("abcdef", 100) == "abcdef"
