"""Line-aware source chunking for retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from orion.workspace import language_for


@dataclass(frozen=True, slots=True)
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    language: str = "text"
    hash: str = ""

    @property
    def id(self) -> str:
        digest = self.hash or hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]
        return f"{self.path}:{self.start_line}-{self.end_line}:{digest[:12]}"

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "hash": self.hash,
        }


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def embedding_text(chunk: Chunk) -> str:
    """Contextual text sent to the embedding model (not the stored document)."""
    return (
        f"File: {chunk.path}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        f"Language: {chunk.language}\n\n"
        f"{chunk.text}"
    )


def chunk_text(
    text: str,
    path: str,
    max_chars: int = 1200,
    overlap: int = 150,
    language: str | None = None,
) -> list[Chunk]:
    """Split source text into overlapping, line-aligned chunks."""
    if not text or not text.strip():
        return []
    max_chars = max(200, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    lang = language or language_for(path)
    chunks: list[Chunk] = []
    start = 0
    total = len(lines)

    while start < total:
        end = start
        size = 0
        while end < total:
            line_size = len(lines[end]) + 1
            if end > start and size + line_size > max_chars:
                break
            size += line_size
            end += 1
            if size >= max_chars:
                break

        block = "\n".join(lines[start:end])
        if block.strip():
            if len(block) > max_chars and end == start + 1:
                # A single huge line (minified file, long JSON): hard split.
                for offset in range(0, len(block), max_chars):
                    piece = block[offset : offset + max_chars]
                    if piece.strip():
                        chunks.append(
                            Chunk(
                                path=path,
                                start_line=start + 1,
                                end_line=end,
                                text=piece,
                                language=lang,
                                hash=content_hash(piece),
                            )
                        )
            else:
                chunks.append(
                    Chunk(
                        path=path,
                        start_line=start + 1,
                        end_line=end,
                        text=block,
                        language=lang,
                        hash=content_hash(block),
                    )
                )

        if end >= total:
            break

        if overlap:
            back = end
            overlap_size = 0
            while back > start + 1 and overlap_size < overlap:
                back -= 1
                overlap_size += len(lines[back]) + 1
            start = max(back, start + 1)
        else:
            start = end

    return chunks


__all__ = ["Chunk", "chunk_text", "content_hash", "embedding_text"]
