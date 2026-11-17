"""Embedding backends for the code index.

Gemini embeddings are the default. A deterministic local hashing embedder keeps
Orion usable offline (and makes tests fast) without pretending to be a neural
model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from orion.config import OrionConfig
from orion.errors import RAGError
from orion.llm import EMBEDDING_DIMENSIONS, GeminiLLM

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "where",
    "what", "which", "how", "why", "are", "was", "were", "you", "your", "can",
    "could", "should", "would", "does", "did", "not", "but", "all", "any", "our",
    "its", "it's", "has", "have", "had", "will", "just", "then", "than", "them",
    "they", "there", "here", "about", "over", "under", "also", "use", "using",
}


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Embeddings from ``gemini-embedding-001`` (768 dimensions by default)."""

    def __init__(
        self,
        llm: GeminiLLM,
        model: str | None = None,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.llm = llm
        self.model = model or llm.embedding_model
        self.dimensions = dimensions
        self.name = f"gemini:{self.model}:{dimensions}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self.llm.embed(
                list(texts),
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            )
        except Exception as exc:
            raise RAGError(
                "Gemini embeddings failed. Run `orion index --embeddings local` "
                f"to use the offline embedder instead.\n\nDetails: {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vectors = self.llm.embed(
                [text],
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            )
        except Exception as exc:
            raise RAGError(f"Gemini query embedding failed: {exc}") from exc
        if not vectors:
            raise RAGError("Gemini returned no query embedding")
        return vectors[0]


class LocalHashEmbedder:
    """Deterministic hashed bag-of-words / trigram embedder.

    Good enough for smoke tests, offline demos, and small repositories. It is
    *not* a replacement for a trained embedding model, so Orion defaults to
    Gemini embeddings whenever a key is available.
    """

    dimensions = 512
    name = "local-hash-v1"

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = [token for token in _TOKEN_RE.findall(lowered) if token not in _STOPWORDS]
        expanded = list(tokens)
        for token in tokens:
            if len(token) > 3:
                expanded.extend(token[index : index + 3] for index in range(len(token) - 2))
            if "_" in token:
                expanded.extend(part for part in token.split("_") if part)
        return expanded

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        if not tokens:
            return vector
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(config: OrionConfig, llm: GeminiLLM | None = None) -> Embedder:
    """Pick the configured embedding backend."""
    if config.embeddings == "local":
        return LocalHashEmbedder()
    if config.embeddings == "gemini" or config.has_api_key:
        return GeminiEmbedder(llm or GeminiLLM(config))
    return LocalHashEmbedder()


def embedder_from_name(
    config: OrionConfig,
    name: str,
    llm: GeminiLLM | None = None,
) -> Embedder | None:
    """Recreate the embedder that produced an existing index.

    Used to keep queries compatible with an index built by an earlier run (for
    example a local-hash index that predates the API key being configured).
    Returns ``None`` when the name is not recognised.
    """
    if not name:
        return None
    if name == LocalHashEmbedder.name or name.startswith("local-hash"):
        return LocalHashEmbedder()
    if name.startswith("gemini:"):
        parts = name.split(":")
        model = parts[1] if len(parts) > 1 and parts[1] else None
        dimensions = EMBEDDING_DIMENSIONS
        if len(parts) > 2:
            try:
                dimensions = int(parts[2])
            except ValueError:
                pass
        return GeminiEmbedder(llm or GeminiLLM(config), model=model, dimensions=dimensions)
    return None


__all__ = [
    "Embedder",
    "GeminiEmbedder",
    "LocalHashEmbedder",
    "build_embedder",
    "embedder_from_name",
]
