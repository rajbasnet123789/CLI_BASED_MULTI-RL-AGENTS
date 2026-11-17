"""Vector stores: ChromaDB by default, a JSON fallback for constrained setups."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from orion.config import OrionConfig
from orion.errors import RAGError
from orion.rag.chunker import Chunk


@dataclass(slots=True)
class SearchHit:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float = 0.0
    language: str = "text"
    source: str = "vector"

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "score": round(self.score, 4),
            "source": self.source,
            "text": self.text,
        }


@runtime_checkable
class VectorStore(Protocol):
    @property
    def count(self) -> int: ...

    @property
    def embedder_name(self) -> str | None: ...

    def set_embedder(self, name: str) -> None: ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def delete_path(self, path: str) -> None: ...

    def query(
        self,
        vector: Sequence[float],
        k: int,
        path_prefix: str | None = None,
    ) -> list[SearchHit]: ...

    def reset(self) -> None: ...

    def paths(self) -> set[str]: ...


class ChromaVectorStore:
    """Persistent ChromaDB collection with cosine distance."""

    COLLECTION = "orion_code"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise RAGError(
                "chromadb is not installed. Reinstall Orion: pip install 'orion-ai'"
            ) from exc
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception:
            self._client = chromadb.PersistentClient(path=str(self.directory))
        self._collection = self._open_collection()
        self._meta_path = self.directory / "orion_meta.json"
        self._lock = threading.RLock()

    def _open_collection(self):
        """Open the collection without re-applying creation metadata.

        Chroma treats ``hnsw:space`` as immutable and raises when a collection
        is re-opened with a different distance function, so we only pass it on
        first creation.
        """
        try:
            return self._client.get_collection(self.COLLECTION)
        except Exception:
            return self._client.create_collection(
                name=self.COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def collection(self):
        return self._collection

    @property
    def count(self) -> int:
        return int(self._collection.count())

    @property
    def embedder_name(self) -> str | None:
        try:
            payload = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("embedder"):
            return str(payload["embedder"])
        # Fall back to metadata written by older Orion versions.
        metadata = dict(self._collection.metadata or {})
        value = metadata.get("embedder")
        return str(value) if value else None

    def set_embedder(self, name: str) -> None:
        # Store the embedder in a sidecar file: Chroma refuses to modify
        # collection metadata that contains the immutable ``hnsw:space`` key.
        temp = self._meta_path.with_suffix(".tmp")
        try:
            temp.write_text(
                json.dumps({"version": 1, "embedder": name}),
                encoding="utf-8",
            )
            temp.replace(self._meta_path)
        except OSError as exc:  # pragma: no cover - disk/permission failures
            raise RAGError(f"Could not persist the embedder name: {exc}") from exc

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise RAGError("Chunk/vector count mismatch while writing the index")
        if not chunks:
            return
        batch_size = 512
        with self._lock:
            for start in range(0, len(chunks), batch_size):
                batch = list(chunks[start : start + batch_size])
                batch_vectors = [list(vector) for vector in vectors[start : start + batch_size]]
                self._collection.upsert(
                    ids=[chunk.id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.as_metadata() for chunk in batch],
                    embeddings=batch_vectors,
                )

    def delete_path(self, path: str) -> None:
        with self._lock:
            try:
                self._collection.delete(where={"path": path})
            except Exception:
                # Older Chroma builds dislike deleting from an empty collection.
                pass

    def query(
        self,
        vector: Sequence[float],
        k: int,
        path_prefix: str | None = None,
    ) -> list[SearchHit]:
        if self.count == 0:
            return []
        fetch = min(max(k * 4, k), max(1, self.count))
        with self._lock:
            result = self._collection.query(
                query_embeddings=[list(vector)],
                n_results=fetch,
                include=["documents", "metadatas", "distances"],
            )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[SearchHit] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            meta = metadata or {}
            path = str(meta.get("path", ""))
            if path_prefix and not path.startswith(path_prefix.rstrip("/")):
                continue
            try:
                score = 1.0 - float(distance)
            except (TypeError, ValueError):
                score = 0.0
            hits.append(
                SearchHit(
                    path=path,
                    start_line=int(meta.get("start_line", 1) or 1),
                    end_line=int(meta.get("end_line", 1) or 1),
                    text=document or "",
                    score=max(0.0, min(1.0, score)),
                    language=str(meta.get("language", "text")),
                    source="vector",
                )
            )
            if len(hits) >= k:
                break
        return hits

    def reset(self) -> None:
        with self._lock:
            try:
                self._client.delete_collection(self.COLLECTION)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self.COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            try:
                self._meta_path.unlink()
            except OSError:
                pass

    def paths(self) -> set[str]:
        try:
            result = self._collection.get(include=["metadatas"], limit=50_000)
        except Exception:
            return set()
        return {
            str(metadata.get("path"))
            for metadata in (result.get("metadatas") or [])
            if metadata and metadata.get("path")
        }


class JsonVectorStore:
    """Simple JSON-backed vector store used when Chroma cannot start."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / "vectors.json"
        self._lock = threading.RLock()
        self._records: dict[str, dict] = {}
        self._embedder: str | None = None
        self._load()

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def embedder_name(self) -> str | None:
        return self._embedder

    def _load(self) -> None:
        if not self.file.is_file():
            return
        try:
            payload = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self._embedder = payload.get("embedder")
        records = payload.get("records")
        if isinstance(records, dict):
            self._records = records

    def _save(self) -> None:
        payload = {"version": 1, "embedder": self._embedder, "records": self._records}
        temp = self.file.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.file)

    def set_embedder(self, name: str) -> None:
        with self._lock:
            self._embedder = name
            self._save()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise RAGError("Chunk/vector count mismatch while writing the index")
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=False):
                record = chunk.as_metadata()
                record["text"] = chunk.text
                record["vector"] = [float(value) for value in vector]
                self._records[chunk.id] = record
            self._save()

    def delete_path(self, path: str) -> None:
        with self._lock:
            stale = [key for key, record in self._records.items() if record.get("path") == path]
            for key in stale:
                self._records.pop(key, None)
            if stale:
                self._save()

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        dot = sum(left[index] * right[index] for index in range(size))
        left_norm = math.sqrt(sum(value * value for value in left[:size]))
        right_norm = math.sqrt(sum(value * value for value in right[:size]))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def query(
        self,
        vector: Sequence[float],
        k: int,
        path_prefix: str | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            records = list(self._records.values())
        scored: list[SearchHit] = []
        for record in records:
            path = str(record.get("path", ""))
            if path_prefix and not path.startswith(path_prefix.rstrip("/")):
                continue
            score = self._cosine(vector, record.get("vector") or [])
            scored.append(
                SearchHit(
                    path=path,
                    start_line=int(record.get("start_line", 1) or 1),
                    end_line=int(record.get("end_line", 1) or 1),
                    text=str(record.get("text", "")),
                    score=max(0.0, min(1.0, score)),
                    language=str(record.get("language", "text")),
                    source="vector",
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def reset(self) -> None:
        with self._lock:
            self._records = {}
            self._embedder = None
            self._save()

    def paths(self) -> set[str]:
        return {str(record.get("path")) for record in self._records.values() if record.get("path")}


def open_store(
    config: OrionConfig,
    on_warning: object | None = None,
) -> VectorStore:
    """Open ChromaDB, falling back to the JSON store if it cannot start."""
    try:
        return ChromaVectorStore(config.chroma_path)
    except Exception as exc:
        warning = f"ChromaDB unavailable ({exc}); using the JSON vector store."
        if callable(on_warning):
            on_warning(warning)
        return JsonVectorStore(config.data_path / "vectors")


__all__ = [
    "ChromaVectorStore",
    "JsonVectorStore",
    "SearchHit",
    "VectorStore",
    "open_store",
]
