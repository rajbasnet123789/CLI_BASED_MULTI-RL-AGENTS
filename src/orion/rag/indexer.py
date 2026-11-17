"""Incremental repository indexer."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orion.config import OrionConfig
from orion.errors import OrionError
from orion.rag.chunker import Chunk, chunk_text, embedding_text
from orion.rag.embeddings import Embedder, build_embedder
from orion.rag.store import VectorStore, open_store
from orion.utils import now_iso
from orion.workspace import iter_source_files, relative_posix, safe_path

IndexEvent = Callable[[dict], None]


@dataclass
class IndexStats:
    files_total: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_skipped: int = 0
    chunks: int = 0
    vectors: int = 0
    duration: float = 0.0
    embedder: str = ""
    store: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = round(self.duration, 3)
        return data


class CodeIndexer:
    """Index source files into the configured vector store."""

    MANIFEST_VERSION = 1

    def __init__(
        self,
        config: OrionConfig,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        on_event: IndexEvent | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or build_embedder(config)
        self.store = store or open_store(config)
        self.on_event = on_event

    # ------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:  # pragma: no cover - UI callbacks must never break indexing
            pass

    def _load_manifest(self) -> dict:
        path = self.config.manifest_path
        if not path.is_file():
            return {"version": self.MANIFEST_VERSION, "files": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": self.MANIFEST_VERSION, "files": {}}
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            return {"version": self.MANIFEST_VERSION, "files": {}}
        return data

    def _save_manifest(self, files: dict[str, dict]) -> None:
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.MANIFEST_VERSION,
            "embedder": self.embedder.name,
            "updated_at": now_iso(),
            "files": files,
        }
        temp = self.config.manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temp.replace(self.config.manifest_path)

    def _targets(self, paths: Sequence[str | Path] | None) -> list[Path]:
        if not paths:
            return list(
                iter_source_files(
                    self.config.workspace,
                    ignore=self.config.ignore,
                    max_bytes=self.config.max_file_bytes,
                )
            )
        targets: list[Path] = []
        seen: set[Path] = set()
        for raw in paths:
            resolved = safe_path(self.config.workspace, raw)
            candidates: Iterable[Path]
            if resolved.is_dir():
                candidates = iter_source_files(
                    resolved,
                    ignore=self.config.ignore,
                    max_bytes=self.config.max_file_bytes,
                )
            else:
                candidates = [resolved]
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    targets.append(candidate)
        return targets

    # ------------------------------------------------------------------
    def index(
        self,
        force: bool = False,
        paths: Sequence[str | Path] | None = None,
    ) -> IndexStats:
        """Index (or re-index) the workspace incrementally."""
        started = time.monotonic()
        stats = IndexStats(embedder=self.embedder.name, store=type(self.store).__name__)
        self.config.ensure_dirs()
        self._emit(
            {
                "type": "index_status",
                "phase": "scan",
                "message": "Scanning workspace",
            }
        )

        manifest = {"version": self.MANIFEST_VERSION, "files": {}}
        if not force:
            manifest = self._load_manifest()
        manifest_files: dict[str, dict] = dict(manifest.get("files") or {})
        reset_needed = force or (
            self.store.embedder_name is not None
            and self.store.embedder_name != self.embedder.name
        )

        files = self._targets(paths)
        stats.files_total = len(files)
        current: dict[str, dict] = {}
        for path in files:
            rel = relative_posix(path, self.config.workspace)
            try:
                raw = path.read_bytes()
                if len(raw) > self.config.max_file_bytes:
                    stats.files_skipped += 1
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                text = raw.decode("utf-8", errors="replace")
            except OSError as exc:
                stats.files_skipped += 1
                stats.errors.append(f"{rel}: {exc}")
                continue
            current[rel] = {
                "path": path,
                "hash": digest,
                "size": len(raw),
                "text": text,
                "mtime": int(path.stat().st_mtime),
            }

        removed = sorted(set(manifest_files) - set(current))
        changed = [
            rel
            for rel, info in current.items()
            if force or reset_needed or manifest_files.get(rel, {}).get("hash") != info["hash"]
        ]
        stats.files_unchanged = len(current) - len(changed)
        stats.files_removed = len(removed)
        stats.files_indexed = len(changed)

        if not changed and not removed and not reset_needed:
            stats.duration = time.monotonic() - started
            self._emit(
                {
                    "type": "index_status",
                    "phase": "done",
                    "message": "Index is up to date",
                    "stats": stats.as_dict(),
                }
            )
            return stats

        self._emit(
            {
                "type": "index_status",
                "phase": "chunk",
                "message": f"Chunking {len(changed)} changed file(s)",
                "current": 0,
                "total": len(changed),
            }
        )
        chunks: list[Chunk] = []
        for index, rel in enumerate(changed, start=1):
            info = current[rel]
            try:
                file_chunks = chunk_text(
                    info["text"],
                    path=rel,
                    max_chars=self.config.chunk_chars,
                    overlap=self.config.chunk_overlap,
                )
            except Exception as exc:
                stats.errors.append(f"{rel}: {exc}")
                continue
            chunks.extend(file_chunks)
            if index % 10 == 0 or index == len(changed):
                self._emit(
                    {
                        "type": "index_status",
                        "phase": "chunk",
                        "message": f"Chunked {index}/{len(changed)} file(s)",
                        "current": index,
                        "total": len(changed),
                    }
                )
        stats.chunks = len(chunks)

        vectors: list[list[float]] = []
        if chunks:
            self._emit(
                {
                    "type": "index_status",
                    "phase": "embed",
                    "message": f"Embedding {len(chunks)} chunk(s) with {self.embedder.name}",
                    "current": 0,
                    "total": len(chunks),
                }
            )
            batch_size = 64
            try:
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    batch_vectors = self.embedder.embed_documents(
                        [embedding_text(chunk) for chunk in batch]
                    )
                    if len(batch_vectors) != len(batch):
                        raise OrionError("Embedding backend returned the wrong number of vectors")
                    vectors.extend(batch_vectors)
                    self._emit(
                        {
                            "type": "index_status",
                            "phase": "embed",
                            "message": (
                                f"Embedded {min(start + len(batch), len(chunks))}/"
                                f"{len(chunks)} chunk(s)"
                            ),
                            "current": min(start + len(batch), len(chunks)),
                            "total": len(chunks),
                        }
                    )
            except Exception as exc:
                stats.duration = time.monotonic() - started
                stats.errors.append(str(exc))
                self._emit(
                    {
                        "type": "index_status",
                        "phase": "error",
                        "message": f"Embedding failed: {exc}",
                    }
                )
                raise

        self._emit(
            {
                "type": "index_status",
                "phase": "write",
                "message": "Writing vectors to the index",
            }
        )
        if reset_needed:
            self.store.reset()
        else:
            for rel in [*removed, *changed]:
                self.store.delete_path(rel)
        self.store.set_embedder(self.embedder.name)
        if chunks:
            self.store.upsert(chunks, vectors)
        stats.vectors = len(vectors)

        updated_manifest: dict[str, dict] = {} if reset_needed else dict(manifest_files)
        for rel in removed:
            updated_manifest.pop(rel, None)
        for rel in changed:
            info = current[rel]
            updated_manifest[rel] = {
                "hash": info["hash"],
                "size": info["size"],
                "mtime": info["mtime"],
                "chunks": sum(1 for chunk in chunks if chunk.path == rel),
            }
        self._save_manifest(updated_manifest)

        stats.duration = time.monotonic() - started
        self._emit(
            {
                "type": "index_status",
                "phase": "done",
                "message": (
                    f"Indexed {stats.files_indexed} file(s), {stats.chunks} chunk(s) "
                    f"in {stats.duration:.1f}s"
                ),
                "stats": stats.as_dict(),
            }
        )
        return stats

    def clear(self) -> None:
        """Remove all vectors and the manifest."""
        self.store.reset()
        try:
            self.config.manifest_path.unlink(missing_ok=True)
        except OSError:
            pass

    def status(self) -> dict:
        manifest = self._load_manifest()
        files = manifest.get("files") or {}
        return {
            "vectors": self.store.count,
            "files": len(files),
            "embedder": self.store.embedder_name or self.embedder.name,
            "configured_embedder": self.embedder.name,
            "store": type(self.store).__name__,
            "path": str(self.config.chroma_path),
            "updated_at": manifest.get("updated_at"),
        }


__all__ = ["CodeIndexer", "IndexStats"]
