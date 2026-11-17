"""Hybrid code retrieval over the ChromaDB index."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from orion.config import OrionConfig
from orion.rag.embeddings import Embedder, build_embedder, embedder_from_name
from orion.rag.indexer import CodeIndexer, IndexEvent, IndexStats
from orion.rag.store import SearchHit, VectorStore, open_store
from orion.workspace import (
    is_source_file,
    iter_source_files,
    read_text_file,
    truncate,
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "where",
    "what", "which", "how", "why", "are", "was", "were", "you", "your", "can",
    "could", "should", "would", "does", "did", "not", "but", "all", "any", "our",
    "its", "has", "have", "had", "will", "just", "then", "than", "them", "they",
    "there", "here", "about", "over", "under", "also", "use", "using", "code",
    "file", "files", "find", "show", "explain", "implement", "please", "help",
}


class CodeRetriever:
    """Combines vector similarity with fast lexical matches."""

    def __init__(
        self,
        config: OrionConfig,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or build_embedder(config)
        self._store = store
        self.on_warning = on_warning

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = open_store(self.config, on_warning=self.on_warning)
        return self._store

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def index(
        self,
        force: bool = False,
        on_event: IndexEvent | None = None,
        paths: Sequence[str] | None = None,
    ) -> IndexStats:
        indexer = CodeIndexer(
            self.config,
            embedder=self.embedder,
            store=self.store,
            on_event=on_event,
        )
        return indexer.index(force=force, paths=paths)

    def ensure_index(self, on_event: IndexEvent | None = None) -> IndexStats | None:
        if not self.config.auto_index:
            return None
        if self.store.count == 0:
            return self.index(on_event=on_event)
        # An index built with a different embedder has a different vector
        # dimension and cannot be queried. Rebuild it instead of falling back
        # to keyword-only search for every request.
        indexed = self.store.embedder_name
        if indexed is not None and indexed != self.embedder.name:
            if on_event:
                on_event(
                    {
                        "type": "index_status",
                        "phase": "rebuild",
                        "message": (
                            f"Index embedder changed ({indexed} -> {self.embedder.name}); "
                            "rebuilding the index"
                        ),
                    }
                )
            return self.index(on_event=on_event)
        return None

    def status(self) -> dict:
        indexer = CodeIndexer(
            self.config,
            embedder=self.embedder,
            store=self.store,
        )
        return indexer.status()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        path_prefix: str | None = None,
        include_keyword: bool = True,
    ) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        limit = k or self.config.top_k
        hits: list[SearchHit] = []

        if self.store.count > 0:
            try:
                embedder = self.embedder
                indexed = self.store.embedder_name
                if indexed is not None and indexed != embedder.name:
                    # Query with the embedder that built the index, otherwise
                    # Chroma rejects vectors with a different dimension.
                    compatible = embedder_from_name(self.config, indexed)
                    if compatible is not None:
                        embedder = compatible
                vector = embedder.embed_query(query)
                hits.extend(self.store.query(vector, k=max(limit * 2, 10), path_prefix=path_prefix))
            except Exception as exc:
                if self.on_warning:
                    self.on_warning(f"Vector search unavailable: {exc}")

        if include_keyword:
            hits.extend(self.keyword_search(query, limit=limit, path_prefix=path_prefix))

        return self._merge(hits, limit)

    def keyword_search(
        self,
        query: str,
        limit: int = 8,
        path_prefix: str | None = None,
    ) -> list[SearchHit]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(query)]
        tokens = [token for token in tokens if token not in _STOPWORDS][:8]
        if not tokens:
            return []
        scored: list[SearchHit] = []
        scanned = 0
        for path in iter_source_files(
            self.config.workspace,
            ignore=self.config.ignore,
            max_bytes=self.config.max_file_bytes,
        ):
            scanned += 1
            if scanned > 4000:
                break
            rel = path.relative_to(self.config.workspace.resolve()).as_posix()
            if path_prefix and not rel.startswith(path_prefix.rstrip("/")):
                continue
            if not is_source_file(path):
                continue
            try:
                text = read_text_file(path, max_bytes=self.config.max_file_bytes)
            except Exception:
                continue
            lowered = text.lower()
            occurrences = sum(lowered.count(token) for token in tokens)
            if not occurrences:
                continue
            name = path.name.lower()
            name_bonus = sum(1 for token in tokens if token in name)
            score = min(0.9, 0.3 + min(occurrences, 12) * 0.04 + name_bonus * 0.12)
            lines = text.splitlines()
            best = 0
            for index, line in enumerate(lines):
                line_lower = line.lower()
                if any(token in line_lower for token in tokens):
                    best = index
                    break
            start = max(0, best - 8)
            end = min(len(lines), best + 28)
            snippet = "\n".join(lines[start:end])
            if not snippet.strip():
                continue
            scored.append(
                SearchHit(
                    path=rel,
                    start_line=start + 1,
                    end_line=end,
                    text=truncate(snippet, 4000, rel),
                    score=score,
                    language="text",
                    source="keyword",
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: max(limit * 2, 8)]

    @staticmethod
    def _overlap(left: SearchHit, right: SearchHit) -> float:
        if left.path != right.path:
            return 0.0
        start = max(left.start_line, right.start_line)
        end = min(left.end_line, right.end_line)
        if end < start:
            return 0.0
        intersection = end - start + 1
        smallest = min(left.end_line - left.start_line + 1, right.end_line - right.start_line + 1)
        return intersection / max(1, smallest)

    def _merge(self, hits: list[SearchHit], limit: int) -> list[SearchHit]:
        if not hits:
            return []
        ordered = sorted(hits, key=lambda hit: hit.score, reverse=True)
        accepted: list[SearchHit] = []
        for hit in ordered:
            duplicate = next(
                (
                    existing
                    for existing in accepted
                    if self._overlap(existing, hit) >= 0.6
                ),
                None,
            )
            if duplicate is not None:
                if hit.score > duplicate.score:
                    duplicate.score = hit.score
                    duplicate.source = "hybrid"
                continue
            accepted.append(hit)
            if len(accepted) >= limit:
                break
        return accepted

    # ------------------------------------------------------------------
    @staticmethod
    def build_context(hits: Sequence[SearchHit], max_chars: int = 9000) -> str:
        if not hits:
            return ""
        blocks: list[str] = []
        used = 0
        for hit in hits:
            body = truncate(hit.text, 4000, hit.location)
            block = (
                f'<file path="{hit.path}" lines="{hit.start_line}-{hit.end_line}" '
                f'score="{hit.score:.2f}" source="{hit.source}">\n'
                f"```{hit.language}\n{body}\n```\n"
                f"</file>"
            )
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        if not blocks:
            return ""
        return (
            "Relevant repository context retrieved with Orion's hybrid RAG "
            "(ChromaDB vectors + lexical matches):\n\n" + "\n\n".join(blocks)
        )


__all__ = ["CodeRetriever"]
