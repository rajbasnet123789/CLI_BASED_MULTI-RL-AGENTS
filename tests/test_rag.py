"""Chunking, embeddings, vector stores, indexing, and retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.config import OrionConfig
from orion.rag.chunker import chunk_text, embedding_text
from orion.rag.embeddings import LocalHashEmbedder, build_embedder
from orion.rag.indexer import CodeIndexer
from orion.rag.retriever import CodeRetriever
from orion.rag.store import JsonVectorStore, open_store


def test_local_embedder_is_deterministic_and_normalized() -> None:
    embedder = LocalHashEmbedder()
    first = embedder.embed_query("database connection")
    second = embedder.embed_query("database connection")
    other = embedder.embed_query("frontend button styles")
    assert first == second
    assert first != other
    assert len(first) == embedder.dimensions
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_build_embedder_selects_local(config: OrionConfig) -> None:
    embedder = build_embedder(config)
    assert isinstance(embedder, LocalHashEmbedder)


def test_local_provider_forces_offline_embeddings(config: OrionConfig) -> None:
    local = config.with_overrides(llm_provider="local", embeddings="gemini")
    assert local.is_local is True
    assert isinstance(build_embedder(local), LocalHashEmbedder)


def test_json_store_roundtrip(tmp_path: Path) -> None:
    embedder = LocalHashEmbedder()
    store = JsonVectorStore(tmp_path / "vectors")
    chunks = chunk_text(
        "def connect():\n    return 'sqlite:///db'\n\n\ndef close():\n    pass\n",
        "pkg/db.py",
        max_chars=200,
        overlap=20,
    )
    vectors = embedder.embed_documents([embedding_text(chunk) for chunk in chunks])
    store.upsert(chunks, vectors)
    store.set_embedder(embedder.name)
    assert store.count == len(chunks)
    assert store.embedder_name == embedder.name
    assert store.paths() == {"pkg/db.py"}

    hits = store.query(embedder.embed_query("sqlite database connection"), k=3)
    assert hits
    assert hits[0].path == "pkg/db.py"
    assert hits[0].source == "vector"

    store.delete_path("pkg/db.py")
    assert store.count == 0
    assert store.query(embedder.embed_query("anything"), k=3) == []


def test_json_store_reloads_from_disk(tmp_path: Path) -> None:
    embedder = LocalHashEmbedder()
    directory = tmp_path / "vectors"
    store = JsonVectorStore(directory)
    chunks = chunk_text("x = 1\n" * 40, "a.py", max_chars=200, overlap=10)
    store.upsert(chunks, embedder.embed_documents([embedding_text(c) for c in chunks]))
    store.set_embedder("test")

    reopened = JsonVectorStore(directory)
    assert reopened.count == len(chunks)
    assert reopened.embedder_name == "test"


def test_indexer_incremental_manifest(config: OrionConfig, workspace: Path) -> None:
    embedder = LocalHashEmbedder()
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    indexer = CodeIndexer(config, embedder=embedder, store=store)

    first = indexer.index()
    assert first.files_indexed >= 3
    assert first.chunks > 0
    assert first.vectors == first.chunks
    assert store.count > 0
    assert config.manifest_path.is_file()

    second = indexer.index()
    assert second.files_indexed == 0
    assert second.files_unchanged == first.files_indexed
    assert second.chunks == 0

    (workspace / "pkg" / "app.py").write_text(
        "def changed():\n    return 42\n", encoding="utf-8"
    )
    third = indexer.index()
    assert third.files_indexed == 1

    (workspace / "pkg" / "utils.py").unlink()
    fourth = indexer.index()
    assert fourth.files_removed == 1
    assert "pkg/utils.py" not in store.paths()

    status = indexer.status()
    assert status["vectors"] == store.count
    assert status["embedder"] == embedder.name


def test_indexer_force_rebuild(config: OrionConfig, workspace: Path) -> None:
    embedder = LocalHashEmbedder()
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    indexer = CodeIndexer(config, embedder=embedder, store=store)
    first = indexer.index()
    forced = indexer.index(force=True)
    assert forced.files_indexed == first.files_indexed
    assert forced.vectors == store.count


def test_indexer_skips_binary_and_ignored(config: OrionConfig, workspace: Path) -> None:
    (workspace / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "x.js").write_text("x\n", encoding="utf-8")
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    indexer = CodeIndexer(config, embedder=LocalHashEmbedder(), store=store)
    stats = indexer.index()
    paths = store.paths()
    assert "image.png" not in paths
    assert "node_modules/x.js" not in paths
    assert stats.files_total >= 3


def test_retriever_hybrid_search(config: OrionConfig, workspace: Path) -> None:
    embedder = LocalHashEmbedder()
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    indexer = CodeIndexer(config, embedder=embedder, store=store)
    indexer.index()

    retriever = CodeRetriever(config, embedder=embedder, store=store)
    hits = retriever.search("database connection string", k=3)
    assert hits
    assert any("app.py" in hit.path for hit in hits)
    context = retriever.build_context(hits)
    assert "<file path=" in context
    assert "database" in context.lower()

    keyword = retriever.keyword_search("slugify", limit=3)
    assert keyword
    assert any("utils.py" in hit.path for hit in keyword)


def test_retriever_ensure_index_respects_auto_index(config: OrionConfig) -> None:
    retriever = CodeRetriever(
        config,
        embedder=LocalHashEmbedder(),
        store=JsonVectorStore(config.workspace / ".orion" / "vectors"),
    )
    assert retriever.ensure_index() is None
    assert retriever.search("database")


class RenamedLocalEmbedder(LocalHashEmbedder):
    """A local embedder with a custom name, to simulate backend changes."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_indexer_rebuilds_when_embedder_changes(
    config: OrionConfig, workspace: Path
) -> None:
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    first = CodeIndexer(
        config, embedder=RenamedLocalEmbedder("local-hash-a"), store=store
    ).index()
    assert store.count > 0

    second = CodeIndexer(
        config, embedder=RenamedLocalEmbedder("local-hash-b"), store=store
    ).index()
    assert second.files_indexed == first.files_indexed
    assert second.vectors == second.chunks > 0
    assert store.embedder_name == "local-hash-b"
    assert store.count == second.vectors


def test_ensure_index_rebuilds_on_embedder_mismatch(
    config: OrionConfig, workspace: Path
) -> None:
    store = JsonVectorStore(workspace / ".orion" / "vectors")
    CodeIndexer(
        config, embedder=RenamedLocalEmbedder("local-hash-a"), store=store
    ).index()

    auto = config.with_overrides(auto_index=True)
    retriever = CodeRetriever(
        auto, embedder=RenamedLocalEmbedder("local-hash-b"), store=store
    )
    stats = retriever.ensure_index()
    assert stats is not None
    assert stats.vectors > 0
    assert store.embedder_name == "local-hash-b"


def test_chroma_embedder_metadata_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from orion.rag.store import ChromaVectorStore

    directory = tmp_path / "chroma"
    store = ChromaVectorStore(directory)
    assert store.embedder_name is None

    store.set_embedder("local-hash-v1")
    assert store.embedder_name == "local-hash-v1"

    # Reopening must not attempt to change the immutable hnsw:space metadata.
    reopened = ChromaVectorStore(directory)
    assert reopened.embedder_name == "local-hash-v1"
    reopened.set_embedder("gemini:gemini-embedding-001:768")
    assert reopened.embedder_name == "gemini:gemini-embedding-001:768"

    reopened.reset()
    assert reopened.count == 0
    assert reopened.embedder_name is None


def test_open_store_falls_back_to_json(
    config: OrionConfig, monkeypatch, tmp_path: Path
) -> None:
    import orion.rag.store as store_module

    def explode(directory):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(store_module, "ChromaVectorStore", explode)
    warnings: list[str] = []
    store = open_store(config, on_warning=warnings.append)
    assert isinstance(store, JsonVectorStore)
    assert warnings
