"""Retrieval-augmented generation for Orion.

The RAG layer indexes a repository into ChromaDB and exposes hybrid
(vector + keyword) search to the agents.
"""

from __future__ import annotations

from orion.rag.chunker import Chunk, chunk_text, content_hash, embedding_text
from orion.rag.embeddings import (
    Embedder,
    GeminiEmbedder,
    LocalHashEmbedder,
    build_embedder,
)
from orion.rag.indexer import CodeIndexer, IndexStats
from orion.rag.retriever import CodeRetriever
from orion.rag.store import (
    ChromaVectorStore,
    JsonVectorStore,
    SearchHit,
    VectorStore,
    open_store,
)

__all__ = [
    "ChromaVectorStore",
    "Chunk",
    "CodeIndexer",
    "CodeRetriever",
    "Embedder",
    "GeminiEmbedder",
    "IndexStats",
    "JsonVectorStore",
    "LocalHashEmbedder",
    "SearchHit",
    "VectorStore",
    "build_embedder",
    "chunk_text",
    "content_hash",
    "embedding_text",
    "open_store",
]
