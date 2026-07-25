"""ChromaDB persistent vector store, seeded with the arXiv agentic-AI/RAG corpus."""
from __future__ import annotations
from typing import List
import logging
import os

import chromadb
from chromadb.utils import embedding_functions

from graph.state import EvidenceChunk
from tools.seed_corpus import CORPUS

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
COLLECTION_NAME = "agentic_ai_rag_corpus"

# Default sentence-transformers embedding function (runs locally, CPU is fine --
# these are 384-dim MiniLM embeddings, no GPU needed, so this respects the
# 4GB VRAM constraint just as much as the retrieval-quality goal).
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=PERSIST_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )

    if _collection.count() == 0:
        logger.info("Seeding ChromaDB collection with %d documents", len(CORPUS))
        _collection.add(
            ids=[doc["doc_id"] for doc in CORPUS],
            documents=[doc["summary"] for doc in CORPUS],
            metadatas=[
                {"title": doc["title"], "arxiv_id": doc["arxiv_id"]} for doc in CORPUS
            ],
        )
    return _collection


def vector_search(query: str, sub_question_id: str, n_results: int = 3) -> List[EvidenceChunk]:
    """
    Query the local vector store. Returns an empty list (never raises) on
    failure -- the hybrid design means web search can still cover the
    sub-question if the local corpus has nothing relevant.
    """
    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=n_results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store query failed for %r: %s", query, exc)
        return []

    chunks: List[EvidenceChunk] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

    for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
        # Chroma returns L2 distance (lower = more similar); convert to a
        # 0-1 "similarity-ish" score so it's comparable to Tavily's relevance_score.
        score = None if dist is None else max(0.0, 1.0 - dist / 2.0)
        chunks.append(
            EvidenceChunk(
                sub_question_id=sub_question_id,
                source_type="vector_store",
                source_id=f"arxiv:{meta.get('arxiv_id', doc_id)}",
                title=meta.get("title", doc_id),
                text=text,
                relevance_score=score,
            )
        )
    return chunks


def reset_and_reseed():
    """Utility for tests / dev: wipe and rebuild the collection from CORPUS."""
    global _client, _collection
    _client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    _get_collection()
