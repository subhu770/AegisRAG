"""AegisRAG Hybrid Retriever Service.

Combines Dense Semantic Embeddings (SentenceTransformer) with Sparse Lexical Search
(BM25Okapi) using Reciprocal Rank Fusion (RRF) for enterprise-grade retrieval precision.
"""

import asyncio
import re
import threading
import time
import uuid
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentIngestRequest,
    DocumentIngestResponse,
    ScoredChunk,
)

# ChromaDB optional/direct integration
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


def tokenize_for_bm25(text: str) -> list[str]:
    """Production tokenization for BM25: alphanumeric extraction, case-folding."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())


class TextChunker:
    """Sliding-window chunker with natural boundary detection and offset tracking."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80, min_chunk_size: int = 40):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        document_id: str,
        title: str = "Untitled",
        source: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split raw text into overlapping windows while respecting sentence/word boundaries."""
        text = text.strip()
        if not text:
            return []

        custom_metadata = custom_metadata or {}
        chunks: list[DocumentChunk] = []
        doc_len = len(text)
        start = 0
        chunk_idx = 0

        # Boundary separators in priority order
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        while start < doc_len:
            # Ideal end offset
            end = min(start + self.chunk_size, doc_len)

            # If not at the end of document, try to find a natural break near the end
            if end < doc_len:
                best_break = -1
                for sep in separators:
                    # Look for break in the last 20% of the window
                    search_start = max(start + int(self.chunk_size * 0.75), start)
                    idx = text.rfind(sep, search_start, end)
                    if idx != -1:
                        best_break = idx + len(sep)
                        break

                if best_break != -1 and best_break > start:
                    end = best_break

            chunk_content = text[start:end].strip()
            if len(chunk_content) >= self.min_chunk_size or not chunks:
                chunk_id = f"{document_id}#c{chunk_idx:04d}"
                meta = ChunkMetadata(
                    document_id=document_id,
                    title=title,
                    chunk_index=chunk_idx,
                    total_chunks=1,  # updated after pass
                    start_char=start,
                    end_char=end,
                    source=source,
                    **custom_metadata,
                )
                chunks.append(DocumentChunk(id=chunk_id, text=chunk_content, metadata=meta))
                chunk_idx += 1

            if end >= doc_len:
                break

            # Slide start forward by (window_size - overlap)
            step = max(self.chunk_size - self.chunk_overlap, 1)
            next_start = start + step

            # Advance to at least the start of the next non-whitespace character
            while next_start < doc_len and text[next_start].isspace():
                next_start += 1

            if next_start <= start:
                next_start = start + 1
            start = next_start

        # Backfill total_chunks in metadata
        total = len(chunks)
        for c in chunks:
            c.metadata.total_chunks = total

        return chunks


class HybridRetriever:
    """Thread-safe hybrid retriever indexing chunks into Vector and BM25 stores."""

    def __init__(
        self,
        embedding_model_name: str | None = None,
        device: str | None = None,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        self.lock = threading.RLock()
        self._embedding_model: SentenceTransformer | None = None

        # In-memory storage of chunks: {chunk_id: DocumentChunk}
        self.chunk_store: dict[str, DocumentChunk] = {}
        # In-memory fallback dense vector store: {chunk_id: np.ndarray}
        self.vector_store: dict[str, np.ndarray] = {}

        # BM25 sparse structures
        self.bm25_corpus_ids: list[str] = []
        self.bm25_tokenized_corpus: list[list[str]] = []
        self.bm25_index: BM25Okapi | None = None

        # ChromaDB client & collection
        self.chroma_client = None
        self.chroma_collection = None
        self._init_vector_store()

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy thread-safe initialization of dense embedding model."""
        if self._embedding_model is None:
            with self.lock:
                if self._embedding_model is None:
                    logger.info("Cold-start loading embedding model: %s on %s", self.embedding_model_name, self.device)
                    self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedding_model

    def warm_up(self) -> None:
        """Pre-warm embedding model weights on application startup."""
        _ = self.embedding_model
        # Execute dummy inference to warm PyTorch computation graph and JIT paths
        _ = self._embedding_model.encode(["AegisRAG warm up query"], normalize_embeddings=True)
        logger.info("Dense embedding model warmed up successfully.")

    def _init_vector_store(self) -> None:
        """Initialize ChromaDB or fall back to internal in-memory vector store."""
        if HAS_CHROMADB and settings.VECTOR_STORE_TYPE == "chroma":
            try:
                import os
                os.makedirs(self.persist_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB persistent collection '%s' ready at %s", self.collection_name, self.persist_dir)
                return
            except Exception as e:
                logger.warning("ChromaDB initialization failed, using high-speed in-memory vector store: %s", e)

        logger.info("Using internal high-speed in-memory vector store")

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        """Process, chunk, and dual-index a document into vector & BM25 indexes."""
        start_time = time.perf_counter()
        doc_id = request.document_id or f"doc_{uuid.uuid4().hex[:10]}"

        chunker = TextChunker(
            chunk_size=request.chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=request.chunk_overlap or settings.CHUNK_OVERLAP,
        )

        chunks = chunker.chunk_text(
            text=request.content,
            document_id=doc_id,
            title=request.title,
            source=request.source,
            custom_metadata=request.metadata,
        )

        if not chunks:
            return DocumentIngestResponse(
                document_id=doc_id,
                chunks_created=0,
                indexing_latency_ms=(time.perf_counter() - start_time) * 1000,
                status="skipped_empty",
            )

        self.index_chunks(chunks)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Ingested document '%s' (%d chunks) in %.2f ms", doc_id, len(chunks), latency_ms)

        return DocumentIngestResponse(
            document_id=doc_id,
            chunks_created=len(chunks),
            indexing_latency_ms=latency_ms,
            status="indexed",
        )

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Dual-index chunks into Dense Vector store and BM25 Sparse Index."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]

        # 1. Compute Dense Embeddings in batches
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=settings.NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )

        with self.lock:
            # 2. Update chunk store and vector index
            for chunk, emb in zip(chunks, embeddings):
                chunk.vector = emb.tolist()
                self.chunk_store[chunk.id] = chunk
                self.vector_store[chunk.id] = emb

            # 3. Upsert to ChromaDB if active
            if self.chroma_collection is not None:
                chroma_metadatas = [
                    {
                        "document_id": c.metadata.document_id,
                        "title": c.metadata.title,
                        "chunk_index": c.metadata.chunk_index,
                        "total_chunks": c.metadata.total_chunks,
                        "start_char": c.metadata.start_char,
                        "end_char": c.metadata.end_char,
                        "source": c.metadata.source or "",
                    }
                    for c in chunks
                ]
                self.chroma_collection.upsert(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=chroma_metadatas,
                )

            # 4. Update BM25 Sparse Index
            for chunk in chunks:
                tokens = tokenize_for_bm25(chunk.text)
                self.bm25_corpus_ids.append(chunk.id)
                self.bm25_tokenized_corpus.append(tokens)

            # Rebuild BM25 with updated corpus
            self.bm25_index = BM25Okapi(
                self.bm25_tokenized_corpus,
                k1=settings.BM25_K1,
                b=settings.BM25_B,
            )
            # Clamp non-positive IDF scores (Lucene standard) so terms in small corpora or frequent terms remain positive
            for term, val in self.bm25_index.idf.items():
                if val <= 0:
                    self.bm25_index.idf[term] = 1e-3

    def dense_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Retrieve top candidates using dense vector cosine similarity.
        
        Returns: list of (chunk_id, cosine_score) sorted descending by score.
        """
        if not self.chunk_store:
            return []

        query_vector = self.embedding_model.encode(
            query,
            normalize_embeddings=settings.NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )

        # Prefer ChromaDB if configured
        if self.chroma_collection is not None:
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=min(top_k, len(self.chunk_store)),
                    include=["distances"],
                )
                hits = []
                if results and results.get("ids") and results["ids"][0]:
                    ids = results["ids"][0]
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                    for cid, dist in zip(ids, distances):
                        # For cosine space: cosine_sim = 1.0 - distance
                        cosine_sim = 1.0 - float(dist)
                        hits.append((cid, cosine_sim))
                    return hits
            except Exception as e:
                logger.warning("ChromaDB dense query failed, falling back to in-memory: %s", e)

        # High-speed NumPy vector search fallback
        chunk_ids = list(self.vector_store.keys())
        if not chunk_ids:
            return []

        matrix = np.array([self.vector_store[cid] for cid in chunk_ids])
        # Since embeddings are normalized, dot product == cosine similarity
        scores = np.dot(matrix, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(chunk_ids[idx], float(scores[idx])) for idx in top_indices]

    def sparse_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Retrieve top candidates using BM25 Okapi lexical matching.
        
        Returns: list of (chunk_id, bm25_score) sorted descending by score.
        """
        if self.bm25_index is None or not self.bm25_corpus_ids:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only include relevant matches with positive BM25 score
                results.append((self.bm25_corpus_ids[idx], score))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[ScoredChunk]:
        """Execute dense and sparse searches and merge candidate ranks via Reciprocal Rank Fusion (RRF).

        RRF Formula:
            RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (rrf_k + rank_m(d))
        """
        if not self.chunk_store:
            return []

        # Candidate pool size to retrieve from each retriever before fusion
        pool_size = max(top_k * 3, 50)

        dense_hits = self.dense_search(query, top_k=pool_size)
        sparse_hits = self.sparse_search(query, top_k=pool_size)

        # Track fusion candidates: {chunk_id: ScoredChunk}
        candidates: dict[str, dict[str, Any]] = {}

        # 1. Process Dense Ranking
        for rank_0, (chunk_id, d_score) in enumerate(dense_hits):
            rank = rank_0 + 1  # 1-based rank
            rrf_delta = 1.0 / (rrf_k + rank)
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "dense_rank": rank,
                    "dense_score": d_score,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "rrf_score": rrf_delta,
                }
            else:
                candidates[chunk_id]["dense_rank"] = rank
                candidates[chunk_id]["dense_score"] = d_score
                candidates[chunk_id]["rrf_score"] += rrf_delta

        # 2. Process Sparse BM25 Ranking
        for rank_0, (chunk_id, s_score) in enumerate(sparse_hits):
            rank = rank_0 + 1
            rrf_delta = 1.0 / (rrf_k + rank)
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": s_score,
                    "rrf_score": rrf_delta,
                }
            else:
                candidates[chunk_id]["sparse_rank"] = rank
                candidates[chunk_id]["sparse_score"] = s_score
                candidates[chunk_id]["rrf_score"] += rrf_delta

        # 3. Sort candidates descending by RRF fused score
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )[:top_k]

        # 4. Construct ScoredChunk objects
        results: list[ScoredChunk] = []
        for chunk_id, stats in sorted_candidates:
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            results.append(
                ScoredChunk(
                    id=chunk.id,
                    text=chunk.text,
                    score=stats["rrf_score"],
                    dense_rank=stats["dense_rank"],
                    dense_score=stats["dense_score"],
                    sparse_rank=stats["sparse_rank"],
                    sparse_score=stats["sparse_score"],
                    rrf_score=stats["rrf_score"],
                    metadata=chunk.metadata.model_dump(),
                )
            )

        return results

    async def ahybrid_search(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[ScoredChunk]:
        """Asynchronous non-blocking wrapper running hybrid search in a thread pool."""
        return await asyncio.to_thread(self.hybrid_search, query, top_k, rrf_k)

    def count(self) -> int:
        """Return total indexed chunk count."""
        return len(self.chunk_store)


# Global singleton instance
retriever_service = HybridRetriever()
