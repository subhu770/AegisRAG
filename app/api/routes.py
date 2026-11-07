"""AegisRAG FastAPI Route Endpoints.

Implements high-throughput asynchronous endpoints for query retrieval, cross-encoder reranking,
document ingestion, health diagnostics, and runtime telemetry.
"""

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
)
from app.services.generator import generator_service
from app.services.reranker import reranker_service
from app.services.retriever import retriever_service

router = APIRouter()

# Server start time for uptime reporting
_SERVER_START_TIME = time.time()

# Diagnostic telemetry counters (thread-safe)
class TelemetryTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_queries = 0
        self.total_docs_ingested = 0
        self.total_retrieval_ms = 0.0
        self.total_rerank_ms = 0.0

    def record_query(self, retrieval_ms: float, rerank_ms: float) -> None:
        with self.lock:
            self.total_queries += 1
            self.total_retrieval_ms += retrieval_ms
            self.total_rerank_ms += rerank_ms

    def record_ingest(self) -> None:
        with self.lock:
            self.total_docs_ingested += 1

    def get_metrics(self) -> MetricsResponse:
        with self.lock:
            queries = self.total_queries
            avg_retrieval = (self.total_retrieval_ms / queries) if queries > 0 else 0.0
            avg_rerank = (self.total_rerank_ms / queries) if queries > 0 else 0.0
            return MetricsResponse(
                total_queries=queries,
                total_documents_ingested=self.total_docs_ingested,
                total_chunks_indexed=retriever_service.count(),
                avg_retrieval_latency_ms=round(avg_retrieval, 2),
                avg_rerank_latency_ms=round(avg_rerank, 2),
            )

telemetry = TelemetryTracker()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Hybrid Retrieval and Contextual Reranking",
    description="Executes hybrid dense vector and sparse BM25 retrieval fused with RRF, optionally followed by Cross-Encoder reranking.",
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    start_req = time.perf_counter()

    # 1. Hybrid Retrieval (Dense + BM25 with Reciprocal Rank Fusion)
    retrieval_start = time.perf_counter()
    candidates: list[ScoredChunk] = await retriever_service.ahybrid_search(
        query=request.query,
        top_k=request.top_k,
        rrf_k=settings.DEFAULT_RRF_K,
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    rerank_latency_ms = 0.0
    final_chunks: list[ScoredChunk] = candidates

    # 2. Contextual Cross-Encoder Reranking
    if request.use_reranker and candidates:
        rerank_start = time.perf_counter()
        final_chunks = await reranker_service.arerank(
            query=request.query,
            candidates=candidates,
            top_n=request.rerank_top_n,
            score_threshold=request.score_threshold,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

    # 3. Optional Grounded Answer Generation
    answer: str | None = None
    citations = []
    generation_latency_ms = 0.0

    if request.generate_answer and final_chunks:
        answer, citations, generation_latency_ms = await generator_service.agenerate(
            query=request.query,
            chunks=final_chunks,
        )

    total_latency_ms = (time.perf_counter() - start_req) * 1000
    telemetry.record_query(retrieval_latency_ms, rerank_latency_ms)

    # Format context blocks for output
    context_output: list[dict[str, Any]] = [
        {
            "id": c.id,
            "text": c.text,
            "score": round(c.score, 4),
            "dense_rank": c.dense_rank,
            "dense_score": round(c.dense_score, 4) if c.dense_score is not None else None,
            "sparse_rank": c.sparse_rank,
            "sparse_score": round(c.sparse_score, 4) if c.sparse_score is not None else None,
            "rrf_score": round(c.rrf_score, 6) if c.rrf_score is not None else None,
            "rerank_score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
            "metadata": c.metadata,
        }
        for c in final_chunks
    ]

    return QueryResponse(
        query=request.query,
        context=context_output,
        answer=answer,
        citations=citations,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        rerank_latency_ms=round(rerank_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Enterprise Document",
    description="Chunk raw document and index into both ChromaDB/vector store and BM25 index.",
)
async def ingest_endpoint(request: DocumentIngestRequest) -> DocumentIngestResponse:
    try:
        response = retriever_service.ingest_document(request)
        telemetry.record_ingest()
        return response
    except Exception as e:
        logger.error("Ingestion error for doc %s: %s", request.document_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness",
)
async def health_endpoint() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        vector_store=settings.VECTOR_STORE_TYPE,
        dense_model=settings.EMBEDDING_MODEL_NAME,
        reranker_model=settings.RERANKER_MODEL_NAME,
        indexed_chunks_count=retriever_service.count(),
        uptime_seconds=round(time.time() - _SERVER_START_TIME, 2),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Service Latency and Index Metrics",
)
async def metrics_endpoint() -> MetricsResponse:
    return telemetry.get_metrics()

