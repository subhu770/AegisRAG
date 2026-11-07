"""Pydantic v2 schemas for AegisRAG.

Defines rigorous data contracts for ingestion, chunking, retrieval, reranking,
synthesis, and diagnostic telemetry.
"""

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class ChunkMetadata(BaseModel):
    """Metadata preserved alongside text chunks."""
    model_config = ConfigDict(extra="allow")

    document_id: str = Field(..., description="Unique parent document identifier")
    title: str = Field(default="Untitled", description="Document title or section heading")
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk index")
    total_chunks: int = Field(default=1, ge=1, description="Total chunks in parent document")
    start_char: int = Field(..., ge=0, description="Character offset start in raw text")
    end_char: int = Field(..., ge=0, description="Character offset end in raw text")
    source: str | None = Field(default=None, description="Origin file path, URL, or data source")


class DocumentChunk(BaseModel):
    """A single discrete chunk ready for hybrid indexing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Globally unique chunk identifier: {doc_id}#c{chunk_index}")
    text: str = Field(..., min_length=1, description="Raw text of the chunk")
    metadata: ChunkMetadata
    vector: list[float] | None = Field(default=None, description="Optional pre-computed dense embedding")


class DocumentIngestRequest(BaseModel):
    """Payload for ingesting raw enterprise documents."""
    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str | None = Field(default=None, description="Optional custom ID. Auto-generated if omitted.")
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=10, description="Full raw document content")
    source: str | None = Field(default="direct_upload", description="Source identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary enterprise metadata")
    chunk_size: int | None = Field(default=None, gt=20, le=4000, description="Override default chunk size")
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000, description="Override default chunk overlap")


class DocumentIngestResponse(BaseModel):
    """Telemetry returned after indexing a document."""
    document_id: str
    chunks_created: int
    indexing_latency_ms: float
    status: str = "indexed"


class ScoredChunk(BaseModel):
    """Candidate chunk scored by dense, sparse, fusion, or reranking stages."""
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str
    score: float = Field(..., description="Primary ranking score (fused RRF or Cross-Encoder logit)")
    dense_rank: int | None = Field(default=None, description="Rank in dense vector search (1-based)")
    dense_score: float | None = Field(default=None, description="Raw cosine similarity score")
    sparse_rank: int | None = Field(default=None, description="Rank in BM25 search (1-based)")
    sparse_score: float | None = Field(default=None, description="Raw BM25 lexical score")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion merged score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder relevance logit")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Retrieval & RAG query payload."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=2, max_length=1500, description="Search query or enterprise prompt")
    top_k: int = Field(default=10, ge=1, le=100, description="Candidate count to pull via hybrid retrieval")
    use_reranker: bool = Field(default=True, description="Enable Cross-Encoder contextual reranking")
    rerank_top_n: int = Field(default=5, ge=1, le=50, description="Number of top chunks returned after reranking")
    score_threshold: float | None = Field(default=None, description="Optional minimum reranker logit / fusion cutoff")
    generate_answer: bool = Field(default=False, description="Whether to assemble context and synthesize an answer")


class Citation(BaseModel):
    """Traceable enterprise citation grounding the answer."""
    citation_id: int
    chunk_id: str
    document_id: str
    title: str | None = None
    snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response payload matching API contracts with precise diagnostic timings."""
    query: str
    context: list[dict[str, Any]] = Field(..., description="Retrieved or reranked context blocks")
    answer: str | None = Field(default=None, description="Synthesized grounded answer if generate_answer is true")
    citations: list[Citation] = Field(default_factory=list, description="Grounding citations mapping directly to context")
    retrieval_latency_ms: float = Field(..., description="Hybrid dense+sparse search latency in ms")
    rerank_latency_ms: float = Field(default=0.0, description="Cross-encoder rerank latency in ms")
    generation_latency_ms: float = Field(default=0.0, description="LLM synthesis latency in ms")
    total_latency_ms: float = Field(..., description="End-to-end request latency in ms")


class HealthResponse(BaseModel):
    """Service health and model readiness check."""
    status: str = "healthy"
    version: str
    vector_store: str
    dense_model: str
    reranker_model: str
    indexed_chunks_count: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """System-level runtime metrics and telemetry."""
    total_queries: int
    total_documents_ingested: int
    total_chunks_indexed: int
    avg_retrieval_latency_ms: float
    avg_rerank_latency_ms: float

