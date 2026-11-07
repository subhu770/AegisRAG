"""AegisRAG: Enterprise High-Precision Hybrid RAG & AI Agent Service.

FastAPI main application entrypoint with asynchronous lifespan model weight pre-warming,
structured error handling, and routing.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health_endpoint, router as api_router
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import DocumentIngestRequest, HealthResponse
from app.services.reranker import reranker_service
from app.services.retriever import retriever_service


def auto_load_sample_documents() -> None:
    """Auto-load enterprise documents from data/ directory on cold start if empty."""
    data_dir = settings.DATA_DIR
    if not data_dir.exists() or retriever_service.count() > 0:
        return

    sample_files = list(data_dir.glob("*.md")) + list(data_dir.glob("*.txt"))
    if not sample_files:
        return

    logger.info("Found %d sample enterprise document(s) in %s. Ingesting...", len(sample_files), data_dir)
    for file_path in sample_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            title = file_path.stem.replace("_", " ").title()
            doc_id = f"doc_{file_path.stem}"
            retriever_service.ingest_document(
                DocumentIngestRequest(
                    document_id=doc_id,
                    title=title,
                    content=content,
                    source=str(file_path.name),
                    metadata={"initial_seed": True, "file_path": str(file_path)},
                )
            )
        except Exception as e:
            logger.warning("Failed to seed file %s: %s", file_path, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager.
    
    Pre-warms dense embedding and cross-encoder weights into GPU/CPU memory
    to eliminate cold-start inference latency for client requests.
    """
    logger.info("=== Initializing AegisRAG Core Models (Lifespan Startup) ===")
    warmup_start = time.perf_counter()

    # 1. Pre-warm Sentence Transformer Dense Embedding Model
    logger.info("Pre-warming Dense Embedding Model: %s", settings.EMBEDDING_MODEL_NAME)
    retriever_service.warm_up()

    # 2. Pre-warm Cross-Encoder Contextual Reranker Model
    logger.info("Pre-warming Cross-Encoder Reranker: %s", settings.RERANKER_MODEL_NAME)
    reranker_service.warm_up()

    # 3. Seed sample data if available
    auto_load_sample_documents()

    warmup_total = (time.perf_counter() - warmup_start) * 1000
    logger.info("=== AegisRAG Ready: Cold-start models cached in %.2f ms ===", warmup_total)

    yield

    logger.info("=== Shutting down AegisRAG Service ===")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-grade Hybrid RAG & AI Agent service with Dense Vector + BM25 Reciprocal Rank Fusion & Cross-Encoder Reranking.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for enterprise web portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Request Logging & Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception for %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred.", "error": str(exc)},
    )


# Top-level /health endpoint as specified in requirements
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def root_health():
    """Top-level health check endpoint."""
    return await health_endpoint()


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX, tags=["AegisRAG v1"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

