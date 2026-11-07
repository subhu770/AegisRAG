"""API integration and contract tests for AegisRAG endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoints(client: TestClient):
    """Verify both /health and /api/v1/health return 200 and valid schema."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "vector_store" in data
        assert "dense_model" in data
        assert "reranker_model" in data
        assert "indexed_chunks_count" in data
        assert "uptime_seconds" in data


def test_document_ingestion_contract(client: TestClient):
    """Test POST /api/v1/ingest validation and success contract."""
    # 1. Invalid payload (missing content)
    invalid_resp = client.post("/api/v1/ingest", json={"title": "No Content"})
    assert invalid_resp.status_code == 422

    # 2. Valid ingestion
    payload = {
        "document_id": "test_iam_policy",
        "title": "IAM Security Standards",
        "content": (
            "All enterprise developers must utilize short-lived STS tokens. "
            "Permanent AWS IAM access keys are deactivated within 24 hours of creation. "
            "Session policies enforce least privilege boundary conditions."
        ),
        "source": "confluence/cloud-sec",
        "metadata": {"confidentiality": "restricted"},
    }
    resp = client.post("/api/v1/ingest", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["document_id"] == "test_iam_policy"
    assert data["chunks_created"] >= 1
    assert data["indexing_latency_ms"] >= 0.0
    assert data["status"] == "indexed"


def test_query_hybrid_and_rerank_contract(client: TestClient):
    """Test POST /api/v1/query returns exact expected schema and timings."""
    # First ingest a test document to ensure query hits
    client.post(
        "/api/v1/ingest",
        json={
            "document_id": "k8s_sizing_spec",
            "title": "Kubernetes Resource Allocation",
            "content": (
                "Kubernetes worker node requirements: Worker pods require 4 vCPU requests "
                "and 8 vCPU limits on AMD EPYC processors. Memory allocation is 8Gi requests and 16Gi limits."
            ),
        },
    )

    # 1. Hybrid query without reranker
    payload_no_rerank = {
        "query": "What are the CPU and memory limits for Kubernetes worker pods?",
        "top_k": 5,
        "use_reranker": False,
    }
    resp = client.post("/api/v1/query", json=payload_no_rerank)
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == payload_no_rerank["query"]
    assert isinstance(data["context"], list)
    assert len(data["context"]) > 0
    assert "retrieval_latency_ms" in data
    assert data["retrieval_latency_ms"] > 0.0
    assert data["rerank_latency_ms"] == 0.0
    assert "total_latency_ms" in data
    assert "X-Process-Time-Ms" in resp.headers

    # 2. Hybrid query with Cross-Encoder reranking
    payload_rerank = {
        "query": "What are the CPU and memory limits for Kubernetes worker pods?",
        "top_k": 5,
        "use_reranker": True,
        "rerank_top_n": 3,
        "generate_answer": True,
    }
    resp2 = client.post("/api/v1/query", json=payload_rerank)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["context"]) <= 3
    assert data2["rerank_latency_ms"] > 0.0
    assert data2["answer"] is not None
    assert isinstance(data2["citations"], list)
    assert len(data2["citations"]) > 0

    first_cit = data2["citations"][0]
    assert "citation_id" in first_cit
    assert "chunk_id" in first_cit
    assert "document_id" in first_cit
    assert "snippet" in first_cit
    assert "relevance_score" in first_cit


def test_metrics_telemetry(client: TestClient):
    """Test GET /api/v1/metrics returns aggregated system metrics."""
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert "total_queries" in metrics
    assert metrics["total_queries"] >= 1
    assert "total_documents_ingested" in metrics
    assert "total_chunks_indexed" in metrics
    assert "avg_retrieval_latency_ms" in metrics
    assert "avg_rerank_latency_ms" in metrics

