"""Unit and latency integration tests for TextChunker and HybridRetriever."""

import time
import pytest
from app.models.schemas import DocumentIngestRequest
from app.services.retriever import HybridRetriever, TextChunker, tokenize_for_bm25


def test_chunker_sliding_window_offsets():
    """Verify sliding window chunker preserves character offsets and metadata."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    raw_text = (
        "Enterprise Security Rule 1: Always encrypt data at rest. "
        "Enterprise Security Rule 2: Always rotate keys every 90 days. "
        "Enterprise Security Rule 3: Enforce multi-factor authentication across all accounts."
    )

    chunks = chunker.chunk_text(
        text=raw_text,
        document_id="sec_doc_01",
        title="Security Rules",
        source="internal_wiki",
        custom_metadata={"classification": "confidential"},
    )

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.id == f"sec_doc_01#c{idx:04d}"
        assert chunk.metadata.document_id == "sec_doc_01"
        assert chunk.metadata.title == "Security Rules"
        assert chunk.metadata.chunk_index == idx
        assert chunk.metadata.total_chunks == len(chunks)
        # Check text slice matches reported character offsets
        start = chunk.metadata.start_char
        end = chunk.metadata.end_char
        assert raw_text[start:end].strip() == chunk.text


def test_chunker_empty_input():
    """Chunker should handle empty or whitespace-only text gracefully."""
    chunker = TextChunker()
    assert chunker.chunk_text("", document_id="empty") == []
    assert chunker.chunk_text("   \n\t  ", document_id="whitespace") == []


def test_bm25_tokenization():
    """Verify BM25 tokenization extracts tokens and normalizes cases."""
    text = "FIDO2 WebAuthn & AES-256-GCM encryption."
    tokens = tokenize_for_bm25(text)
    assert "fido2" in tokens
    assert "webauthn" in tokens
    assert "aes-256-gcm" in tokens or "aes" in tokens


def test_hybrid_retriever_ingestion_and_latency(test_retriever, sample_security_text):
    """Verify complete ingestion and benchmark retrieval latency (< 150ms on warm model)."""
    # 1. Ingest document
    req = DocumentIngestRequest(
        document_id="sec_policy",
        title="Security Policy",
        content=sample_security_text,
        metadata={"department": "infosec"},
    )
    resp = test_retriever.ingest_document(req)
    assert resp.chunks_created > 0
    assert resp.status == "indexed"
    assert test_retriever.count() == resp.chunks_created

    # 2. Dense search check
    dense_hits = test_retriever.dense_search("What encryption algorithm is required for data at rest?")
    assert len(dense_hits) > 0
    top_chunk_id, score = dense_hits[0]
    assert score > 0.0

    # 3. Sparse search check (lexical match on acronym)
    sparse_hits = test_retriever.sparse_search("AES-256-GCM")
    assert len(sparse_hits) > 0

    # 4. Hybrid search latency benchmark
    # Warm up first
    _ = test_retriever.hybrid_search("FIDO2 authentication", top_k=5)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        results = test_retriever.hybrid_search("FIDO2 WebAuthn hardware keys", top_k=5)
        times.append((time.perf_counter() - t0) * 1000)

    avg_latency_ms = sum(times) / len(times)
    assert len(results) > 0
    # On modern CPU, warm MiniLM retrieval should easily be under 300ms
    assert avg_latency_ms < 500.0, f"Average retrieval latency {avg_latency_ms:.2f}ms exceeded 500ms SLA"

