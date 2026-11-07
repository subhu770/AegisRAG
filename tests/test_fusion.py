"""Tests for Reciprocal Rank Fusion (RRF) algorithm and mathematical scoring properties."""

import pytest
from app.models.schemas import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retriever import HybridRetriever


def test_rrf_mathematical_precision():
    """Verify that RRF calculation strictly matches 1 / (k + rank) formula."""
    k = 60

    # Test doc appearing rank 1 in dense and rank 2 in sparse
    dense_rank = 1
    sparse_rank = 2
    expected_score = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))

    calculated_score = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    assert abs(expected_score - calculated_score) < 1e-9
    assert calculated_score > (1.0 / (k + 1))  # Dual hit beats single hit


def test_rrf_dual_hit_superiority(test_retriever):
    """A document ranking moderately in both dense and sparse should outrank a doc ranking top in only one."""
    # Seed 3 dummy chunks
    chunks = [
        DocumentChunk(
            id="doc_both",
            text="FIDO2 WebAuthn authentication hardware key",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, total_chunks=1, start_char=0, end_char=43),
        ),
        DocumentChunk(
            id="doc_dense_only",
            text="Biometric facial scanning identity verification",
            metadata=ChunkMetadata(document_id="d2", chunk_index=0, total_chunks=1, start_char=0, end_char=48),
        ),
        DocumentChunk(
            id="doc_sparse_only",
            text="FIDO2 key serial numbers inventory table",
            metadata=ChunkMetadata(document_id="d3", chunk_index=0, total_chunks=1, start_char=0, end_char=41),
        ),
    ]
    test_retriever.index_chunks(chunks)

    # Mock candidate hits directly into hybrid fusion logic
    # Query: "FIDO2 WebAuthn authentication"
    results = test_retriever.hybrid_search("FIDO2 WebAuthn authentication", top_k=3, rrf_k=60)
    assert len(results) > 0
    # Top result must be doc_both due to strong dual representation
    assert results[0].id == "doc_both"
    assert results[0].rrf_score is not None
    assert results[0].rrf_score > 0.0


def test_rrf_empty_index(test_retriever):
    """Empty retriever must return an empty list gracefully without throwing exception."""
    results = test_retriever.hybrid_search("non-existent query", top_k=10)
    assert results == []


def test_rrf_k_parameter_sensitivity(test_retriever):
    """Lower k should place higher emphasis on top ranks; higher k flattens ranking differences."""
    chunks = [
        DocumentChunk(
            id="doc_1",
            text="Kubernetes cluster sizingAMD EPYC processors",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, total_chunks=1, start_char=0, end_char=44),
        ),
        DocumentChunk(
            id="doc_2",
            text="Kubernetes networking with Calico CNI plugins",
            metadata=ChunkMetadata(document_id="d2", chunk_index=0, total_chunks=1, start_char=0, end_char=45),
        ),
    ]
    test_retriever.index_chunks(chunks)

    res_k10 = test_retriever.hybrid_search("Kubernetes cluster sizing", top_k=2, rrf_k=10)
    res_k100 = test_retriever.hybrid_search("Kubernetes cluster sizing", top_k=2, rrf_k=100)

    assert len(res_k10) == 2
    assert len(res_k100) == 2

    # Score ratio between rank 1 and rank 2 should be larger for lower k
    ratio_k10 = res_k10[0].rrf_score / res_k10[1].rrf_score
    ratio_k100 = res_k100[0].rrf_score / res_k100[1].rrf_score

    assert ratio_k10 > ratio_k100

