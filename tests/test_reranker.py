"""Unit and precision tests for CrossEncoderReranker service."""

import time
import pytest
from app.models.schemas import ScoredChunk
from app.services.reranker import CrossEncoderReranker


def test_cross_encoder_precision(warmed_reranker):
    """Cross-encoder must prioritize passage with deep contextual match over lexical distractors."""
    query = "What is the mandatory protocol for data in transit?"

    candidates = [
        ScoredChunk(
            id="c1",
            text="Employee parking permits must be renewed annually at the front desk.",
            score=0.01,
            metadata={"doc_id": "parking"},
        ),
        ScoredChunk(
            id="c2",
            text="Data in transit strictly requires TLS 1.3 with mandatory forward secrecy across all load balancers.",
            score=0.02,
            metadata={"doc_id": "network_sec"},
        ),
        ScoredChunk(
            id="c3",
            text="Public transit passes are subsidized up to 50 percent for all engineering personnel.",
            score=0.015,
            metadata={"doc_id": "benefits"},
        ),
    ]

    reranked = warmed_reranker.rerank(query=query, candidates=candidates, top_n=3)

    assert len(reranked) == 3
    # c2 must be ranked first by a wide margin
    assert reranked[0].id == "c2"
    assert reranked[0].score > reranked[1].score
    assert reranked[0].score > reranked[2].score


def test_cross_encoder_score_threshold(warmed_reranker):
    """Verify that score_threshold prunes low-scoring candidates."""
    query = "What is the CPU allocation for Kubernetes worker pods?"

    candidates = [
        ScoredChunk(
            id="c1",
            text="AegisRAG worker pods require 4 vCPU requests and 8 vCPU limits on AMD EPYC processors.",
            score=0.02,
            metadata={},
        ),
        ScoredChunk(
            id="c2",
            text="The cafeteria serves organic vegetarian lunch from 12pm to 2pm every weekday.",
            score=0.01,
            metadata={},
        ),
    ]

    # Relevant passage scores ~ -0.85, irrelevant cafeteria passage scores ~ -11.45
    # With a threshold of -5.0, the irrelevant cafeteria chunk will be pruned and c1 retained
    reranked = warmed_reranker.rerank(query=query, candidates=candidates, top_n=2, score_threshold=-5.0)

    assert len(reranked) == 1
    assert reranked[0].id == "c1"


def test_cross_encoder_empty_candidates(warmed_reranker):
    """Empty candidate list should return empty list immediately."""
    assert warmed_reranker.rerank("any query", candidates=[], top_n=5) == []


def test_cross_encoder_latency_benchmark(warmed_reranker):
    """Benchmark reranker latency on 5 candidates (should be < 300ms on CPU)."""
    query = "How often must service account tokens be rotated?"
    candidates = [
        ScoredChunk(id=f"c{i}", text=f"Service account tokens policy clause {i}: rotation interval rules.", score=0.01, metadata={})
        for i in range(5)
    ]

    t0 = time.perf_counter()
    results = warmed_reranker.rerank(query=query, candidates=candidates, top_n=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(results) == 5
    assert elapsed_ms < 600.0, f"Reranker latency {elapsed_ms:.2f}ms exceeded 600ms budget"
