"""Pytest configuration and shared fixtures for AegisRAG."""

import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import DocumentIngestRequest
from app.services.reranker import CrossEncoderReranker, reranker_service
from app.services.retriever import HybridRetriever, retriever_service


@pytest.fixture(scope="session")
def sample_security_text() -> str:
    return (
        "Enterprise Information Security Policy: All employees must use FIDO2 WebAuthn "
        "hardware keys for authentication. Passwords alone and SMS OTP are strictly forbidden. "
        "Data at rest must be encrypted using AES-256-GCM with customer-managed keys (CMEK). "
        "All vector contexts must pass through the Aegis Content Firewall to detect indirect "
        "prompt injections and redact PII before reaching the LLM."
    )


@pytest.fixture(scope="session")
def sample_infra_text() -> str:
    return (
        "Kubernetes Infrastructure Architecture: AegisRAG worker pods require 4 vCPUs and "
        "8Gi memory requests. HPA triggers when CPU exceeds 70% or P95 latency exceeds 250ms. "
        "Ingress Envoy gateway enforces rate limiting at 1,000 requests per minute. "
        "The RTO is under 30 minutes and RPO is under 5 minutes for multi-region failover."
    )


@pytest.fixture
def temp_chroma_dir():
    temp_dir = tempfile.mkdtemp(prefix="aegis_test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_retriever(temp_chroma_dir):
    """Isolated retriever instance using a fresh directory."""
    retriever = HybridRetriever(
        persist_dir=temp_chroma_dir,
        collection_name="test_collection",
    )
    return retriever


@pytest.fixture(scope="session")
def warmed_reranker():
    """Shared pre-warmed reranker instance for fast test execution."""
    reranker_service.warm_up()
    return reranker_service


@pytest.fixture(scope="session")
def warmed_retriever():
    """Shared pre-warmed retriever instance."""
    retriever_service.warm_up()
    return retriever_service


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with initialized lifespan."""
    with TestClient(app) as test_client:
        yield test_client

