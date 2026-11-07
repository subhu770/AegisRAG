"""AegisRAG System Configuration.

Centralized configuration managed via Pydantic Settings.
Environment variables can override any field prefixing with AEGIS_.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Settings and Model Hyperparameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AEGIS_",
        extra="ignore",
    )

    # Service Metadata
    PROJECT_NAME: str = "AegisRAG"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Dense Embedding Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    NORMALIZE_EMBEDDINGS: bool = True
    EMBEDDING_BATCH_SIZE: int = 32

    # Sparse BM25 Settings
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # Cross-Encoder Reranker Settings
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_MAX_LENGTH: int = 512
    RERANKER_BATCH_SIZE: int = 16

    # Hybrid Retrieval Defaults
    DEFAULT_TOP_K: int = 10
    DEFAULT_RERANK_TOP_N: int = 5
    DEFAULT_RRF_K: int = 60
    DEFAULT_SCORE_THRESHOLD: float = -5.0  # Cross-encoder logits threshold

    # Chunking Parameters
    CHUNK_SIZE: int = 400  # Target token/character window
    CHUNK_OVERLAP: int = 80  # Sliding overlap
    MIN_CHUNK_SIZE: int = 40

    # Vector Storage
    VECTOR_STORE_TYPE: Literal["chroma", "memory"] = "chroma"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "aegis_enterprise_knowledge"

    # Generator / Synthesizer Settings
    GENERATOR_PROVIDER: Literal["local_synthesis", "openai", "gemini"] = "local_synthesis"
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # System Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for cached configuration instance."""
    return Settings()


settings = get_settings()

