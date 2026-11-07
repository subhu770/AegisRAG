"""AegisRAG Cross-Encoder Contextual Reranking Service.

Applies full self-attention over (query, document) pairs using a transformer cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2) to suppress retrieval false-positives and
boost domain precision.
"""

import asyncio
import threading
import time
from typing import Sequence

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import ScoredChunk


class CrossEncoderReranker:
    """Production Cross-Encoder reranking service with batch inference and lazy caching."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device or settings.RERANKER_DEVICE
        self.max_length = max_length or settings.RERANKER_MAX_LENGTH

        self.lock = threading.RLock()
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy thread-safe initialization of Cross-Encoder model."""
        if self._model is None:
            with self.lock:
                if self._model is None:
                    logger.info("Cold-start loading Cross-Encoder: %s on %s", self.model_name, self.device)
                    self._model = CrossEncoder(
                        self.model_name,
                        max_length=self.max_length,
                        device=self.device,
                    )
        return self._model

    def warm_up(self) -> None:
        """Pre-warm cross-encoder weights to eliminate latency spike on first query."""
        _ = self.model
        # Execute dummy inference pair
        _ = self._model.predict([("warmup query", "warmup passage context")])
        logger.info("Cross-Encoder reranker model warmed up successfully.")

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_n: int = 5,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        """Score (query, chunk_text) pairs and return top_n candidates ordered by relevance logit.

        Args:
            query: User search query or prompt.
            candidates: ScoredChunk candidates from hybrid retrieval.
            top_n: Number of reranked candidates to return.
            score_threshold: Optional logit cutoff to prune irrelevant chunks.

        Returns:
            list[ScoredChunk] ordered descending by Cross-Encoder score.
        """
        if not candidates:
            return []

        start_time = time.perf_counter()

        # Build pair tuples for transformer cross-attention
        pairs = [(query, c.text) for c in candidates]

        # Predict logits in batches
        scores = self.model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

        reranked_chunks: list[ScoredChunk] = []
        for orig_chunk, score_val in zip(candidates, scores):
            score_float = float(score_val)
            if score_threshold is not None and score_float < score_threshold:
                continue

            # Clone chunk with updated scores
            reranked_chunks.append(
                ScoredChunk(
                    id=orig_chunk.id,
                    text=orig_chunk.text,
                    score=score_float,  # Cross-Encoder score becomes primary ranking score
                    dense_rank=orig_chunk.dense_rank,
                    dense_score=orig_chunk.dense_score,
                    sparse_rank=orig_chunk.sparse_rank,
                    sparse_score=orig_chunk.sparse_score,
                    rrf_score=orig_chunk.rrf_score,
                    rerank_score=score_float,
                    metadata=orig_chunk.metadata,
                )
            )

        # Sort descending by cross-encoder score
        reranked_chunks.sort(key=lambda x: x.score, reverse=True)
        final_top = reranked_chunks[:top_n]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("Reranked %d candidates down to %d in %.2f ms", len(candidates), len(final_top), elapsed_ms)

        return final_top

    async def arerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_n: int = 5,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        """Asynchronous non-blocking wrapper for FastAPI request workers."""
        return await asyncio.to_thread(self.rerank, query, candidates, top_n, score_threshold)


# Global singleton instance
reranker_service = CrossEncoderReranker()

