"""AegisRAG Context Assembly, Prompt Formatting, and Generation Service.

Assembles retrieved context, enforces citation attribution, formats system prompts
to eliminate hallucinations, and generates grounded responses.
"""

import asyncio
import time
from typing import Sequence

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import Citation, ScoredChunk


SYSTEM_PROMPT_TEMPLATE = """You are AegisRAG, an enterprise AI assistant with zero tolerance for hallucination.
Answer the user's question STRICTLY and ONLY using the provided context blocks.
Rules:
1. Every factual statement must cite its source using bracketed numbers like [1], [2].
2. If the answer cannot be found or deduced from the provided context, state clearly: "I cannot answer this question based on the provided enterprise documentation."
3. Do not extrapolate, assume, or fabricate any policies, numbers, or facts outside the context.
4. Keep the tone concise, authoritative, and professional.
"""


def format_context_prompt(query: str, chunks: Sequence[ScoredChunk]) -> tuple[str, list[Citation]]:
    """Assemble chunks into an indexed context block and generate traceable citation mappings."""
    context_lines = []
    citations: list[Citation] = []

    for idx, chunk in enumerate(chunks, start=1):
        doc_id = chunk.metadata.get("document_id", "unknown_doc")
        title = chunk.metadata.get("title", "Untitled Document")

        # Create snippet (first 160 characters)
        clean_text = chunk.text.replace("\n", " ").strip()
        snippet = clean_text[:160] + "..." if len(clean_text) > 160 else clean_text

        citation = Citation(
            citation_id=idx,
            chunk_id=chunk.id,
            document_id=doc_id,
            title=title,
            snippet=snippet,
            relevance_score=round(chunk.score, 4),
        )
        citations.append(citation)

        context_lines.append(
            f"--- Context Block [{idx}] (Document: '{title}', ID: {doc_id}, Score: {chunk.score:.3f}) ---\n"
            f"{chunk.text}\n"
        )

    context_str = "\n".join(context_lines)
    full_prompt = (
        f"{SYSTEM_PROMPT_TEMPLATE}\n\n"
        f"=== AUTHORIZED ENTERPRISE CONTEXT ===\n"
        f"{context_str}\n"
        f"=== END OF CONTEXT ===\n\n"
        f"USER QUERY: {query}\n\n"
        f"GROUNDED ANSWER (with citations [1], [2], etc.):"
    )

    return full_prompt, citations


class GroundedGenerator:
    """Enterprise generation engine supporting local factual synthesis and cloud LLM providers."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.GENERATOR_PROVIDER

    def _local_synthesize(self, query: str, chunks: Sequence[ScoredChunk], citations: list[Citation]) -> str:
        """Deterministic, hallucination-free extractive synthesis for local/offline environments.

        Extracts the most relevant sentences matching the query terms and formats citations.
        """
        if not chunks:
            return "I cannot answer this question based on the provided enterprise documentation."

        query_terms = set(query.lower().split())
        scored_sentences: list[tuple[float, str, int]] = []

        for idx, chunk in enumerate(chunks, start=1):
            sentences = [s.strip() for s in chunk.text.replace("\n", " ").split(". ") if s.strip()]
            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms.intersection(sent_terms))
                if overlap > 0:
                    score = overlap / max(len(query_terms), 1)
                    scored_sentences.append((score, sent, idx))

        if not scored_sentences:
            # Fall back to primary top chunk excerpt
            top_chunk = chunks[0]
            excerpt = top_chunk.text.split("\n")[0]
            return f"According to enterprise documentation [1]: {excerpt}"

        # Sort sentences by keyword relevance
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sentences[:3]

        answer_parts = []
        for _, sent, cit_idx in top_sentences:
            clean_sent = sent if sent.endswith(".") else f"{sent}."
            answer_parts.append(f"{clean_sent} [{cit_idx}]")

        return " ".join(answer_parts)

    def generate(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
    ) -> tuple[str, list[Citation], float]:
        """Generate grounded answer with citation traceability.
        
        Returns:
            (answer_text, citations, latency_ms)
        """
        start_time = time.perf_counter()
        if not chunks:
            return (
                "I cannot answer this question based on the provided enterprise documentation.",
                [],
                (time.perf_counter() - start_time) * 1000,
            )

        prompt, citations = format_context_prompt(query, chunks)

        # Check for OpenAI API key if configured
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                import httpx
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    json={
                        "model": settings.LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": settings.LLM_TEMPERATURE,
                        "max_tokens": settings.LLM_MAX_TOKENS,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"].strip()
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return answer, citations, latency_ms
                logger.warning("OpenAI API returned %d: %s. Using local synthesis.", resp.status_code, resp.text)
            except Exception as e:
                logger.warning("OpenAI synthesis error: %s. Falling back to local.", e)

        # High-precision local synthesis
        answer = self._local_synthesize(query, chunks, citations)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return answer, citations, latency_ms

    async def agenerate(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
    ) -> tuple[str, list[Citation], float]:
        """Asynchronous non-blocking generation wrapper."""
        return await asyncio.to_thread(self.generate, query, chunks)


# Global singleton generator instance
generator_service = GroundedGenerator()

