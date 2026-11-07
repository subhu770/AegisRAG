# ==============================================================================
# AegisRAG Production Container Image
# Multi-stage security-hardened container for high-precision hybrid RAG service
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools for native C++ bindings (ChromaDB / tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM python:3.11-slim AS runtime

# Create non-root system user and group
RUN groupadd -g 1001 aegisgroup && \
    useradd -u 1001 -g aegisgroup -s /bin/bash -m aegisuser

WORKDIR /app

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code and seed data
COPY app/ /app/app/
COPY data/ /app/data/

# Set environment variables for high-performance inference
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    AEGIS_VECTOR_STORE_TYPE=chroma \
    AEGIS_CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db \
    TORCH_HOME=/home/aegisuser/.cache/torch \
    HF_HOME=/home/aegisuser/.cache/huggingface

# Ensure correct file permissions
RUN mkdir -p /app/data/chroma_db /home/aegisuser/.cache && \
    chown -R aegisuser:aegisgroup /app /home/aegisuser

USER aegisuser

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

