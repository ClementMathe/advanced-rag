# =============================================================================
# Advanced RAG API — Multi-stage Docker build
# =============================================================================
# Base: PyTorch with CUDA for GPU inference
# Single Uvicorn worker (VRAM constraint: 6 GB)
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd --create-home appuser

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY index/ index/
COPY pyproject.toml setup.cfg ./

# Install package in editable mode (for src imports)
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER appuser

# Environment defaults (overridable at runtime)
ENV PIPELINE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
    PIPELINE_LOAD_IN_4BIT=true \
    PIPELINE_TOP_K_RETRIEVE=20 \
    PIPELINE_TOP_K_RERANK=3 \
    PIPELINE_MAX_NEW_TOKENS=80 \
    PIPELINE_TEMPERATURE=0.1 \
    CACHE_REDIS_URL=redis://redis:6379/0 \
    CACHE_SIMILARITY_THRESHOLD=0.92 \
    AUTH_ENABLED=false \
    API_LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# Health check: readiness probe with generous start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')" || exit 1

# Single worker: 3B model in 4-bit fits once in VRAM.
# Two workers would load it twice and OOM on 6 GB.
ENTRYPOINT ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
