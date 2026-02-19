"""
FastAPI application factory with lifespan context manager.

Usage:
    # Development (no model loading):
    uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000

    # Production (loads pipeline):
    PIPELINE_LOAD_IN_4BIT=true uvicorn src.api.app:create_app --factory --workers 1
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from loguru import logger
from prometheus_client import make_asgi_app

from src.api.cache import SemanticCache
from src.api.circuit_breaker import CostCircuitBreaker
from src.api.config import Settings
from src.api.state import AppState


def _load_pipeline(settings: Settings):
    """Load the RAG pipeline synchronously (called in executor)."""
    from src.chunking import Chunk
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.generator import LLMGenerator
    from src.reranker import CrossEncoderReranker
    from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    cfg = settings.pipeline

    logger.info(f"Loading embedding model: {cfg.embedding_model}")
    embed_model = EmbeddingModel(cfg.embedding_model, device="cuda")

    logger.info("Loading FAISS index from index/squad")
    faiss_index = FAISSIndex.load("index/squad")
    dense_retriever = DenseRetriever(faiss_index, embed_model)

    logger.info("Building BM25 index from FAISS chunk metadata")
    chunks = []
    for meta in faiss_index.chunk_metadata:
        chunk = Chunk(
            content=meta["content"],
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            start_char=0,
            end_char=len(meta["content"]),
            chunk_index=meta["chunk_index"],
            metadata=meta.get("metadata", {}),
        )
        chunks.append(chunk)
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever, k_rrf=60, dense_weight=0.9, sparse_weight=0.1
    )

    logger.info(f"Loading reranker: {cfg.reranker_model}")
    reranker = CrossEncoderReranker(cfg.reranker_model)

    logger.info(f"Loading LLM: {cfg.model_name} (4-bit={cfg.load_in_4bit})")
    generator = LLMGenerator(
        model_name=cfg.model_name,
        load_in_4bit=cfg.load_in_4bit,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        k_retrieve=cfg.top_k_retrieve,
        k_rerank=cfg.top_k_rerank,
        use_reranking=True,
        use_generation=True,
    )

    logger.info("Pipeline loaded successfully")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load pipeline + cache + circuit breaker. Shutdown: cleanup."""
    state: AppState = app.state.app_state
    settings: Settings = state.settings

    # Initialize circuit breaker
    cb_cfg = settings.circuit_breaker
    state.circuit_breaker = CostCircuitBreaker(
        cost_limit_eur=cb_cfg.cost_limit_eur,
        cost_per_1k_input_tokens=cb_cfg.cost_per_1k_input_tokens,
        cost_per_1k_output_tokens=cb_cfg.cost_per_1k_output_tokens,
        window_seconds=cb_cfg.window_seconds,
    )

    # Initialize semantic cache
    cache_cfg = settings.cache
    try:
        from sentence_transformers import SentenceTransformer

        cache_embed_model = SentenceTransformer(cache_cfg.cache_model, device="cpu")
        cache = SemanticCache(
            redis_url=cache_cfg.redis_url,
            similarity_threshold=cache_cfg.similarity_threshold,
            ttl_seconds=cache_cfg.ttl_seconds,
            embed_model=cache_embed_model,
        )
        await cache.connect()
        if await cache.ping():
            state.cache = cache
            logger.info("Semantic cache connected")
        else:
            logger.warning("Redis not reachable — cache disabled")
    except Exception as e:
        logger.warning(f"Cache init failed: {e} — cache disabled")

    # Load pipeline in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        logger.info("Loading RAG pipeline...")
        state.pipeline = await loop.run_in_executor(None, _load_pipeline, settings)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.warning("API will start in degraded mode (no pipeline)")

    yield

    # Shutdown
    if state.cache:
        await state.cache.disconnect()
    logger.info("Shutting down API")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """FastAPI application factory."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Advanced RAG API",
        description="Production RAG pipeline with hybrid retrieval, reranking, and generation.",
        version="0.8.0",
        lifespan=lifespan,
    )

    # Attach state
    app_state = AppState(settings=settings)
    app.state.app_state = app_state

    # Middleware
    from src.api.middleware import RequestIDMiddleware

    app.add_middleware(RequestIDMiddleware)

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Register routes
    from src.api.routes.health import router as health_router
    from src.api.routes.query import router as query_router

    app.include_router(health_router)
    app.include_router(query_router)

    return app
