"""
Query endpoints.

- POST /query        — standard JSON response (with cache check)
- POST /query/stream — Server-Sent Events (token-by-token streaming)
- POST /batch        — up to 50 queries, sequential processing
"""

import asyncio
import json
import time
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from src.api.auth import verify_api_key
from src.api.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    GENERATION_LATENCY,
    QUERY_LATENCY,
    REQUEST_COUNT,
    RETRIEVAL_LATENCY,
)
from src.api.schemas import (
    BatchQueryRequest,
    BatchQueryResponse,
    QueryRequest,
    QueryResponse,
    Source,
)

router = APIRouter(tags=["query"])


def _run_pipeline(pipeline, query: str, top_k: int = 3):
    """Run pipeline.query() synchronously (called via run_in_executor)."""
    return pipeline.query(query, return_intermediate=True)


def _build_sources(result: dict, include: bool) -> list:
    """Extract sources from pipeline result."""
    if not include:
        return []
    chunks = result.get("reranked_chunks", result.get("chunks", []))
    sources = []
    for chunk in chunks:
        sources.append(
            Source(
                content=chunk.get("content", ""),
                score=chunk.get("rerank_score", chunk.get("score", 0.0)),
                document_id=chunk.get("doc_id", ""),
            )
        )
    return sources


def _response_to_cache_dict(resp: QueryResponse) -> dict:
    """Convert QueryResponse to a JSON-serializable dict for caching."""
    return resp.model_dump()


@router.post("/query", dependencies=[Depends(verify_api_key)])
async def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    """Standard JSON query endpoint with semantic cache."""
    state = request.app.state.app_state

    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    request_id = getattr(request.state, "request_id", "")

    # Check semantic cache
    if state.cache is not None:
        try:
            cached = await state.cache.get(body.query)
            if cached is not None:
                CACHE_HITS.inc()
                REQUEST_COUNT.labels(method="POST", endpoint="/query", status=200).inc()
                cached["cache_hit"] = True
                cached["request_id"] = request_id
                return QueryResponse(**cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    CACHE_MISSES.inc()

    # Run pipeline in thread pool (sync -> async)
    loop = asyncio.get_event_loop()
    start = time.time()
    result = await loop.run_in_executor(None, _run_pipeline, state.pipeline, body.query, body.top_k)
    latency_ms = (time.time() - start) * 1000
    QUERY_LATENCY.observe(latency_ms / 1000)

    timings = result.get("timings", {})
    RETRIEVAL_LATENCY.observe(
        (timings.get("retrieval_ms", 0) + timings.get("reranking_ms", 0)) / 1000
    )
    GENERATION_LATENCY.observe(timings.get("generation_ms", 0) / 1000)
    REQUEST_COUNT.labels(method="POST", endpoint="/query", status=200).inc()

    sources = _build_sources(result, body.include_sources)

    response = QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        latency_ms=round(latency_ms, 2),
        cache_hit=False,
        request_id=request_id,
    )

    # Fire-and-forget cache set
    if state.cache is not None:
        try:
            asyncio.create_task(state.cache.set(body.query, _response_to_cache_dict(response)))
        except Exception:
            pass

    return response


@router.post("/query/stream", dependencies=[Depends(verify_api_key)])
async def query_stream(body: QueryRequest, request: Request):
    """SSE streaming endpoint — tokens sent as Server-Sent Events."""
    state = request.app.state.app_state

    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    request_id = getattr(request.state, "request_id", "")

    async def event_generator():
        from transformers import TextIteratorStreamer

        pipeline = state.pipeline
        generator = pipeline.generator

        # Step 1: Retrieve + rerank (sync, in executor)
        loop = asyncio.get_event_loop()
        retrieval_result = await loop.run_in_executor(
            None, _run_pipeline, pipeline, body.query, body.top_k
        )

        # Step 2: Build prompt from retrieved chunks
        chunks = retrieval_result.get("reranked_chunks", retrieval_result.get("chunks", []))
        prompt = generator.build_prompt(body.query, chunks, max_chunks=body.top_k)

        # Step 3: Set up streamer
        streamer = TextIteratorStreamer(
            generator.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Step 4: Launch generation in background thread
        def generate_fn():
            generator._generate_text(prompt, streamer=streamer)

        thread = Thread(target=generate_fn)
        thread.start()

        # Step 5: Yield tokens as SSE events
        for token in streamer:
            if token:
                yield {"event": "token", "data": json.dumps({"token": token})}
            await asyncio.sleep(0)  # Yield control to event loop

        yield {"event": "done", "data": json.dumps({"request_id": request_id})}
        thread.join()

    return EventSourceResponse(
        event_generator(),
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/batch", dependencies=[Depends(verify_api_key)])
async def batch_query(body: BatchQueryRequest, request: Request) -> BatchQueryResponse:
    """Batch query endpoint. Sequential processing (VRAM constraint)."""
    state = request.app.state.app_state

    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    request_id = getattr(request.state, "request_id", "")

    loop = asyncio.get_event_loop()
    results = []
    total_start = time.time()

    for query_text in body.queries:
        start = time.time()
        result = await loop.run_in_executor(None, _run_pipeline, state.pipeline, query_text, 3)
        latency_ms = (time.time() - start) * 1000

        sources = _build_sources(result, body.include_sources)

        results.append(
            QueryResponse(
                answer=result.get("answer", ""),
                sources=sources,
                latency_ms=round(latency_ms, 2),
                cache_hit=False,
                request_id=request_id,
            )
        )

    total_latency_ms = (time.time() - total_start) * 1000

    return BatchQueryResponse(
        results=results,
        total_latency_ms=round(total_latency_ms, 2),
    )
