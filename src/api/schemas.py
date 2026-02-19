"""
Pydantic request/response models for the RAG API.

Field examples populate OpenAPI docs at /docs.
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Single query request."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=512,
        examples=["When did Beyonce become famous?"],
    )
    top_k: int = Field(default=3, ge=1, le=20, examples=[3])
    include_sources: bool = Field(default=True)


class Source(BaseModel):
    """A retrieved source chunk."""

    content: str
    score: float
    document_id: str


class QueryResponse(BaseModel):
    """Response to a single query."""

    answer: str
    sources: List[Source] = []
    latency_ms: float
    cache_hit: bool = False
    request_id: str = ""


class BatchQueryRequest(BaseModel):
    """Batch of queries (max 50)."""

    queries: List[str] = Field(..., min_length=1, max_length=50)
    include_sources: bool = Field(default=False)


class BatchQueryResponse(BaseModel):
    """Response to a batch of queries."""

    results: List[QueryResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Liveness health check response."""

    status: str
    pipeline_loaded: bool
    cache_connected: bool
    model_name: str = ""
    uptime_seconds: float = 0.0


class GPUHealthResponse(BaseModel):
    """GPU health check response."""

    status: str
    model_loaded_in_vram: bool = False
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_utilization_pct: float = 0.0
    load_in_4bit: bool = False


class StatsResponse(BaseModel):
    """Extended diagnostics for dashboards."""

    uptime_seconds: float
    cache: Dict = {}
    circuit_breaker: Dict = {}
    pipeline: Dict = {}
