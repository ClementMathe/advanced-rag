"""
Health check endpoints.

- GET /health       — liveness probe
- GET /health/ready — readiness probe (pipeline loaded?)
- GET /health/gpu   — VRAM + model load check
- GET /health/stats — cache + circuit breaker diagnostics
"""

import torch
from fastapi import APIRouter, Request

from src.api.metrics import GPU_VRAM_USED_GB
from src.api.schemas import GPUHealthResponse, HealthResponse, StatsResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(request: Request):
    """Liveness probe. Returns 200 if process is alive."""
    state = request.app.state.app_state
    cache_connected = False
    if state.cache is not None:
        try:
            cache_connected = await state.cache.ping()
        except Exception:
            cache_connected = False

    model_name = ""
    if state.settings and state.settings.pipeline:
        model_name = state.settings.pipeline.model_name

    return HealthResponse(
        status="ok",
        pipeline_loaded=state.pipeline is not None,
        cache_connected=cache_connected,
        model_name=model_name,
        uptime_seconds=state.uptime_seconds,
    )


@router.get("/ready")
async def readiness(request: Request):
    """Readiness probe. Returns not_ready during model loading."""
    state = request.app.state.app_state
    if state.is_ready:
        return {"status": "ready"}
    return {"status": "not_ready"}


@router.get("/gpu", response_model=GPUHealthResponse)
async def gpu_health(request: Request):
    """Deep GPU check: VRAM usage and model device verification."""
    state = request.app.state.app_state

    if not torch.cuda.is_available():
        return GPUHealthResponse(status="no_gpu", model_loaded_in_vram=False)

    vram_used_gb = torch.cuda.memory_allocated() / 1e9
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_utilization_pct = round((vram_used_gb / vram_total_gb) * 100, 1)

    model_in_vram = False
    if state.pipeline and state.pipeline.generator:
        try:
            device = next(state.pipeline.generator.model.parameters()).device
            model_in_vram = device.type == "cuda"
        except Exception:
            pass

    status = "ok" if model_in_vram else "degraded"

    # Update Prometheus gauge so Grafana reflects current VRAM
    GPU_VRAM_USED_GB.set(round(vram_used_gb, 3))

    return GPUHealthResponse(
        status=status,
        model_loaded_in_vram=model_in_vram,
        vram_used_gb=round(vram_used_gb, 3),
        vram_total_gb=round(vram_total_gb, 3),
        vram_utilization_pct=vram_utilization_pct,
        load_in_4bit=state.settings.pipeline.load_in_4bit if state.settings else False,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request):
    """Extended diagnostics for dashboards."""
    state = request.app.state.app_state

    cache_stats = {}
    if state.cache is not None:
        try:
            cache_stats = await state.cache.stats()
        except Exception:
            cache_stats = {"error": "unavailable"}

    cb_stats = {}
    if state.circuit_breaker is not None:
        try:
            cb_stats = state.circuit_breaker.status()
        except Exception:
            cb_stats = {"error": "unavailable"}

    pipeline_stats = {}
    if state.pipeline is not None:
        pipeline_stats = {
            "top_k_retrieve": state.pipeline.k_retrieve,
            "top_k_rerank": state.pipeline.k_rerank,
            "load_in_4bit": state.settings.pipeline.load_in_4bit if state.settings else False,
        }

    return StatsResponse(
        uptime_seconds=state.uptime_seconds,
        cache=cache_stats,
        circuit_breaker=cb_stats,
        pipeline=pipeline_stats,
    )
