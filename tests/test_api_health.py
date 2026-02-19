"""
Tests for Phase 1: config, schemas, state, app factory, health endpoints.

All tests mock the pipeline and Redis — no GPU or external services needed.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.config import (
    APIConfig,
    AuthConfig,
    CacheConfig,
    CircuitBreakerConfig,
    PipelineConfig,
    Settings,
)
from src.api.schemas import (
    GPUHealthResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
)
from src.api.state import AppState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> Settings:
    """Create Settings with defaults suitable for testing."""
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_app(pipeline=None, cache=None, circuit_breaker=None, settings=None):
    """Create a test app with optional mocked components."""
    if settings is None:
        settings = make_settings()
    app = create_app(settings=settings)
    state: AppState = app.state.app_state
    state.pipeline = pipeline
    state.cache = cache
    state.circuit_breaker = circuit_breaker
    return app


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_pipeline_config(self):
        cfg = PipelineConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert cfg.top_k_rerank == 3
        assert cfg.load_in_4bit is True
        assert cfg.max_new_tokens == 80

    def test_default_cache_config(self):
        cfg = CacheConfig()
        assert cfg.similarity_threshold == 0.92
        assert cfg.ttl_seconds == 86400

    def test_default_auth_config(self):
        cfg = AuthConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""

    def test_default_circuit_breaker_config(self):
        cfg = CircuitBreakerConfig()
        assert cfg.cost_limit_eur == 5.0

    def test_default_api_config(self):
        cfg = APIConfig()
        assert cfg.port == 8000
        assert cfg.log_level == "INFO"

    def test_settings_aggregates_all(self):
        s = Settings()
        assert isinstance(s.pipeline, PipelineConfig)
        assert isinstance(s.cache, CacheConfig)
        assert isinstance(s.auth, AuthConfig)
        assert isinstance(s.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(s.api, APIConfig)

    def test_pipeline_env_override(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_TOP_K_RERANK", "7")
        cfg = PipelineConfig()
        assert cfg.top_k_rerank == 7

    def test_cache_env_override(self, monkeypatch):
        monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
        cfg = CacheConfig()
        assert cfg.similarity_threshold == 0.85

    def test_auth_env_override(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("AUTH_API_KEY", "secret123")
        cfg = AuthConfig()
        assert cfg.enabled is True
        assert cfg.api_key == "secret123"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_query_request_valid(self):
        req = QueryRequest(query="What is the capital of France?")
        assert req.query == "What is the capital of France?"
        assert req.top_k == 3
        assert req.include_sources is True

    def test_query_request_too_short(self):
        with pytest.raises(Exception):  # noqa: B017
            QueryRequest(query="hi")

    def test_query_response_round_trip(self):
        resp = QueryResponse(
            answer="Paris",
            sources=[Source(content="Paris is...", score=0.95, document_id="d1")],
            latency_ms=123.4,
            cache_hit=False,
            request_id="abc123",
        )
        data = resp.model_dump()
        assert data["answer"] == "Paris"
        assert len(data["sources"]) == 1
        assert data["cache_hit"] is False

    def test_health_response(self):
        resp = HealthResponse(
            status="ok",
            pipeline_loaded=True,
            cache_connected=False,
            model_name="test",
            uptime_seconds=42.0,
        )
        assert resp.status == "ok"

    def test_gpu_health_response(self):
        resp = GPUHealthResponse(
            status="ok",
            model_loaded_in_vram=True,
            vram_used_gb=2.2,
            vram_total_gb=6.4,
            vram_utilization_pct=34.4,
        )
        assert resp.vram_utilization_pct == 34.4


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestAppState:
    def test_default_state(self):
        state = AppState()
        assert state.pipeline is None
        assert state.is_ready is False
        assert state.uptime_seconds >= 0

    def test_is_ready_with_pipeline(self):
        state = AppState(pipeline=MagicMock())
        assert state.is_ready is True

    def test_uptime_increases(self):
        state = AppState(start_time=time.time() - 10)
        assert state.uptime_seconds >= 9


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthEndpoints:
    async def test_health_no_pipeline(self):
        app = make_app(pipeline=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["pipeline_loaded"] is False
        assert data["cache_connected"] is False

    async def test_health_with_pipeline(self):
        app = make_app(pipeline=MagicMock())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["pipeline_loaded"] is True

    async def test_health_with_cache_connected(self):
        cache = AsyncMock()
        cache.ping.return_value = True
        app = make_app(pipeline=MagicMock(), cache=cache)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["cache_connected"] is True

    async def test_health_cache_ping_fails(self):
        cache = AsyncMock()
        cache.ping.side_effect = Exception("connection refused")
        app = make_app(cache=cache)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["cache_connected"] is False

    async def test_ready_when_pipeline_loaded(self):
        app = make_app(pipeline=MagicMock())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")
        assert resp.json()["status"] == "ready"

    async def test_not_ready_during_startup(self):
        app = make_app(pipeline=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")
        assert resp.json()["status"] == "not_ready"

    async def test_gpu_health_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2.2e9
        props = MagicMock()
        props.total_memory = 6.4e9
        mock_torch.cuda.get_device_properties.return_value = props

        # Mock pipeline with model on CUDA
        pipeline = MagicMock()
        param = MagicMock()
        param.device = MagicMock(type="cuda")
        pipeline.generator.model.parameters.return_value = iter([param])

        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        with patch("src.api.routes.health.torch", mock_torch):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/gpu")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded_in_vram"] is True
        assert data["vram_used_gb"] == 2.2
        assert data["vram_total_gb"] == 6.4

    async def test_gpu_health_no_cuda(self):
        """When CUDA is not available, return no_gpu status."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        app = make_app()
        transport = ASGITransport(app=app)
        with patch("src.api.routes.health.torch", mock_torch):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/gpu")
        data = resp.json()
        assert data["status"] == "no_gpu"
        assert data["model_loaded_in_vram"] is False

    async def test_stats_no_components(self):
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert data["uptime_seconds"] >= 0
        assert data["cache"] == {}
        assert data["circuit_breaker"] == {}
        assert data["pipeline"] == {}

    async def test_stats_with_pipeline(self):
        pipeline = MagicMock()
        pipeline.k_retrieve = 20
        pipeline.k_rerank = 3
        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert data["pipeline"]["top_k_retrieve"] == 20
        assert data["pipeline"]["top_k_rerank"] == 3

    async def test_stats_with_cache_and_cb(self):
        cache = AsyncMock()
        cache.stats.return_value = {"total_entries": 42, "ttl_seconds": 86400}
        cb = MagicMock()
        cb.status.return_value = {"state": "closed", "cumulative_cost_eur": 0.5}

        app = make_app(pipeline=MagicMock(), cache=cache, circuit_breaker=cb)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert data["cache"]["total_entries"] == 42
        assert data["circuit_breaker"]["state"] == "closed"


# ---------------------------------------------------------------------------
# App factory tests
# ---------------------------------------------------------------------------


class TestAppFactory:
    def test_create_app_default_settings(self):
        app = create_app()
        assert app.state.app_state is not None
        assert app.state.app_state.settings is not None

    def test_create_app_custom_settings(self):
        settings = make_settings()
        app = create_app(settings=settings)
        assert app.state.app_state.settings is settings

    def test_health_router_registered(self):
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/health/ready" in routes
        assert "/health/gpu" in routes
        assert "/health/stats" in routes
