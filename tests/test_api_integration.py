"""
Integration tests — full middleware stack with auth + cache + circuit breaker + metrics.

All tests mock the pipeline and use fakeredis. No GPU needed.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.cache import SemanticCache
from src.api.circuit_breaker import CostCircuitBreaker
from src.api.config import AuthConfig, Settings
from src.api.state import AppState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_mock_pipeline():
    """Mock pipeline returning a realistic result dict."""
    pipeline = MagicMock()
    pipeline.query.return_value = {
        "query": "test query",
        "answer": "Paris is the capital of France.",
        "reranked_chunks": [
            {
                "content": "Paris is the capital and largest city of France.",
                "doc_id": "doc1",
                "rerank_score": 0.95,
                "chunk_index": 0,
            },
        ],
        "timings": {
            "retrieval_ms": 150.0,
            "reranking_ms": 50.0,
            "generation_ms": 3000.0,
        },
        "total_time_ms": 3200.0,
    }
    pipeline.generator = MagicMock()
    pipeline.generator.tokenizer = MagicMock()
    pipeline.generator.build_prompt.return_value = "formatted prompt"
    pipeline.k_retrieve = 20
    pipeline.k_rerank = 3
    return pipeline


class MockEmbedModel:
    """Simple mock embed model for cache integration tests."""

    def encode(self, texts):
        results = []
        for text in texts:
            # Deterministic hash-based vector
            h = hash(text) % 1000 / 1000.0
            vec = np.array([h, 1 - h, 0.5], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return results


def make_full_app(
    auth_enabled=False,
    api_key="test-key",  # pragma: allowlist secret
    with_cache=True,
    with_cb=True,
):
    """Create app with all components wired together."""
    import fakeredis.aioredis

    settings = Settings()
    if auth_enabled:
        settings.auth = AuthConfig(enabled=True, api_key=api_key)

    app = create_app(settings=settings)
    state: AppState = app.state.app_state
    state.pipeline = create_mock_pipeline()

    if with_cache:
        cache = SemanticCache(
            similarity_threshold=0.92,
            ttl_seconds=3600,
            embed_model=MockEmbedModel(),
        )
        cache._redis = fakeredis.aioredis.FakeRedis()
        state.cache = cache

    if with_cb:
        state.circuit_breaker = CostCircuitBreaker(cost_limit_eur=5.0)

    return app


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFullQueryFlow:
    async def test_cache_miss_then_pipeline(self):
        """First query: cache miss → pipeline runs → response returned."""
        app = make_full_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital of France?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Paris is the capital of France."
        assert data["cache_hit"] is False

    async def test_second_identical_query_hits_cache(self):
        """Second identical query hits cache — pipeline NOT called again."""
        app = make_full_app()
        transport = ASGITransport(app=app)
        pipeline = app.state.app_state.pipeline

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First query: cache miss
            resp1 = await client.post("/query", json={"query": "What is the capital of France?"})
            assert resp1.json()["cache_hit"] is False
            assert pipeline.query.call_count == 1

            # Need to wait briefly for fire-and-forget cache.set()
            import asyncio

            await asyncio.sleep(0.1)

            # Second query: cache hit
            resp2 = await client.post("/query", json={"query": "What is the capital of France?"})
            assert resp2.json()["cache_hit"] is True
            # Pipeline should NOT have been called again
            assert pipeline.query.call_count == 1


@pytest.mark.asyncio
class TestAuthIntegration:
    async def test_auth_blocks_query_allows_health(self):
        """Auth blocks /query without key but allows /health."""
        app = make_full_app(auth_enabled=True, api_key="my-secret")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Health: no auth needed
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200

            # Query without key: 401
            query_resp = await client.post("/query", json={"query": "What is the capital?"})
            assert query_resp.status_code == 401

            # Query with valid key: 200
            auth_resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-API-Key": "my-secret"},
            )
            assert auth_resp.status_code == 200


@pytest.mark.asyncio
class TestRequestIDPropagation:
    async def test_request_id_in_header_and_body(self):
        """X-Request-ID appears in both response header and body."""
        app = make_full_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-Request-ID": "integration-trace-42"},
            )
        assert resp.headers["x-request-id"] == "integration-trace-42"
        assert resp.json()["request_id"] == "integration-trace-42"

    async def test_auto_generated_request_id(self):
        """When no X-Request-ID sent, one is generated."""
        app = make_full_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital?"})
        assert len(resp.headers["x-request-id"]) > 0
        assert len(resp.json()["request_id"]) > 0


@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    async def test_circuit_breaker_visible_in_stats(self):
        """Circuit breaker status appears in /health/stats."""
        app = make_full_app(with_cb=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert data["circuit_breaker"]["state"] == "closed"
        assert data["circuit_breaker"]["cost_limit_eur"] == 5.0

    async def test_circuit_breaker_open_in_stats(self):
        """When breaker is open, stats reflect it."""
        app = make_full_app(with_cb=True)
        # Force breaker open
        app.state.app_state.circuit_breaker._cumulative_cost_eur = 10.0
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert data["circuit_breaker"]["state"] == "open"


@pytest.mark.asyncio
class TestCacheIntegration:
    async def test_cache_stats_in_health(self):
        """Cache stats appear in /health/stats."""
        app = make_full_app(with_cache=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/stats")
        data = resp.json()
        assert "total_entries" in data["cache"]
        assert data["cache"]["similarity_threshold"] == 0.92

    async def test_cache_connected_in_health(self):
        """Health endpoint shows cache_connected=True when cache is up."""
        app = make_full_app(with_cache=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["cache_connected"] is True


@pytest.mark.asyncio
class TestMetricsEndpoint:
    async def test_metrics_endpoint_exists(self):
        """GET /metrics returns Prometheus metrics."""
        app = make_full_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics/", follow_redirects=True)
        assert resp.status_code == 200
        text = resp.text
        # Should contain Prometheus format markers
        assert "# HELP" in text or "# TYPE" in text
