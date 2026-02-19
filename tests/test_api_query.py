"""
Tests for Phase 2: query routes, auth, streaming, middleware, logging.

All tests mock the pipeline — no GPU or real models needed.
"""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.config import AuthConfig, Settings
from src.api.state import AppState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_mock_pipeline():
    """Create a mocked RAGPipeline returning a realistic result dict."""
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
            {
                "content": "France is in Western Europe.",
                "doc_id": "doc2",
                "rerank_score": 0.72,
                "chunk_index": 1,
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


def make_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_app(pipeline=None, settings=None):
    if settings is None:
        settings = make_settings()
    app = create_app(settings=settings)
    state: AppState = app.state.app_state
    state.pipeline = pipeline
    return app


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestQueryEndpoint:
    async def test_query_returns_answer(self):
        pipeline = create_mock_pipeline()
        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital of France?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Paris is the capital of France."
        assert data["cache_hit"] is False
        assert data["latency_ms"] > 0
        assert len(data["sources"]) == 2
        assert data["sources"][0]["document_id"] == "doc1"

    async def test_query_without_sources(self):
        pipeline = create_mock_pipeline()
        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query", json={"query": "What is the capital?", "include_sources": False}
            )
        data = resp.json()
        assert data["sources"] == []

    async def test_query_too_short_returns_422(self):
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "hi"})
        assert resp.status_code == 422

    async def test_query_pipeline_not_loaded_returns_503(self):
        app = make_app(pipeline=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital of France?"})
        assert resp.status_code == 503

    async def test_query_calls_pipeline(self):
        pipeline = create_mock_pipeline()
        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/query", json={"query": "What is the capital of France?"})
        pipeline.query.assert_called_once()


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAuth:
    async def test_auth_disabled_no_key_needed(self):
        """When auth is disabled, queries work without API key."""
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital?"})
        assert resp.status_code == 200

    async def test_auth_enabled_valid_key(self):
        settings = make_settings()
        settings.auth = AuthConfig(enabled=True, api_key="test-secret-key")
        app = make_app(pipeline=create_mock_pipeline(), settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200

    async def test_auth_enabled_missing_key_returns_401(self):
        settings = make_settings()
        settings.auth = AuthConfig(enabled=True, api_key="test-secret-key")
        app = make_app(pipeline=create_mock_pipeline(), settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital?"})
        assert resp.status_code == 401

    async def test_auth_enabled_invalid_key_returns_401(self):
        settings = make_settings()
        settings.auth = AuthConfig(enabled=True, api_key="correct-key")
        app = make_app(pipeline=create_mock_pipeline(), settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    async def test_health_does_not_require_auth(self):
        """Health endpoints should not require auth even when enabled."""
        settings = make_settings()
        settings.auth = AuthConfig(enabled=True, api_key="secret")
        app = make_app(pipeline=create_mock_pipeline(), settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBatchEndpoint:
    async def test_batch_returns_multiple_results(self):
        pipeline = create_mock_pipeline()
        app = make_app(pipeline=pipeline)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/batch",
                json={"queries": ["Question 1?", "Question 2?", "Question 3?"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["total_latency_ms"] > 0

    async def test_batch_pipeline_not_loaded_returns_503(self):
        app = make_app(pipeline=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/batch", json={"queries": ["Question?"]})
        assert resp.status_code == 503

    async def test_batch_empty_returns_422(self):
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/batch", json={"queries": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMiddleware:
    async def test_request_id_generated(self):
        """When no X-Request-ID is sent, one is generated and returned."""
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"query": "What is the capital?"})
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    async def test_client_request_id_propagated(self):
        """Client-provided X-Request-ID is echoed back."""
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-Request-ID": "my-trace-123"},
            )
        assert resp.headers["x-request-id"] == "my-trace-123"

    async def test_request_id_in_response_body(self):
        """Request ID appears in the QueryResponse body."""
        app = make_app(pipeline=create_mock_pipeline())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={"query": "What is the capital?"},
                headers={"X-Request-ID": "trace-abc"},
            )
        data = resp.json()
        assert data["request_id"] == "trace-abc"

    async def test_request_id_on_health_endpoints(self):
        """Middleware applies to all endpoints, including health."""
        app = make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamingEndpoint:
    async def test_stream_pipeline_not_loaded_returns_503(self):
        app = make_app(pipeline=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query/stream", json={"query": "What is the capital?"})
        assert resp.status_code == 503

    async def test_stream_auth_required(self):
        settings = make_settings()
        settings.auth = AuthConfig(enabled=True, api_key="secret")
        app = make_app(pipeline=create_mock_pipeline(), settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query/stream", json={"query": "What is the capital?"})
        assert resp.status_code == 401
