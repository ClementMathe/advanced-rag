"""
Tests for SemanticCache using fakeredis.

Uses a mock embedding model that returns predictable vectors.
"""

import numpy as np
import pytest

from src.api.cache import SemanticCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEmbedModel:
    """Deterministic embedding model for cache tests."""

    def __init__(self):
        self._vectors = {
            "capital of france": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "what is france capital": np.array([0.98, 0.2, 0.0], dtype=np.float32),
            "python programming language": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        # Normalize all vectors
        for k in self._vectors:
            v = self._vectors[k]
            self._vectors[k] = v / np.linalg.norm(v)

    def encode(self, texts):
        results = []
        for text in texts:
            text_lower = text.lower()
            best_key = None
            for key in self._vectors:
                if key in text_lower:
                    best_key = key
                    break
            if best_key:
                results.append(self._vectors[best_key])
            else:
                results.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        return results


def _make_cache():
    """Create a SemanticCache with fakeredis backend (sync helper)."""
    import fakeredis.aioredis

    cache = SemanticCache(
        redis_url="redis://fake",
        similarity_threshold=0.92,
        ttl_seconds=3600,
        embed_model=MockEmbedModel(),
    )
    cache._redis = fakeredis.aioredis.FakeRedis()
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSemanticCache:
    async def test_cache_miss_on_empty(self):
        cache = _make_cache()
        result = await cache.get("capital of France")
        assert result is None

    async def test_cache_set_and_exact_hit(self):
        cache = _make_cache()
        stored_result = {"answer": "Paris", "latency_ms": 100}
        await cache.set("capital of France", stored_result)

        hit = await cache.get("capital of France")
        assert hit is not None
        assert hit["answer"] == "Paris"

    async def test_cache_hit_semantic_match(self):
        """Paraphrased query should hit cache (similarity > threshold)."""
        cache = _make_cache()
        stored_result = {"answer": "Paris", "latency_ms": 100}
        await cache.set("capital of France", stored_result)

        # "what is france capital" has cosine ~0.98 with "capital of france"
        hit = await cache.get("What is France capital?")
        assert hit is not None
        assert hit["answer"] == "Paris"

    async def test_cache_miss_unrelated_query(self):
        """Unrelated query should not hit cache."""
        cache = _make_cache()
        stored_result = {"answer": "Paris", "latency_ms": 100}
        await cache.set("capital of France", stored_result)

        # "python programming language" is orthogonal
        hit = await cache.get("python programming language")
        assert hit is None

    async def test_cache_clear(self):
        cache = _make_cache()
        await cache.set("capital of France", {"answer": "Paris"})
        await cache.set("python programming language", {"answer": "Python"})

        stats = await cache.stats()
        assert stats["total_entries"] == 2

        await cache.clear()

        stats = await cache.stats()
        assert stats["total_entries"] == 0

    async def test_ping_when_connected(self):
        cache = _make_cache()
        assert await cache.ping() is True

    async def test_ping_when_disconnected(self):
        cache = SemanticCache(embed_model=MockEmbedModel())
        assert await cache.ping() is False

    async def test_stats_returns_config(self):
        cache = _make_cache()
        stats = await cache.stats()
        assert stats["similarity_threshold"] == 0.92
        assert stats["ttl_seconds"] == 3600

    async def test_set_returns_key(self):
        cache = _make_cache()
        key = await cache.set("capital of France", {"answer": "Paris"})
        assert key.startswith("rag:cache:")

    async def test_get_without_embed_model(self):
        """Cache without embed model returns None."""
        import fakeredis.aioredis

        cache = SemanticCache(embed_model=None)
        cache._redis = fakeredis.aioredis.FakeRedis()
        result = await cache.get("test query")
        assert result is None

    async def test_set_without_embed_model(self):
        """Cache without embed model returns empty string."""
        import fakeredis.aioredis

        cache = SemanticCache(embed_model=None)
        cache._redis = fakeredis.aioredis.FakeRedis()
        key = await cache.set("test", {"answer": "test"})
        assert key == ""

    async def test_multiple_entries_best_match(self):
        """With multiple cache entries, returns the best matching one."""
        cache = _make_cache()
        await cache.set("capital of France", {"answer": "Paris"})
        await cache.set("python programming language", {"answer": "Python"})

        hit = await cache.get("capital of France")
        assert hit["answer"] == "Paris"
