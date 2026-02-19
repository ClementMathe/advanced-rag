"""
Redis-backed semantic query cache.

Two queries are equivalent if their cosine similarity
(using all-MiniLM-L6-v2) exceeds the configurable threshold.
O(N) scan — acceptable for prototype (<10K entries).
"""

import json
import time
import uuid

import numpy as np
import redis.asyncio as aioredis


class SemanticCache:
    """Semantic cache backed by Redis."""

    CACHE_PREFIX = "rag:cache:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400,
        embed_model=None,
    ):
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embed_model = embed_model
        self._redis = None

    async def connect(self):
        """Open Redis connection pool."""
        self._redis = aioredis.from_url(self.redis_url, decode_responses=False)

    async def disconnect(self):
        """Close Redis connection pool."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def ping(self) -> bool:
        """Check Redis reachability."""
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    def _encode(self, text: str) -> np.ndarray:
        """Compute L2-normalized embedding for a query."""
        embedding = self.embed_model.encode([text])[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    async def get(self, query: str):
        """Check cache for semantically similar query. Returns dict or None."""
        if self._redis is None or self.embed_model is None:
            return None

        query_embedding = self._encode(query)

        # Scan all cache keys
        best_match = None
        best_similarity = -1.0

        async for key in self._redis.scan_iter(match=f"{self.CACHE_PREFIX}*"):
            raw = await self._redis.get(key)
            if raw is None:
                continue
            entry = json.loads(raw)
            cached_embedding = np.array(entry["embedding"], dtype=np.float32)

            similarity = float(np.dot(query_embedding, cached_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_similarity >= self.similarity_threshold and best_match:
            return best_match["result"]

        return None

    async def set(self, query: str, result: dict) -> str:
        """Store query result with embedding. Returns Redis key."""
        if self._redis is None or self.embed_model is None:
            return ""

        embedding = self._encode(query)
        key = f"{self.CACHE_PREFIX}{uuid.uuid4().hex}"

        entry = {
            "embedding": embedding.tolist(),
            "query": query,
            "result": result,
            "created_at": time.time(),
        }

        await self._redis.set(key, json.dumps(entry), ex=self.ttl_seconds)
        return key

    async def clear(self):
        """Delete all cache entries."""
        if self._redis is None:
            return
        async for key in self._redis.scan_iter(match=f"{self.CACHE_PREFIX}*"):
            await self._redis.delete(key)

    async def stats(self) -> dict:
        """Return cache statistics."""
        total_entries = 0
        if self._redis:
            async for _ in self._redis.scan_iter(match=f"{self.CACHE_PREFIX}*"):
                total_entries += 1

        return {
            "total_entries": total_entries,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
        }
