"""
Prometheus metric definitions.

All metrics prefixed with rag_ to avoid naming collisions.
Latency histogram buckets calibrated to Step 7 data.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- Request metrics ---
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval + rerank latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
)

# --- Cache metrics ---
CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Semantic cache misses")
CACHE_SIZE = Gauge("rag_cache_size", "Number of entries in semantic cache")

# --- Error metrics ---
ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total errors",
    ["error_type"],
)

# --- Circuit breaker ---
CIRCUIT_BREAKER_COST_EUR = Gauge(
    "rag_circuit_breaker_cumulative_cost_eur",
    "Cumulative EUR cost in current window",
)

# --- GPU ---
GPU_VRAM_USED_GB = Gauge(
    "rag_gpu_vram_used_gb",
    "GPU VRAM allocated in GB",
)
