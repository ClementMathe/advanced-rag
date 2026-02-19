"""
Configuration via Pydantic BaseSettings. Single source of truth for all env vars.

All fields are overridable at runtime via environment variables (no RAG_ prefix).
"""

from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Pipeline-level configuration."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_k_retrieve: int = 20
    top_k_rerank: int = 3
    max_new_tokens: int = 80
    temperature: float = 0.1
    load_in_4bit: bool = True

    model_config = {"env_prefix": "PIPELINE_"}


class CacheConfig(BaseSettings):
    """Semantic cache configuration."""

    redis_url: str = "redis://localhost:6379/0"
    similarity_threshold: float = 0.92
    ttl_seconds: int = 86400
    cache_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = {"env_prefix": "CACHE_"}


class AuthConfig(BaseSettings):
    """API key authentication configuration."""

    api_key: str = ""
    enabled: bool = False

    model_config = {"env_prefix": "AUTH_"}


class CircuitBreakerConfig(BaseSettings):
    """Cost-based circuit breaker configuration."""

    cost_limit_eur: float = 5.0
    cost_per_1k_input_tokens: float = 0.004
    cost_per_1k_output_tokens: float = 0.012
    window_seconds: int = 3600

    model_config = {"env_prefix": "CB_"}


class APIConfig(BaseSettings):
    """Server-level configuration."""

    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "API_"}


class Settings:
    """Aggregated settings from all config groups."""

    def __init__(self):
        self.pipeline = PipelineConfig()
        self.cache = CacheConfig()
        self.auth = AuthConfig()
        self.circuit_breaker = CircuitBreakerConfig()
        self.api = APIConfig()
