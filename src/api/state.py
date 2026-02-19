"""
Application state shared across requests.

Holds the pipeline, cache, circuit breaker, and runtime counters.
Attached to app.state during lifespan.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.pipeline import RAGPipeline


@dataclass
class AppState:
    """Mutable application state attached to FastAPI app."""

    pipeline: Optional["RAGPipeline"] = None
    cache: Optional[Any] = None
    circuit_breaker: Optional[Any] = None
    settings: Optional[Any] = None
    start_time: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None
