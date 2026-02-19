"""
Cost-based circuit breaker for cloud LLM fallback.

Tracks cumulative EUR spend in memory. Opens when the configured
budget is exhausted within the time window.

States:
- CLOSED: normal operation, cloud API calls allowed
- OPEN: budget exceeded, cloud API calls rejected
"""

import threading
import time


class CostCircuitBreaker:
    """Thread-safe cost-based circuit breaker."""

    def __init__(
        self,
        cost_limit_eur: float = 5.0,
        cost_per_1k_input_tokens: float = 0.004,
        cost_per_1k_output_tokens: float = 0.012,
        window_seconds: int = 3600,
    ):
        self.cost_limit_eur = cost_limit_eur
        self.cost_per_1k_input = cost_per_1k_input_tokens
        self.cost_per_1k_output = cost_per_1k_output_tokens
        self.window_seconds = window_seconds

        self._lock = threading.Lock()
        self._cumulative_cost_eur = 0.0
        self._window_start = time.time()

    @property
    def is_open(self) -> bool:
        """True if circuit breaker is open (budget exhausted)."""
        with self._lock:
            self._maybe_reset_window()
            return self._cumulative_cost_eur >= self.cost_limit_eur

    def record_spend(self, input_tokens: int, output_tokens: int) -> float:
        """Record token usage. Returns EUR cost of this call."""
        cost = (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output

        with self._lock:
            self._maybe_reset_window()
            self._cumulative_cost_eur += cost

        return cost

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._cumulative_cost_eur = 0.0
            self._window_start = time.time()

    def status(self) -> dict:
        """Return current status for /health/stats."""
        with self._lock:
            self._maybe_reset_window()
            return {
                "state": "open" if self._cumulative_cost_eur >= self.cost_limit_eur else "closed",
                "cumulative_cost_eur": round(self._cumulative_cost_eur, 6),
                "cost_limit_eur": self.cost_limit_eur,
                "window_seconds": self.window_seconds,
                "window_remaining_seconds": round(
                    max(0, self.window_seconds - (time.time() - self._window_start)), 1
                ),
            }

    def _maybe_reset_window(self):
        """Reset if the time window has elapsed (called under lock)."""
        if time.time() - self._window_start >= self.window_seconds:
            self._cumulative_cost_eur = 0.0
            self._window_start = time.time()
