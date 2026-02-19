"""
Tests for CostCircuitBreaker.
"""

import time

from src.api.circuit_breaker import CostCircuitBreaker


class TestCostCircuitBreaker:
    def test_starts_closed(self):
        cb = CostCircuitBreaker(cost_limit_eur=5.0)
        assert cb.is_open is False

    def test_stays_closed_under_limit(self):
        cb = CostCircuitBreaker(cost_limit_eur=5.0, cost_per_1k_input_tokens=0.004)
        cb.record_spend(input_tokens=1000, output_tokens=0)  # $0.004
        assert cb.is_open is False

    def test_opens_when_limit_exceeded(self):
        cb = CostCircuitBreaker(
            cost_limit_eur=0.01,
            cost_per_1k_input_tokens=0.004,
            cost_per_1k_output_tokens=0.012,
        )
        # 10 * 0.004 = 0.04 EUR > 0.01 limit
        for _ in range(10):
            cb.record_spend(input_tokens=1000, output_tokens=0)
        assert cb.is_open is True

    def test_record_spend_returns_cost(self):
        cb = CostCircuitBreaker(cost_per_1k_input_tokens=0.004, cost_per_1k_output_tokens=0.012)
        cost = cb.record_spend(input_tokens=1000, output_tokens=1000)
        assert abs(cost - 0.016) < 1e-9

    def test_reset_closes_breaker(self):
        cb = CostCircuitBreaker(cost_limit_eur=0.001)
        cb.record_spend(input_tokens=1000, output_tokens=0)
        assert cb.is_open is True
        cb.reset()
        assert cb.is_open is False

    def test_status_returns_dict(self):
        cb = CostCircuitBreaker(cost_limit_eur=5.0, window_seconds=3600)
        cb.record_spend(input_tokens=500, output_tokens=100)
        status = cb.status()
        assert status["state"] == "closed"
        assert status["cost_limit_eur"] == 5.0
        assert status["cumulative_cost_eur"] > 0
        assert status["window_seconds"] == 3600
        assert "window_remaining_seconds" in status

    def test_status_shows_open(self):
        cb = CostCircuitBreaker(cost_limit_eur=0.001)
        cb.record_spend(input_tokens=1000, output_tokens=0)
        status = cb.status()
        assert status["state"] == "open"

    def test_window_auto_reset(self):
        """Breaker resets automatically when window expires."""
        cb = CostCircuitBreaker(cost_limit_eur=0.001, window_seconds=1)
        cb.record_spend(input_tokens=1000, output_tokens=0)
        assert cb.is_open is True

        # Simulate window expiry
        cb._window_start = time.time() - 2
        assert cb.is_open is False

    def test_cumulative_cost_tracks_multiple_calls(self):
        cb = CostCircuitBreaker(cost_per_1k_input_tokens=0.004)
        cb.record_spend(input_tokens=1000, output_tokens=0)
        cb.record_spend(input_tokens=2000, output_tokens=0)
        status = cb.status()
        expected = 3 * 0.004
        assert abs(status["cumulative_cost_eur"] - expected) < 1e-9

    def test_zero_tokens_zero_cost(self):
        cb = CostCircuitBreaker()
        cost = cb.record_spend(input_tokens=0, output_tokens=0)
        assert cost == 0.0
