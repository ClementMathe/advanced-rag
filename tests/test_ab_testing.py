"""
Unit tests for A/B testing infrastructure.

Tests:
- MetricComparison / ABTestResult: data classes, serialization
- ABTestRunner: statistical comparison, winner determination, edge cases
- Bootstrap CI: correctness on synthetic data
- Power analysis: minimum sample size estimates

All tests use synthetic per-query result dicts — no real models loaded.
"""

import numpy as np
import pytest

from src.evaluation.ab_testing import (
    ABTestResult,
    ABTestRunner,
    MetricComparison,
)

# ============================================================
# Helpers
# ============================================================


def make_results(
    n: int, f1: float = 0.5, faithfulness: float = 0.8, noise: float = 0.05, seed: int = 42
):
    """Generate n synthetic per-query result dicts with controlled metrics."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        results.append(
            {
                "query": "test query",
                "prediction": "test answer",
                "ground_truth": "test answer",
                "f1": float(np.clip(f1 + rng.normal(0, noise), 0, 1)),
                "exact_match": 1.0 if f1 > 0.9 else 0.0,
                "rouge_l": float(np.clip(f1 * 0.9 + rng.normal(0, noise), 0, 1)),
                "faithfulness": float(np.clip(faithfulness + rng.normal(0, noise), 0, 1)),
            }
        )
    return results


# ============================================================
# TestMetricComparison
# ============================================================


class TestMetricComparison:
    def test_creation(self):
        comp = MetricComparison(
            metric_name="f1",
            champion_mean=0.5,
            challenger_mean=0.6,
            difference=0.1,
            p_value=0.03,
            ci_lower=0.02,
            ci_upper=0.18,
            effect_size=0.5,
            significant=True,
        )
        assert comp.metric_name == "f1"
        assert comp.significant is True

    def test_to_dict(self):
        comp = MetricComparison(
            metric_name="f1",
            champion_mean=0.5,
            challenger_mean=0.6,
            difference=0.1,
            p_value=0.03,
            ci_lower=0.02,
            ci_upper=0.18,
            effect_size=0.5,
            significant=True,
        )
        d = comp.to_dict()
        assert d["metric_name"] == "f1"
        assert d["significant"] is True
        assert "p_value" in d


# ============================================================
# TestABTestResult
# ============================================================


class TestABTestResult:
    @pytest.fixture
    def sample_result(self):
        comp = MetricComparison(
            metric_name="f1",
            champion_mean=0.5,
            challenger_mean=0.6,
            difference=0.1,
            p_value=0.03,
            ci_lower=0.02,
            ci_upper=0.18,
            effect_size=0.5,
            significant=True,
        )
        return ABTestResult(
            champion_name="linear",
            challenger_name="agentic",
            n_queries=100,
            metric_comparisons={"f1": comp},
            winner="challenger",
            recommendation="Deploy challenger.",
        )

    def test_summary_contains_key_info(self, sample_result):
        summary = sample_result.summary()
        assert "linear" in summary
        assert "agentic" in summary
        assert "100 queries" in summary
        assert "challenger" in summary

    def test_to_dict_keys(self, sample_result):
        d = sample_result.to_dict()
        assert "champion_name" in d
        assert "challenger_name" in d
        assert "metric_comparisons" in d
        assert "winner" in d
        assert isinstance(d["metric_comparisons"]["f1"], dict)


# ============================================================
# TestBootstrapCI
# ============================================================


class TestBootstrapCI:
    @pytest.fixture
    def runner(self):
        return ABTestRunner(n_bootstrap=500)

    def test_identical_data_ci_contains_zero(self, runner):
        diffs = np.zeros(100)
        ci_lower, ci_upper = runner._bootstrap_ci(diffs)
        assert ci_lower == pytest.approx(0.0, abs=0.01)
        assert ci_upper == pytest.approx(0.0, abs=0.01)

    def test_shifted_data_ci_excludes_zero(self, runner):
        rng = np.random.default_rng(42)
        diffs = rng.normal(0.5, 0.1, size=200)
        ci_lower, ci_upper = runner._bootstrap_ci(diffs)
        assert ci_lower > 0
        assert ci_upper > ci_lower

    def test_negative_shift_ci_below_zero(self, runner):
        rng = np.random.default_rng(42)
        diffs = rng.normal(-0.3, 0.05, size=200)
        ci_lower, ci_upper = runner._bootstrap_ci(diffs)
        assert ci_upper < 0


# ============================================================
# TestCompareSingleMetric
# ============================================================


class TestCompareSingleMetric:
    @pytest.fixture
    def runner(self):
        return ABTestRunner(n_bootstrap=500)

    def test_identical_not_significant(self, runner):
        vals = np.array([0.5] * 50)
        comp = runner._compare_single_metric(vals, vals, "f1")
        assert not comp.significant
        assert comp.difference == pytest.approx(0.0, abs=1e-10)

    def test_large_difference_significant(self, runner):
        rng = np.random.default_rng(42)
        champ = rng.normal(0.4, 0.05, size=100)
        chall = rng.normal(0.7, 0.05, size=100)
        comp = runner._compare_single_metric(champ, chall, "f1")
        assert comp.significant
        assert comp.difference > 0.2
        assert comp.effect_size > 1.0  # large effect

    def test_effect_size_small(self, runner):
        rng = np.random.default_rng(42)
        champ = rng.normal(0.5, 0.1, size=100)
        chall = rng.normal(0.52, 0.1, size=100)
        comp = runner._compare_single_metric(champ, chall, "f1")
        assert abs(comp.effect_size) < 0.5  # small effect

    def test_ci_bounds_order(self, runner):
        rng = np.random.default_rng(42)
        champ = rng.normal(0.5, 0.1, size=50)
        chall = rng.normal(0.6, 0.1, size=50)
        comp = runner._compare_single_metric(champ, chall, "f1")
        assert comp.ci_lower < comp.ci_upper

    def test_p_value_range(self, runner):
        rng = np.random.default_rng(42)
        champ = rng.normal(0.5, 0.1, size=50)
        chall = rng.normal(0.6, 0.1, size=50)
        comp = runner._compare_single_metric(champ, chall, "f1")
        assert 0.0 <= comp.p_value <= 1.0


# ============================================================
# TestCompare
# ============================================================


class TestCompare:
    @pytest.fixture
    def runner(self):
        return ABTestRunner(
            primary_metric="f1",
            guard_metrics=["faithfulness"],
            n_bootstrap=200,
        )

    def test_identical_results_no_winner(self, runner):
        results = make_results(50, f1=0.5, faithfulness=0.8, noise=0.0)
        result = runner.compare(results, results)
        assert result.winner == "no_significant_difference"

    def test_better_challenger_wins(self, runner):
        champ = make_results(100, f1=0.4, faithfulness=0.8, noise=0.03, seed=1)
        chall = make_results(100, f1=0.7, faithfulness=0.8, noise=0.03, seed=2)
        result = runner.compare(champ, chall)
        assert result.winner == "challenger"
        assert "f1" in result.metric_comparisons

    def test_worse_challenger_loses(self, runner):
        champ = make_results(100, f1=0.7, faithfulness=0.8, noise=0.03, seed=1)
        chall = make_results(100, f1=0.4, faithfulness=0.8, noise=0.03, seed=2)
        result = runner.compare(champ, chall)
        assert result.winner == "champion"

    def test_guard_metric_regression_blocks_challenger(self, runner):
        champ = make_results(100, f1=0.4, faithfulness=0.9, noise=0.02, seed=1)
        chall = make_results(100, f1=0.7, faithfulness=0.3, noise=0.02, seed=2)
        result = runner.compare(champ, chall)
        assert result.winner == "champion"
        assert (
            "regressed" in result.recommendation.lower() or "guard" in result.recommendation.lower()
        )

    def test_explicit_metrics_list(self, runner):
        champ = make_results(50, seed=1)
        chall = make_results(50, seed=2)
        result = runner.compare(champ, chall, metrics=["f1", "rouge_l"])
        assert set(result.metric_comparisons.keys()) == {"f1", "rouge_l"}

    def test_auto_detect_metrics(self, runner):
        champ = make_results(30, seed=1)
        chall = make_results(30, seed=2)
        result = runner.compare(champ, chall)
        # Should detect f1, exact_match, rouge_l, faithfulness
        assert "f1" in result.metric_comparisons
        assert "faithfulness" in result.metric_comparisons

    def test_different_lengths_raises(self, runner):
        champ = make_results(10)
        chall = make_results(20)
        with pytest.raises(ValueError, match="same length"):
            runner.compare(champ, chall)

    def test_empty_results_raises(self, runner):
        with pytest.raises(ValueError, match="not be empty"):
            runner.compare([], [])

    def test_names_in_result(self, runner):
        champ = make_results(30, seed=1)
        chall = make_results(30, seed=2)
        result = runner.compare(champ, chall, champion_name="linear", challenger_name="agentic")
        assert result.champion_name == "linear"
        assert result.challenger_name == "agentic"


# ============================================================
# TestDetermineWinner
# ============================================================


class TestDetermineWinner:
    @pytest.fixture
    def runner(self):
        return ABTestRunner(primary_metric="f1", guard_metrics=["faithfulness"])

    def _make_comp(self, metric, diff, p, significant):
        return MetricComparison(
            metric_name=metric,
            champion_mean=0.5,
            challenger_mean=0.5 + diff,
            difference=diff,
            p_value=p,
            ci_lower=diff - 0.05,
            ci_upper=diff + 0.05,
            effect_size=diff / 0.1,
            significant=significant,
        )

    def test_challenger_wins(self, runner):
        comparisons = {
            "f1": self._make_comp("f1", 0.1, 0.01, True),
            "faithfulness": self._make_comp("faithfulness", 0.02, 0.3, False),
        }
        winner, _ = runner._determine_winner(comparisons)
        assert winner == "challenger"

    def test_champion_wins_primary_worse(self, runner):
        comparisons = {
            "f1": self._make_comp("f1", -0.1, 0.01, True),
            "faithfulness": self._make_comp("faithfulness", 0.02, 0.3, False),
        }
        winner, _ = runner._determine_winner(comparisons)
        assert winner == "champion"

    def test_tie_not_significant(self, runner):
        comparisons = {
            "f1": self._make_comp("f1", 0.01, 0.5, False),
            "faithfulness": self._make_comp("faithfulness", 0.01, 0.6, False),
        }
        winner, _ = runner._determine_winner(comparisons)
        assert winner == "no_significant_difference"

    def test_guard_veto(self, runner):
        comparisons = {
            "f1": self._make_comp("f1", 0.15, 0.001, True),
            "faithfulness": self._make_comp("faithfulness", -0.2, 0.001, True),
        }
        winner, rec = runner._determine_winner(comparisons)
        assert winner == "champion"
        assert "faithfulness" in rec


# ============================================================
# TestMinimumSampleSize
# ============================================================


class TestMinimumSampleSize:
    def test_medium_effect(self):
        n = ABTestRunner.minimum_sample_size(effect_size=0.5)
        assert 30 < n < 100

    def test_small_effect_needs_more(self):
        n_small = ABTestRunner.minimum_sample_size(effect_size=0.2)
        n_large = ABTestRunner.minimum_sample_size(effect_size=0.8)
        assert n_small > n_large

    def test_returns_integer(self):
        n = ABTestRunner.minimum_sample_size()
        assert isinstance(n, int)
        assert n > 0


# ============================================================
# TestEdgeCases
# ============================================================


class TestEdgeCases:
    def test_single_query(self):
        runner = ABTestRunner(n_bootstrap=100)
        champ = [{"f1": 0.5, "faithfulness": 0.8}]
        chall = [{"f1": 0.6, "faithfulness": 0.9}]
        result = runner.compare(champ, chall)
        # Should not crash; p-value will be NaN or 1.0 for n=1
        assert result.n_queries == 1

    def test_all_identical_values(self):
        runner = ABTestRunner(n_bootstrap=100)
        results = [{"f1": 0.5, "faithfulness": 0.8}] * 20
        result = runner.compare(results, results)
        assert result.winner == "no_significant_difference"

    def test_primary_metric_missing(self):
        runner = ABTestRunner(primary_metric="nonexistent", n_bootstrap=100)
        champ = make_results(20, seed=1)
        chall = make_results(20, seed=2)
        result = runner.compare(champ, chall)
        assert result.winner == "no_significant_difference"
        assert "not available" in result.recommendation

    def test_no_guard_metrics(self):
        runner = ABTestRunner(guard_metrics=[], n_bootstrap=200)
        champ = make_results(100, f1=0.4, noise=0.03, seed=1)
        chall = make_results(100, f1=0.7, noise=0.03, seed=2)
        result = runner.compare(champ, chall)
        assert result.winner == "challenger"
