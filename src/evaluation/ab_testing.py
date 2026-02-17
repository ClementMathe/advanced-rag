"""
A/B testing infrastructure for RAG pipeline comparison.

Provides statistical comparison of two pipeline configurations using:
- Paired t-test for significance testing
- Bootstrap confidence intervals for non-parametric estimation
- Cohen's d for effect size measurement
- Power analysis for minimum sample size estimation

Usage::

    runner = ABTestRunner(primary_metric="f1", guard_metrics=["faithfulness"])
    result = runner.compare(champion_results, challenger_results)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Metric keys to exclude from auto-detection
# ---------------------------------------------------------------------------

_NON_METRIC_KEYS = frozenset(
    {
        "query",
        "prediction",
        "ground_truth",
        "retrieved_contexts",
        "contexts",
        "categories",
        "retry_count",
        "used_fallback_retrieval",
        "used_web_search",
        "answer_is_acceptable",
        "prompt",
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """Statistical comparison for a single metric between two configurations."""

    metric_name: str
    champion_mean: float
    challenger_mean: float
    difference: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "champion_mean": self.champion_mean,
            "challenger_mean": self.challenger_mean,
            "difference": self.difference,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "effect_size": self.effect_size,
            "significant": self.significant,
        }


@dataclass
class ABTestResult:
    """Full A/B test results with statistical comparisons."""

    champion_name: str
    challenger_name: str
    n_queries: int
    metric_comparisons: Dict[str, MetricComparison]
    winner: str
    recommendation: str

    def summary(self) -> str:
        """Human-readable summary of A/B test results."""
        lines = [
            f"A/B Test: {self.champion_name} vs {self.challenger_name} "
            f"({self.n_queries} queries)",
            f"Winner: {self.winner}",
            f"Recommendation: {self.recommendation}",
            "",
        ]
        for name, comp in self.metric_comparisons.items():
            sig = "*" if comp.significant else " "
            lines.append(
                f"  {name:<20s} {sig} champion={comp.champion_mean:.4f}  "
                f"challenger={comp.challenger_mean:.4f}  "
                f"diff={comp.difference:+.4f}  p={comp.p_value:.4f}  "
                f"d={comp.effect_size:.3f}  "
                f"CI=[{comp.ci_lower:+.4f}, {comp.ci_upper:+.4f}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "champion_name": self.champion_name,
            "challenger_name": self.challenger_name,
            "n_queries": self.n_queries,
            "metric_comparisons": {k: v.to_dict() for k, v in self.metric_comparisons.items()},
            "winner": self.winner,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# ABTestRunner
# ---------------------------------------------------------------------------


class ABTestRunner:
    """
    Statistical A/B testing for RAG pipeline configurations.

    Compares per-query results from two pipeline runs on the same queries,
    using paired statistical tests.

    Args:
        alpha: Significance level (default 0.05 for 95% confidence).
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        primary_metric: Metric used to determine the winner.
        guard_metrics: Metrics that must not significantly regress for the
            challenger to win.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        primary_metric: str = "f1",
        guard_metrics: Optional[List[str]] = None,
    ) -> None:
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.primary_metric = primary_metric
        self.guard_metrics = guard_metrics if guard_metrics is not None else ["faithfulness"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        champion_results: List[Dict[str, Any]],
        challenger_results: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        champion_name: str = "champion",
        challenger_name: str = "challenger",
    ) -> ABTestResult:
        """
        Compare two sets of per-query results with statistical tests.

        Both result lists must have the same length (same queries, same order).

        Args:
            champion_results: Per-query results from the champion pipeline.
            challenger_results: Per-query results from the challenger pipeline.
            metrics: List of metric names to compare. If None, auto-detects
                common numeric keys.
            champion_name: Display name for champion.
            challenger_name: Display name for challenger.

        Returns:
            ABTestResult with per-metric comparisons and overall winner.

        Raises:
            ValueError: If result lists have different lengths or are empty.
        """
        if len(champion_results) != len(challenger_results):
            raise ValueError(
                f"Result lists must have same length: "
                f"{len(champion_results)} vs {len(challenger_results)}"
            )
        if not champion_results:
            raise ValueError("Result lists must not be empty")

        if metrics is None:
            metrics = self._detect_metrics(champion_results, challenger_results)

        comparisons: Dict[str, MetricComparison] = {}
        for metric in metrics:
            champ_vals = [r.get(metric, 0.0) for r in champion_results]
            chall_vals = [r.get(metric, 0.0) for r in challenger_results]

            # Skip if all values are None/missing
            if all(v == 0.0 for v in champ_vals) and all(v == 0.0 for v in chall_vals):
                continue

            comparisons[metric] = self._compare_single_metric(
                np.array(champ_vals, dtype=float),
                np.array(chall_vals, dtype=float),
                metric,
            )

        winner, recommendation = self._determine_winner(comparisons)

        result = ABTestResult(
            champion_name=champion_name,
            challenger_name=challenger_name,
            n_queries=len(champion_results),
            metric_comparisons=comparisons,
            winner=winner,
            recommendation=recommendation,
        )

        logger.info(
            f"A/B test complete: {champion_name} vs {challenger_name}, "
            f"winner={winner}, n={len(champion_results)}"
        )
        return result

    @staticmethod
    def minimum_sample_size(
        effect_size: float = 0.3,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """
        Estimate minimum sample size per group for a paired t-test.

        Uses the formula: N = ((z_alpha + z_beta) / effect_size)^2
        where effect_size is Cohen's d.

        Args:
            effect_size: Expected Cohen's d (0.2=small, 0.5=medium, 0.8=large).
            alpha: Significance level (two-tailed).
            power: Desired statistical power.

        Returns:
            Minimum number of paired observations needed.
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compare_single_metric(
        self,
        champion_values: np.ndarray,
        challenger_values: np.ndarray,
        metric_name: str,
    ) -> MetricComparison:
        """Paired t-test + bootstrap CI + Cohen's d for one metric."""
        differences = challenger_values - champion_values

        champion_mean = float(np.mean(champion_values))
        challenger_mean = float(np.mean(challenger_values))
        difference = float(np.mean(differences))

        # Paired t-test
        if np.std(differences) == 0:
            # All differences are identical (including all-zero)
            p_value = 1.0
        else:
            _, p_value = stats.ttest_rel(challenger_values, champion_values)
            p_value = float(p_value)

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(differences)

        # Cohen's d for paired data
        std_diff = float(np.std(differences, ddof=1))
        effect_size = difference / std_diff if std_diff > 0 else 0.0

        return MetricComparison(
            metric_name=metric_name,
            champion_mean=champion_mean,
            challenger_mean=challenger_mean,
            difference=difference,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=effect_size,
            significant=p_value < self.alpha,
        )

    def _bootstrap_ci(
        self,
        differences: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Bootstrap 95% confidence interval on paired differences.

        Resamples the paired differences (not raw values) and computes
        percentile-based CI on the mean difference.
        """
        rng = np.random.default_rng()
        n = len(differences)
        boot_means = np.empty(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            sample = rng.choice(differences, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        return ci_lower, ci_upper

    def _determine_winner(
        self,
        comparisons: Dict[str, MetricComparison],
    ) -> Tuple[str, str]:
        """
        Determine overall winner based on primary metric and guard metrics.

        Logic:
        - Challenger wins if primary metric is significantly better AND
          no guard metric is significantly worse.
        - Champion wins if primary metric is significantly worse for challenger
          OR any guard metric is significantly worse.
        - Otherwise: no significant difference.
        """
        primary = comparisons.get(self.primary_metric)

        # Check guard metrics for regression
        guard_regression = False
        regressed_guards: List[str] = []
        for guard in self.guard_metrics:
            guard_comp = comparisons.get(guard)
            if guard_comp is not None and guard_comp.significant and guard_comp.difference < 0:
                guard_regression = True
                regressed_guards.append(guard)

        if guard_regression:
            reason = ", ".join(regressed_guards)
            return (
                "champion",
                f"Guard metric(s) regressed significantly: {reason}. " f"Keep champion.",
            )

        if primary is None:
            return (
                "no_significant_difference",
                f"Primary metric '{self.primary_metric}' not available in results.",
            )

        if primary.significant and primary.difference > 0:
            return (
                "challenger",
                f"Challenger significantly better on {self.primary_metric} "
                f"(+{primary.difference:.4f}, p={primary.p_value:.4f}, "
                f"d={primary.effect_size:.3f}). No guard metric regression.",
            )

        if primary.significant and primary.difference < 0:
            return (
                "champion",
                f"Challenger significantly worse on {self.primary_metric} "
                f"({primary.difference:.4f}, p={primary.p_value:.4f}). "
                f"Keep champion.",
            )

        return (
            "no_significant_difference",
            f"No significant difference on {self.primary_metric} "
            f"(p={primary.p_value:.4f}). Consider more samples "
            f"(minimum ~{self.minimum_sample_size()} for medium effect).",
        )

    def _detect_metrics(
        self,
        champion_results: List[Dict[str, Any]],
        challenger_results: List[Dict[str, Any]],
    ) -> List[str]:
        """Auto-detect common numeric metric keys from both result lists."""
        champ_keys: set = set()
        for r in champion_results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in _NON_METRIC_KEYS:
                    champ_keys.add(k)

        chall_keys: set = set()
        for r in challenger_results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in _NON_METRIC_KEYS:
                    chall_keys.add(k)

        common = sorted(champ_keys & chall_keys)
        logger.debug(f"Auto-detected metrics: {common}")
        return common
