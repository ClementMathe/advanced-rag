"""
Regression testing for RAG pipeline metrics.

Detects metric degradation by comparing current results against:
1. Absolute thresholds — minimum acceptable values
2. Relative tolerance — maximum allowed drop from a saved baseline

Usage::

    tester = RegressionTester(
        baseline_path="outputs/baseline_metrics.json",
        relative_tolerance=0.10,  # 10% max drop allowed
    )
    report = tester.check(current_results)
    if not report.passed:
        print(report.summary())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class RegressionFailure:
    """Detail of a single regression failure."""

    metric_name: str
    current_value: float
    baseline_value: Optional[float]
    threshold: float
    failure_type: str  # "absolute" or "relative"
    message: str


@dataclass
class RegressionReport:
    """Result of a regression check."""

    passed: bool
    failures: List[RegressionFailure]
    warnings: List[str]
    config_name: str

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Regression check: {status} (config: {self.config_name})"]

        if self.failures:
            lines.append(f"  {len(self.failures)} failure(s):")
            for f in self.failures:
                lines.append(f"    - {f.message}")

        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"    - {w}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "passed": self.passed,
            "config_name": self.config_name,
            "failures": [asdict(f) for f in self.failures],
            "warnings": self.warnings,
        }


class RegressionTester:
    """
    Detect metric regressions against baselines and absolute floors.

    Args:
        baseline_path: Path to baseline JSON file. None to skip relative checks.
        absolute_thresholds: Dict of metric_name -> minimum acceptable value.
            Defaults to sensible RAG thresholds.
        relative_tolerance: Maximum relative decrease from baseline (e.g. 0.10 = 10%).
    """

    DEFAULT_ABSOLUTE_THRESHOLDS: Dict[str, float] = {
        "f1": 0.25,
        "rouge_l": 0.20,
        "faithfulness": 0.40,
        "exact_match": 0.05,
    }

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        absolute_thresholds: Optional[Dict[str, float]] = None,
        relative_tolerance: float = 0.10,
    ) -> None:
        self._baseline_path = baseline_path
        self._absolute_thresholds = absolute_thresholds or self.DEFAULT_ABSOLUTE_THRESHOLDS
        self._relative_tolerance = relative_tolerance
        self._baseline: Optional[Dict[str, Any]] = None

        if baseline_path and Path(baseline_path).exists():
            self._baseline = self.load_baseline(baseline_path)

    def check(
        self,
        current_results: Dict[str, Any],
        config_name: str = "default",
    ) -> RegressionReport:
        """
        Check current results for regressions.

        Args:
            current_results: Dict mapping metric_name -> {mean: float, ...}
                or metric_name -> float.
            config_name: Label for this check.

        Returns:
            RegressionReport with pass/fail and details.
        """
        failures: List[RegressionFailure] = []
        warnings: List[str] = []

        for metric, threshold in self._absolute_thresholds.items():
            current_value = self._extract_mean(current_results, metric)
            if current_value is None:
                continue

            if current_value < threshold:
                failures.append(
                    RegressionFailure(
                        metric_name=metric,
                        current_value=current_value,
                        baseline_value=None,
                        threshold=threshold,
                        failure_type="absolute",
                        message=f"{metric} ({current_value:.3f}) below absolute threshold ({threshold:.3f})",
                    )
                )
            elif current_value < threshold * 1.1:
                warnings.append(
                    f"{metric} ({current_value:.3f}) approaching absolute threshold ({threshold:.3f})"
                )

        # Relative check against baseline
        if self._baseline is not None:
            for metric in self._baseline:
                baseline_value = self._extract_mean(self._baseline, metric)
                current_value = self._extract_mean(current_results, metric)
                if baseline_value is None or current_value is None:
                    continue
                if baseline_value == 0:
                    continue

                relative_drop = (baseline_value - current_value) / baseline_value
                if relative_drop > self._relative_tolerance:
                    failures.append(
                        RegressionFailure(
                            metric_name=metric,
                            current_value=current_value,
                            baseline_value=baseline_value,
                            threshold=self._relative_tolerance,
                            failure_type="relative",
                            message=(
                                f"{metric} dropped {relative_drop:.1%} from baseline "
                                f"({baseline_value:.3f} -> {current_value:.3f}), "
                                f"tolerance is {self._relative_tolerance:.1%}"
                            ),
                        )
                    )

        passed = len(failures) == 0
        report = RegressionReport(
            passed=passed,
            failures=failures,
            warnings=warnings,
            config_name=config_name,
        )

        if not passed:
            logger.warning(report.summary())
        else:
            logger.info(f"Regression check passed for {config_name}")

        return report

    def save_baseline(
        self,
        results: Dict[str, Any],
        path: Optional[str] = None,
    ) -> None:
        """Save current results as baseline."""
        save_path = Path(path or self._baseline_path or "outputs/baseline_metrics.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Baseline saved to {save_path}")
        self._baseline = results

    def load_baseline(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load baseline from JSON file."""
        load_path = Path(path or self._baseline_path or "outputs/baseline_metrics.json")
        with open(load_path) as f:
            return json.load(f)

    @staticmethod
    def _extract_mean(results: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract mean value from results dict (handles nested and flat formats)."""
        if metric not in results:
            return None
        value = results[metric]
        if isinstance(value, dict):
            return value.get("mean")
        if isinstance(value, (int, float)):
            return float(value)
        return None
