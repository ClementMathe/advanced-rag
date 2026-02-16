"""
Error analysis taxonomy for RAG pipeline evaluation.

Automatically categorizes each prediction into failure modes using
rule-based heuristics on computed metrics. Supports both flat
(BenchmarkSuite) and nested (script) prediction formats.

Usage::

    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze(per_query_results)
    print(analysis.error_distribution)
    analyzer.export_report(analysis, "outputs/evaluation/error_analysis.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# ---------------------------------------------------------------------------
# Error category definitions
# ---------------------------------------------------------------------------

ERROR_CATEGORIES: Dict[str, str] = {
    # Retrieval failures
    "retrieval_failure": ("No retrieved context contains the ground truth answer"),
    "low_context_relevance": (
        "Retrieved contexts present but model fails to use them "
        "(low faithfulness despite answer in context)"
    ),
    # Generation failures
    "hallucination": ("Answer not supported by retrieved contexts (low faithfulness)"),
    "incomplete_answer": ("Answer too short or missing key information"),
    "verbose_answer": (
        "Answer contains relevant tokens but is overly verbose "
        "(high ROUGE-L, low exact match, low F1)"
    ),
    "wrong_answer": ("Answer is factually incorrect (low F1, not empty, not hallucination)"),
    # Edge cases
    "empty_response": "Model returned empty or near-empty answer",
    "correct": "Answer is correct (high F1 and/or exact match)",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ErrorAnalysis:
    """Results of automatic error categorization."""

    error_distribution: Dict[str, int]
    error_rate: float
    total_predictions: int
    per_query_errors: List[Dict[str, Any]]
    worst_predictions: List[Dict[str, Any]]
    examples_per_category: Dict[str, List[Dict[str, Any]]]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Error Analysis ({self.total_predictions} predictions, "
            f"error rate: {self.error_rate:.1%})",
        ]
        sorted_dist = sorted(self.error_distribution.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_dist:
            pct = count / self.total_predictions if self.total_predictions else 0
            lines.append(f"  {cat:.<30s} {count:>4d}  ({pct:.1%})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "error_distribution": self.error_distribution,
            "error_rate": self.error_rate,
            "total_predictions": self.total_predictions,
            "per_query_errors": self.per_query_errors,
            "worst_predictions": self.worst_predictions,
            "examples_per_category": self.examples_per_category,
        }


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------


class ErrorAnalyzer:
    """
    Rule-based error categorization for RAG predictions.

    Accepts per-query result dicts in two formats:

    - **Flat** (from BenchmarkSuite): metrics at top level
      ``{"f1": 0.3, "faithfulness": 0.6, "prediction": "...", ...}``
    - **Nested** (from evaluation scripts): metrics under ``"metrics"`` key
      ``{"metrics": {"f1": 0.3, ...}, "prediction": "...", ...}``

    Args:
        faithfulness_threshold: Below this value, answer is considered
            unfaithful to context. Default 0.5.
        f1_threshold: Below this value, answer quality is considered poor.
            Default 0.3.
        max_examples: Maximum examples stored per error category.
        n_worst: Number of worst predictions to keep.
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.5,
        f1_threshold: float = 0.3,
        max_examples: int = 5,
        n_worst: int = 20,
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.f1_threshold = f1_threshold
        self.max_examples = max_examples
        self.n_worst = n_worst

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, predictions: List[Dict[str, Any]]) -> ErrorAnalysis:
        """
        Categorize all predictions into error types.

        Args:
            predictions: List of per-query result dicts (flat or nested).

        Returns:
            ErrorAnalysis with distribution, examples, and worst cases.
        """
        if not predictions:
            return ErrorAnalysis(
                error_distribution={cat: 0 for cat in ERROR_CATEGORIES},
                error_rate=0.0,
                total_predictions=0,
                per_query_errors=[],
                worst_predictions=[],
                examples_per_category={cat: [] for cat in ERROR_CATEGORIES},
            )

        error_distribution: Dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}
        examples: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in ERROR_CATEGORIES}
        per_query_errors: List[Dict[str, Any]] = []
        scored_predictions: List[tuple] = []

        for pred in predictions:
            normalized = self._normalize_prediction(pred)
            categories = self._categorize_single(normalized)

            for cat in categories:
                error_distribution[cat] += 1
                if len(examples[cat]) < self.max_examples:
                    examples[cat].append(self._make_example(normalized, categories))

            per_query_errors.append(
                {
                    "query": normalized.get("query", ""),
                    "prediction": normalized.get("prediction", ""),
                    "ground_truth": normalized.get("ground_truth", ""),
                    "categories": categories,
                    "f1": normalized.get("f1", 0.0),
                    "faithfulness": normalized.get("faithfulness"),
                }
            )

            score = self._composite_badness_score(normalized)
            scored_predictions.append((score, normalized, categories))

        # Sort by badness (ascending = worst first)
        scored_predictions.sort(key=lambda x: x[0])
        worst = [
            self._make_example(pred, cats) for _, pred, cats in scored_predictions[: self.n_worst]
        ]

        n_correct = error_distribution.get("correct", 0)
        n_total = len(predictions)
        error_rate = (n_total - n_correct) / n_total if n_total > 0 else 0.0

        analysis = ErrorAnalysis(
            error_distribution=error_distribution,
            error_rate=error_rate,
            total_predictions=n_total,
            per_query_errors=per_query_errors,
            worst_predictions=worst,
            examples_per_category=examples,
        )

        logger.info(
            f"Error analysis complete: {n_total} predictions, " f"error rate {error_rate:.1%}"
        )
        return analysis

    def export_report(
        self,
        analysis: ErrorAnalysis,
        path: str,
    ) -> Path:
        """Save analysis to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
        logger.info(f"Error analysis report saved to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten prediction dict so metrics are at top level.

        Handles nested format: ``{"metrics": {"f1": ...}, ...}``
        and flat format: ``{"f1": ..., ...}``.
        """
        result = dict(pred)
        if "metrics" in result and isinstance(result["metrics"], dict):
            metrics = result.pop("metrics")
            for key, value in metrics.items():
                if key not in result:
                    result[key] = value
        return result

    # ------------------------------------------------------------------
    # Categorization
    # ------------------------------------------------------------------

    def _categorize_single(self, pred: Dict[str, Any]) -> List[str]:
        """
        Assign error categories to a single prediction.

        A prediction marked "correct" gets no other categories.
        Non-correct predictions can have multiple categories.
        """
        categories: List[str] = []

        prediction_text = str(pred.get("prediction", "")).strip()
        ground_truth = str(pred.get("ground_truth", ""))
        f1 = pred.get("f1", 0.0)
        em = pred.get("exact_match", 0.0)
        rouge_l = pred.get("rouge_l", 0.0)
        faithfulness = pred.get("faithfulness")
        contexts = pred.get("retrieved_contexts", [])

        # --- Correct ---
        if em == 1.0 or f1 >= 0.8:
            return ["correct"]

        # --- Empty response ---
        if not prediction_text:
            return ["empty_response"]

        # --- Retrieval failure ---
        has_retrieval_failure = False
        if contexts and ground_truth:
            gt_lower = ground_truth.lower()
            found_in_context = any(gt_lower in ctx.lower() for ctx in contexts if ctx)
            if not found_in_context:
                categories.append("retrieval_failure")
                has_retrieval_failure = True

        # --- Hallucination ---
        is_hallucination = False
        if faithfulness is not None and faithfulness < self.faithfulness_threshold:
            if f1 > 0:
                categories.append("hallucination")
                is_hallucination = True

        # --- Low context relevance ---
        if (
            faithfulness is not None
            and faithfulness < self.faithfulness_threshold
            and not has_retrieval_failure
            and not is_hallucination
        ):
            categories.append("low_context_relevance")

        # --- Incomplete answer ---
        if len(prediction_text.split()) < 3 and f1 < self.f1_threshold:
            categories.append("incomplete_answer")

        # --- Verbose answer ---
        if rouge_l > 0.4 and em == 0.0 and f1 < 0.5:
            categories.append("verbose_answer")

        # --- Wrong answer (catch-all for remaining bad predictions) ---
        if (
            f1 < self.f1_threshold
            and em == 0.0
            and not is_hallucination
            and "incomplete_answer" not in categories
        ):
            categories.append("wrong_answer")

        return categories if categories else ["wrong_answer"]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_badness_score(pred: Dict[str, Any]) -> float:
        """
        Compute composite quality score. Lower = worse.

        Combines F1, faithfulness, and exact match into a single score
        for ranking worst predictions.
        """
        f1 = pred.get("f1", 0.0)
        em = pred.get("exact_match", 0.0)
        faithfulness = pred.get("faithfulness", 0.5)
        return 0.5 * f1 + 0.3 * (faithfulness or 0.0) + 0.2 * em

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_example(pred: Dict[str, Any], categories: List[str]) -> Dict[str, Any]:
        """Create a compact example dict for storage."""
        return {
            "query": pred.get("query", ""),
            "prediction": pred.get("prediction", ""),
            "ground_truth": pred.get("ground_truth", ""),
            "categories": categories,
            "f1": pred.get("f1", 0.0),
            "exact_match": pred.get("exact_match", 0.0),
            "rouge_l": pred.get("rouge_l", 0.0),
            "faithfulness": pred.get("faithfulness"),
        }
