"""
Evaluation framework for the Advanced RAG system.

Provides:
- MetricsCalculator: Local metrics (EM, F1, ROUGE-L, faithfulness, BERTScore)
- CostTracker: Token counting and hypothetical API cost estimation
- TTFTMeasurer: Time to First Token measurement wrapper
- RagasEvaluator: LLM-based metrics via RAGAS (requires Mistral API key)
- BenchmarkSuite: Full evaluation runner across pipeline configurations
- RegressionTester: Threshold-based regression detection
"""

from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.metrics import (
    CostTracker,
    MetricsCalculator,
    RagasEvaluator,
    TTFTMeasurer,
)
from src.evaluation.regression_tests import RegressionTester

__all__ = [
    "MetricsCalculator",
    "CostTracker",
    "TTFTMeasurer",
    "RagasEvaluator",
    "BenchmarkSuite",
    "RegressionTester",
]
