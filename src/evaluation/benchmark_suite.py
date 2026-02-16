"""
Benchmark suite runner for RAG pipeline evaluation.

Orchestrates running evaluation queries through one or more pipeline
configurations, computing all metrics (local + optional RAGAS),
tracking cost/TTFT, and producing structured results.

Supports both linear (RAGPipeline) and agentic (AgenticRAGPipeline)
pipeline interfaces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.evaluation.metrics import (
    CostTracker,
    MetricsCalculator,
    RagasEvaluator,
    TTFTMeasurer,
)


class BenchmarkSuite:
    """
    Production-grade benchmark runner for RAG pipeline evaluation.

    Usage::

        calc = MetricsCalculator(embed_model=embed)
        suite = BenchmarkSuite(metrics_calculator=calc)
        result = suite.evaluate_pipeline(pipeline, queries, config_name="linear")

    Args:
        metrics_calculator: MetricsCalculator instance for local metrics.
        cost_tracker: Optional CostTracker for token/cost tracking.
        ttft_measurer: Optional TTFTMeasurer for time-to-first-token.
        ragas_evaluator: Optional RagasEvaluator for LLM-based metrics.
        output_dir: Directory for saving results.
    """

    def __init__(
        self,
        metrics_calculator: MetricsCalculator,
        cost_tracker: Optional[CostTracker] = None,
        ttft_measurer: Optional[TTFTMeasurer] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        output_dir: str = "outputs/benchmark",
    ) -> None:
        self.metrics_calculator = metrics_calculator
        self.cost_tracker = cost_tracker
        self.ttft_measurer = ttft_measurer
        self.ragas_evaluator = ragas_evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_pipeline(
        self,
        pipeline: Any,
        queries: List[Dict[str, Any]],
        config_name: str = "default",
        pipeline_type: str = "linear",
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a single pipeline configuration.

        Args:
            pipeline: RAGPipeline or AgenticRAGPipeline instance.
            queries: List of dicts with 'query' and 'ground_truth' keys.
            config_name: Name for this configuration (used in output).
            pipeline_type: "linear" or "agentic".

        Returns:
            Dict with config_name, per_query_results, aggregate_metrics,
            and optional cost_summary / ragas_metrics.
        """
        logger.info(f"Evaluating pipeline: {config_name} ({pipeline_type})")
        per_query_results = []

        for q_data in tqdm(queries, desc=f"Evaluating {config_name}"):
            result = self._evaluate_single_query(pipeline, q_data, pipeline_type)
            per_query_results.append(result)

        # Aggregate metrics
        aggregate = self.aggregate_results(per_query_results)

        output: Dict[str, Any] = {
            "config_name": config_name,
            "pipeline_type": pipeline_type,
            "n_queries": len(queries),
            "per_query_results": per_query_results,
            "aggregate_metrics": aggregate,
        }

        # Cost summary
        if self.cost_tracker is not None:
            output["cost_summary"] = self.cost_tracker.get_summary()
            self.cost_tracker.reset()

        # RAGAS batch evaluation
        if self.ragas_evaluator is not None and self.ragas_evaluator.is_available:
            logger.info("Running RAGAS LLM-based evaluation...")
            ragas_samples = [
                {
                    "query": r["query"],
                    "answer": r["prediction"],
                    "contexts": r.get("retrieved_contexts", []),
                    "ground_truth": r["ground_truth"],
                }
                for r in per_query_results
            ]
            output["ragas_metrics"] = self.ragas_evaluator.evaluate_batch(ragas_samples)

        return output

    def run_comparison(
        self,
        pipelines: Dict[str, Any],
        queries: List[Dict[str, Any]],
        pipeline_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple pipeline configurations and compare.

        Args:
            pipelines: Map of config_name -> pipeline instance.
            queries: Shared query dataset.
            pipeline_types: Map of config_name -> "linear" | "agentic".
                Defaults to "linear" for all.

        Returns:
            Dict with 'configs' (per-config results) and 'comparison' (deltas).
        """
        if pipeline_types is None:
            pipeline_types = {name: "linear" for name in pipelines}

        configs: Dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            p_type = pipeline_types.get(name, "linear")
            configs[name] = self.evaluate_pipeline(pipeline, queries, name, p_type)

        # Compute deltas between configs
        comparison = self._compute_comparison(configs)

        return {
            "configs": configs,
            "comparison": comparison,
        }

    def _evaluate_single_query(
        self,
        pipeline: Any,
        query_data: Dict[str, Any],
        pipeline_type: str,
    ) -> Dict[str, Any]:
        """Evaluate a single query: run pipeline, compute metrics."""
        query = query_data["query"]
        ground_truth = query_data["ground_truth"]

        # Run pipeline with timing
        start_time = time.time()

        if pipeline_type == "linear":
            raw_result = pipeline.query(query, return_intermediate=True)
            answer = raw_result.get("answer", "")
            contexts = raw_result.get("reranked_chunks", [])
            context_texts = [c.get("content", "") for c in contexts]
        else:  # agentic
            raw_result = pipeline.query(query)
            answer = raw_result.get("answer", "")
            contexts = raw_result.get("context_documents", [])
            context_texts = [c.get("content", "") for c in contexts]

        latency_s = time.time() - start_time

        # Compute local metrics
        metrics = self.metrics_calculator.compute_all(answer, ground_truth, context_texts)
        metrics["latency_s"] = latency_s

        # Build result
        result: Dict[str, Any] = {
            "query": query,
            "ground_truth": ground_truth,
            "prediction": answer,
            "retrieved_contexts": context_texts,
            **metrics,
        }

        # Add agentic-specific fields
        if pipeline_type == "agentic":
            for key in [
                "retry_count",
                "min_rerank_score",
                "used_fallback_retrieval",
                "used_web_search",
                "answer_is_acceptable",
            ]:
                if key in raw_result:
                    result[key] = raw_result[key]

        # Track cost
        if self.cost_tracker is not None:
            prompt = raw_result.get("prompt", query)
            self.cost_tracker.record_query(prompt, answer)

        return result

    @staticmethod
    def aggregate_results(per_query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute mean/std/min/max/p50/p95 for each numeric metric.

        Args:
            per_query_results: List of per-query result dicts.

        Returns:
            Dict mapping metric_name -> {mean, std, min, max, p50, p95}.
        """
        if not per_query_results:
            return {}

        # Collect numeric fields
        numeric_keys = set()
        for r in per_query_results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in ("retry_count",):
                    numeric_keys.add(k)

        aggregate: Dict[str, Any] = {}
        for key in sorted(numeric_keys):
            values = [r[key] for r in per_query_results if key in r]
            if not values:
                continue
            arr = np.array(values, dtype=float)
            aggregate[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }

        return aggregate

    def _compute_comparison(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metric deltas between all pairs of configurations."""
        names = list(configs.keys())
        if len(names) < 2:
            return {}

        comparison: Dict[str, Any] = {}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                agg_a = configs[name_a].get("aggregate_metrics", {})
                agg_b = configs[name_b].get("aggregate_metrics", {})
                deltas = {}
                for metric in set(agg_a.keys()) & set(agg_b.keys()):
                    mean_a = agg_a[metric].get("mean", 0)
                    mean_b = agg_b[metric].get("mean", 0)
                    deltas[metric] = {
                        f"{name_a}_mean": mean_a,
                        f"{name_b}_mean": mean_b,
                        "delta": mean_b - mean_a,
                        "relative_delta": (mean_b - mean_a) / mean_a if mean_a != 0 else 0,
                    }
                comparison[f"{name_a}_vs_{name_b}"] = deltas

        return comparison

    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = "benchmark_results.json",
    ) -> Path:
        """Save results to JSON file in output_dir."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
        return path
