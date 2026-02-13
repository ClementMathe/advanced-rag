"""
Evaluate agentic RAG pipeline ablation: 5 configurations vs linear baseline.

Runs all pipelines on the same SQuAD queries and compares:
- Standard metrics: Exact Match, F1, ROUGE-L, Faithfulness
- Agentic-specific metrics: retry_rate, fallback_rate, avg_retries

Configurations:
1. Linear: standard RAG pipeline (retrieve -> rerank -> generate)
2. Adaptive: adaptive retrieval (fallback to BM25-heavy on poor rerank scores)
3. AnswerGrading: + LLM answer grading with retry on failure
4. RerankThreshold: + rerank score threshold (metadata only, no answer grading)
5. Full: adaptive retrieval + answer grading + rerank threshold

Outputs:
- outputs/agentic_eval/ablation_comparison.json    (aggregate comparison)
- outputs/agentic_eval/linear_results.json         (per-query linear results)
- outputs/agentic_eval/<config>_results.json       (per-query per-config results)

Usage:
    python scripts/evaluate_agentic.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from ablation_study import GenerationMetrics
from loguru import logger
from tqdm import tqdm

from src.agentic_pipeline import AgenticRAGPipeline
from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.generator import LLMGenerator
from src.graders import AnswerGrader
from src.pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import LoggerConfig, Timer
from src.web_search import DuckDuckGoSearchTool

# Ablation configurations for AgenticRAGPipeline
# Each config maps to constructor kwargs for the feature flags.
# Configs with enable_adaptive_retrieval=True require a fallback_retriever.
AGENTIC_CONFIGS = {
    "adaptive": {
        "enable_adaptive_retrieval": True,
        "enable_answer_grading": False,
        "enable_rerank_threshold": False,
    },
    "adaptive_web": {
        "enable_adaptive_retrieval": True,
        "enable_web_search": True,
        "enable_answer_grading": False,
        "enable_rerank_threshold": False,
        "web_search_threshold": -2.0,
    },
}


def load_components():
    """Load all shared RAG components.

    Returns:
        Tuple of (hybrid_retriever, reranker, generator, answer_grader, embed_model).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FAISS index
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load("index/squad")

    # BM25 (rebuild from FAISS metadata)
    logger.info("Building BM25 index...")
    chunks = []
    for meta in faiss_index.chunk_metadata:
        chunk = Chunk(
            content=meta["content"],
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            start_char=0,
            end_char=len(meta["content"]),
            chunk_index=meta["chunk_index"],
            metadata=meta.get("metadata", {}),
        )
        chunks.append(chunk)

    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    # Embedding model (shared: retrieval + faithfulness)
    logger.info("Loading embedding model...")
    embed_model = EmbeddingModel(model_name="BAAI/bge-large-en-v1.5", device=device)

    # Retriever
    dense_retriever = DenseRetriever(faiss_index, embed_model)
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        k_rrf=60,
        dense_weight=0.9,
        sparse_weight=0.1,
    )

    # Reranker
    logger.info("Loading reranker...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
    )

    # Generator (shared: generation + answer grading)
    logger.info("Loading LLM generator...")
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        temperature=0.1,
        max_new_tokens=80,
    )

    # Answer grader (reuses generator, no extra VRAM)
    answer_grader = AnswerGrader(generator)

    return hybrid_retriever, reranker, generator, answer_grader, embed_model


def evaluate_linear(
    pipeline: RAGPipeline,
    queries: List[Dict],
    metrics_calc: GenerationMetrics,
) -> List[Dict[str, Any]]:
    """Evaluate linear pipeline on queries.

    Args:
        pipeline: Linear RAG pipeline instance.
        queries: List of query dicts with 'query' and 'answer' fields.
        metrics_calc: GenerationMetrics instance for scoring.

    Returns:
        List of per-query result dicts with metrics and timings.
    """
    results = []

    for query_data in tqdm(queries, desc="Linear pipeline"):
        question = query_data["query"]
        ground_truth = query_data.get("answer", "")

        start_time = time.time()
        result = pipeline.query(question, return_intermediate=True)
        elapsed = time.time() - start_time

        prediction = result.get("answer", "")
        chunks = result.get("reranked_chunks", [])
        chunk_texts = [c.get("content", "") for c in chunks]

        metrics = metrics_calc.compute_all(prediction, ground_truth, chunk_texts)

        results.append(
            {
                "query_id": query_data.get("id", "unknown"),
                "query": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": metrics,
                "num_chunks": len(chunks),
                "latency_s": elapsed,
            }
        )

    return results


def evaluate_agentic(
    pipeline: AgenticRAGPipeline,
    queries: List[Dict],
    metrics_calc: GenerationMetrics,
    config_name: str,
) -> List[Dict[str, Any]]:
    """Evaluate an agentic pipeline configuration on queries.

    Args:
        pipeline: Agentic RAG pipeline instance.
        queries: List of query dicts with 'query' and 'answer' fields.
        metrics_calc: GenerationMetrics instance for scoring.
        config_name: Configuration name for progress bar display.

    Returns:
        List of per-query result dicts with metrics, timings, and retry info.
    """
    results = []

    for query_data in tqdm(queries, desc=f"Agentic ({config_name})"):
        question = query_data["query"]
        ground_truth = query_data.get("answer", "")

        start_time = time.time()
        result = pipeline.query(question)
        elapsed = time.time() - start_time

        prediction = result.get("answer", "")
        context_docs = result.get("context_documents", [])
        chunk_texts = [d.get("content", "") for d in context_docs]

        metrics = metrics_calc.compute_all(prediction, ground_truth, chunk_texts)

        results.append(
            {
                "query_id": query_data.get("id", "unknown"),
                "query": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": metrics,
                "num_docs_retrieved": result.get("num_docs_retrieved", 0),
                "retry_count": result.get("retry_count", 0),
                "min_rerank_score": result.get("min_rerank_score", 0.0),
                "answer_is_acceptable": result.get("answer_is_acceptable", True),
                "used_fallback_retrieval": result.get("used_fallback_retrieval", False),
                "used_web_search": result.get("used_web_search", False),
                "steps": result.get("steps", []),
                "latency_s": elapsed,
            }
        )

    return results


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query results into summary statistics.

    Args:
        results: List of per-query result dicts.

    Returns:
        Dictionary with mean/std/min/max for each metric.
    """
    metric_names = ["exact_match", "f1", "rouge_l", "faithfulness"]
    agg = {}

    for name in metric_names:
        scores = [r["metrics"][name] for r in results]
        agg[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    # Latency
    latencies = [r["latency_s"] for r in results]
    agg["latency_s"] = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }

    return agg


def aggregate_agentic_specific(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute agentic-specific metrics (retry rate, answer quality).

    Args:
        results: List of per-query agentic result dicts.

    Returns:
        Dictionary with retry stats, answer quality, and rerank scores.
    """
    retry_counts = [r["retry_count"] for r in results]
    num_with_retry = sum(1 for c in retry_counts if c > 0)
    max_retry = int(np.max(retry_counts)) if retry_counts else 0

    rerank_scores = [r["min_rerank_score"] for r in results]
    acceptable_count = sum(1 for r in results if r["answer_is_acceptable"])
    fallback_count = sum(1 for r in results if r.get("used_fallback_retrieval", False))
    web_search_count = sum(1 for r in results if r.get("used_web_search", False))

    agg = {
        "retry_rate": float(num_with_retry / len(results)) if results else 0.0,
        "avg_retries": float(np.mean(retry_counts)) if retry_counts else 0.0,
        "max_retries": max_retry,
        "answer_acceptable_rate": float(acceptable_count / len(results)) if results else 0.0,
        "fallback_rate": float(fallback_count / len(results)) if results else 0.0,
        "web_search_rate": float(web_search_count / len(results)) if results else 0.0,
        "min_rerank_score": {
            "mean": float(np.mean(rerank_scores)),
            "min": float(np.min(rerank_scores)),
            "max": float(np.max(rerank_scores)),
        },
    }

    if max_retry > 0:
        agg["retry_distribution"] = {
            str(i): int(retry_counts.count(i)) for i in range(max_retry + 1)
        }
    else:
        agg["retry_distribution"] = {"0": len(results)}

    return agg


def build_comparison(
    all_agg: Dict[str, Dict],
    all_agentic_specific: Dict[str, Dict],
) -> Dict[str, Any]:
    """Build comparison dictionary across all configurations.

    Args:
        all_agg: Map of config_name -> aggregated metrics (includes "linear").
        all_agentic_specific: Map of config_name -> agentic-specific metrics.

    Returns:
        Comparison dictionary with per-config aggregates and deltas vs linear.
    """
    linear_agg = all_agg["linear"]

    comparison = {
        "configs": {},
        "deltas_vs_linear": {},
    }

    for config_name, agg in all_agg.items():
        comparison["configs"][config_name] = {
            "metrics": agg,
        }
        if config_name in all_agentic_specific:
            comparison["configs"][config_name]["agentic_specific"] = all_agentic_specific[
                config_name
            ]

        # Compute deltas vs linear
        if config_name != "linear":
            delta = {}
            for metric in ["exact_match", "f1", "rouge_l", "faithfulness"]:
                delta[metric] = float(agg[metric]["mean"] - linear_agg[metric]["mean"])
            delta["latency_s"] = float(agg["latency_s"]["mean"] - linear_agg["latency_s"]["mean"])
            comparison["deltas_vs_linear"][config_name] = delta

    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print formatted multi-config comparison table.

    Args:
        comparison: Comparison dictionary from build_comparison().
    """
    configs = list(comparison["configs"].keys())
    # Column labels (abbreviated for readability)
    col_labels = {
        "linear": "Linear",
        "adaptive": "Adaptive",
        "adaptive_web": "Adpt+Web",
        "answer_grading": "AnsGrade",
        "rerank_threshold": "RerankTh",
        "full": "Full",
    }

    header = f"{'Metric':<20}"
    for cfg in configs:
        header += f" {col_labels.get(cfg, cfg):>10}"
    header += f" {'Best Δ':>10}"

    logger.info("")
    logger.info("=" * len(header))
    logger.info("ABLATION: Agentic RAG Configurations")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    # Standard metrics
    for metric in ["exact_match", "f1", "rouge_l", "faithfulness"]:
        row = f"{metric:<20}"
        values = []
        for cfg in configs:
            val = comparison["configs"][cfg]["metrics"][metric]["mean"]
            row += f" {val:>10.2%}"
            values.append(val)
        # Best delta vs linear
        linear_val = values[0]
        deltas = [v - linear_val for v in values[1:]]
        if deltas:
            best_delta = max(deltas, key=abs)
            sign = "+" if best_delta >= 0 else ""
            row += f" {sign}{best_delta:>9.2%}"
        logger.info(row)

    # Latency
    row = f"{'latency (avg)':<20}"
    lat_values = []
    for cfg in configs:
        lat = comparison["configs"][cfg]["metrics"]["latency_s"]["mean"]
        row += f" {lat:>9.1f}s"
        lat_values.append(lat)
    if len(lat_values) > 1:
        linear_lat = lat_values[0]
        lat_deltas = [v - linear_lat for v in lat_values[1:]]
        best_lat_delta = max(lat_deltas, key=abs)
        sign = "+" if best_lat_delta >= 0 else ""
        row += f" {sign}{best_lat_delta:>8.1f}s"
    logger.info(row)

    # Agentic-specific metrics
    logger.info("-" * len(header))
    for spec_metric in [
        "retry_rate",
        "avg_retries",
        "answer_acceptable_rate",
        "fallback_rate",
        "web_search_rate",
    ]:
        row = f"{spec_metric:<20}"
        for cfg in configs:
            agentic_spec = comparison["configs"][cfg].get("agentic_specific")
            if agentic_spec and spec_metric in agentic_spec:
                val = agentic_spec[spec_metric]
                if "rate" in spec_metric:
                    row += f" {val:>10.2%}"
                else:
                    row += f" {val:>10.2f}"
            else:
                row += f" {'n/a':>10}"
        logger.info(row)

    # Retry distribution for configs that have retries
    logger.info("-" * len(header))
    for cfg in configs:
        agentic_spec = comparison["configs"][cfg].get("agentic_specific")
        if agentic_spec and agentic_spec.get("max_retries", 0) > 0:
            logger.info(f"Retry distribution ({col_labels.get(cfg, cfg)}):")
            for count, num in sorted(agentic_spec["retry_distribution"].items()):
                logger.info(f"  {count} retries: {num} queries")

    logger.info("=" * len(header))


def main():
    """Run 5-config ablation: linear baseline + 4 agentic configurations."""
    LoggerConfig.setup(level="INFO")

    # Configuration
    SAMPLE_SIZE = 100
    QUERIES_PATH = "data/squad/queries_500_with_answers.json"
    OUTPUT_DIR = "outputs/agentic_eval"
    RERANK_THRESHOLD = 0.0  # Calibrate after first run using rerank score stats
    RETRIEVAL_THRESHOLD = 0.0  # Min rerank score for acceptable retrieval

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ABLATION: Agentic RAG — 5-Config Evaluation")
    logger.info("=" * 70)

    # 1. Load shared components
    logger.info("\n1. Loading components...")
    with Timer("Component loading"):
        (
            hybrid_retriever,
            reranker,
            generator,
            answer_grader,
            embed_model,
        ) = load_components()

    # 2. Create pipelines
    logger.info("\n2. Creating pipelines...")

    linear_pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        k_retrieve=20,
        k_rerank=5,
        use_reranking=True,
        use_generation=True,
    )

    agentic_pipelines = {}
    web_search_tool = DuckDuckGoSearchTool(max_results=5)
    for config_name, config_flags in AGENTIC_CONFIGS.items():
        needs_grader = config_flags.get("enable_answer_grading", False)
        needs_web = config_flags.get("enable_web_search", False)
        agentic_pipelines[config_name] = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader if needs_grader else None,
            web_search_tool=web_search_tool if needs_web else None,
            k_retrieve=20,
            k_rerank=5,
            max_retries=1,
            rerank_threshold=RERANK_THRESHOLD,
            retrieval_threshold=RETRIEVAL_THRESHOLD,
            **config_flags,
        )

    # 3. Metrics calculator (reuses embed_model for faithfulness)
    metrics_calc = GenerationMetrics(embed_model)

    # 4. Load queries
    logger.info(f"\n3. Loading queries from {QUERIES_PATH}...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    queries = all_queries[:SAMPLE_SIZE]
    logger.info(f"Using {len(queries)} queries")

    # 5. Evaluate linear baseline
    logger.info("\n4. Evaluating linear baseline...")
    with Timer("Linear evaluation"):
        linear_results = evaluate_linear(linear_pipeline, queries, metrics_calc)

    # 6. Evaluate each agentic config
    all_agentic_results = {}
    for i, (config_name, pipeline) in enumerate(agentic_pipelines.items(), start=1):
        logger.info(f"\n5.{i}. Evaluating agentic ({config_name})...")
        with Timer(f"Agentic ({config_name}) evaluation"):
            all_agentic_results[config_name] = evaluate_agentic(
                pipeline, queries, metrics_calc, config_name
            )

    # 7. Aggregate metrics
    logger.info("\n6. Computing aggregate metrics...")
    all_agg = {"linear": aggregate_metrics(linear_results)}
    all_agentic_specific = {}

    for config_name, results in all_agentic_results.items():
        all_agg[config_name] = aggregate_metrics(results)
        all_agentic_specific[config_name] = aggregate_agentic_specific(results)

    # 8. Compare and display
    comparison = build_comparison(all_agg, all_agentic_specific)
    print_comparison_table(comparison)

    # 9. Save results
    logger.info("\n7. Saving results...")

    with open(output_dir / "ablation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(output_dir / "linear_results.json", "w", encoding="utf-8") as f:
        json.dump(linear_results, f, indent=2, ensure_ascii=False)

    for config_name, results in all_agentic_results.items():
        filename = f"{config_name}_results.json"
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/:")
    logger.info("  - ablation_comparison.json (aggregate metrics + deltas)")
    logger.info(f"  - linear_results.json ({len(linear_results)} per-query results)")
    for config_name, results in all_agentic_results.items():
        logger.info(f"  - {config_name}_results.json ({len(results)} per-query results)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
