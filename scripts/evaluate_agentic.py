"""
Evaluate agentic RAG pipeline vs linear baseline.

Runs both pipelines on the same 100 SQuAD queries and compares:
- Standard metrics: Exact Match, F1, ROUGE-L, Faithfulness
- Agentic-specific metrics: retry_rate, avg_retries

Based on the ablation_study.py structure. Reuses GenerationMetrics for
consistent metric computation.

Outputs:
- outputs/agentic_eval/comparison.json     (aggregate comparison)
- outputs/agentic_eval/linear_results.json  (per-query linear results)
- outputs/agentic_eval/agentic_results.json (per-query agentic results)

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
from src.graders import DocumentGrader, QueryRewriter
from src.pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import LoggerConfig, Timer


def load_components():
    """Load all shared RAG components.

    Returns:
        Tuple of (hybrid_retriever, reranker, generator, grader,
                  query_rewriter, embed_model).
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

    # Retrievers
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

    # Generator (shared: generation + grading + rewriting)
    logger.info("Loading LLM generator...")
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        temperature=0.1,
        max_new_tokens=80,
    )

    # Grader and rewriter (reuse generator)
    grader = DocumentGrader(generator)
    query_rewriter = QueryRewriter(generator)

    return hybrid_retriever, reranker, generator, grader, query_rewriter, embed_model


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
) -> List[Dict[str, Any]]:
    """Evaluate agentic pipeline on queries.

    Args:
        pipeline: Agentic RAG pipeline instance.
        queries: List of query dicts with 'query' and 'answer' fields.
        metrics_calc: GenerationMetrics instance for scoring.

    Returns:
        List of per-query result dicts with metrics, timings, and retry info.
    """
    results = []

    for query_data in tqdm(queries, desc="Agentic pipeline"):
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
                "num_docs_graded": result.get("num_docs_graded", 0),
                "retry_count": result.get("retry_count", 0),
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
    """Compute agentic-specific metrics.

    Args:
        results: List of per-query agentic result dicts.

    Returns:
        Dictionary with retry_rate, avg_retries, and retry distribution.
    """
    retry_counts = [r["retry_count"] for r in results]
    num_with_retry = sum(1 for r in retry_counts if r > 0)

    return {
        "retry_rate": float(num_with_retry / len(results)) if results else 0.0,
        "avg_retries": float(np.mean(retry_counts)),
        "max_retries": int(np.max(retry_counts)),
        "retry_distribution": {
            str(i): int(retry_counts.count(i)) for i in range(int(np.max(retry_counts)) + 1)
        },
        "avg_docs_graded": float(np.mean([r["num_docs_graded"] for r in results])),
    }


def compare_results(linear_agg: Dict, agentic_agg: Dict, agentic_specific: Dict) -> Dict[str, Any]:
    """Generate comparison between linear and agentic pipelines.

    Args:
        linear_agg: Aggregated linear metrics.
        agentic_agg: Aggregated agentic metrics.
        agentic_specific: Agentic-only metrics (retry rate, etc.).

    Returns:
        Comparison dictionary with both aggregates and deltas.
    """
    comparison = {
        "linear": linear_agg,
        "agentic": agentic_agg,
        "agentic_specific": agentic_specific,
        "delta": {},
    }

    for metric in ["exact_match", "f1", "rouge_l", "faithfulness"]:
        linear_val = linear_agg[metric]["mean"]
        agentic_val = agentic_agg[metric]["mean"]
        comparison["delta"][metric] = float(agentic_val - linear_val)

    # Latency delta
    comparison["delta"]["latency_s"] = float(
        agentic_agg["latency_s"]["mean"] - linear_agg["latency_s"]["mean"]
    )

    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print formatted comparison table to logger.

    Args:
        comparison: Comparison dictionary from compare_results().
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("AGENTIC vs LINEAR COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<20} {'Linear':>10} {'Agentic':>10} {'Delta':>10}")
    logger.info("-" * 70)

    for metric in ["exact_match", "f1", "rouge_l", "faithfulness"]:
        linear_val = comparison["linear"][metric]["mean"]
        agentic_val = comparison["agentic"][metric]["mean"]
        delta = comparison["delta"][metric]
        sign = "+" if delta >= 0 else ""
        logger.info(f"{metric:<20} {linear_val:>10.2%} {agentic_val:>10.2%} {sign}{delta:>9.2%}")

    # Latency
    linear_lat = comparison["linear"]["latency_s"]["mean"]
    agentic_lat = comparison["agentic"]["latency_s"]["mean"]
    delta_lat = comparison["delta"]["latency_s"]
    sign = "+" if delta_lat >= 0 else ""
    logger.info(
        f"{'latency (avg)':<20} {linear_lat:>9.1f}s {agentic_lat:>9.1f}s {sign}{delta_lat:>8.1f}s"
    )

    # Agentic-specific
    agentic_spec = comparison["agentic_specific"]
    logger.info("-" * 70)
    logger.info(f"{'retry_rate':<20} {'n/a':>10} {agentic_spec['retry_rate']:>10.2%}")
    logger.info(f"{'avg_retries':<20} {'n/a':>10} {agentic_spec['avg_retries']:>10.2f}")
    logger.info(f"{'avg_docs_graded':<20} {'n/a':>10} {agentic_spec['avg_docs_graded']:>10.1f}")

    # Retry distribution
    logger.info("-" * 70)
    logger.info("Retry distribution:")
    for count, num in sorted(agentic_spec["retry_distribution"].items()):
        logger.info(f"  {count} retries: {num} queries")

    logger.info("=" * 70)


def main():
    """Run full evaluation: linear vs agentic pipeline."""
    LoggerConfig.setup(level="INFO")

    # Configuration
    SAMPLE_SIZE = 10
    QUERIES_PATH = "data/squad/queries_500_with_answers.json"
    OUTPUT_DIR = "outputs/agentic_eval"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EVALUATION: Agentic RAG vs Linear Baseline")
    logger.info("=" * 70)

    # Load components
    logger.info("\n1. Loading components...")
    with Timer("Component loading"):
        (
            hybrid_retriever,
            reranker,
            generator,
            grader,
            query_rewriter,
            embed_model,
        ) = load_components()

    # Create pipelines
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

    agentic_pipeline = AgenticRAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        grader=grader,
        query_rewriter=query_rewriter,
        k_retrieve=20,
        k_rerank=10,
        min_relevant=3,
        max_retries=3,
    )

    # Metrics calculator (reuses embed_model for faithfulness)
    metrics_calc = GenerationMetrics(embed_model)

    # Load queries
    logger.info(f"\n3. Loading queries from {QUERIES_PATH}...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    queries = all_queries[:SAMPLE_SIZE]
    logger.info(f"Using {len(queries)} queries")

    # Evaluate linear pipeline
    logger.info("\n4. Evaluating linear pipeline...")
    with Timer("Linear evaluation"):
        linear_results = evaluate_linear(linear_pipeline, queries, metrics_calc)

    # Evaluate agentic pipeline
    logger.info("\n5. Evaluating agentic pipeline...")
    with Timer("Agentic evaluation"):
        agentic_results = evaluate_agentic(agentic_pipeline, queries, metrics_calc)

    # Aggregate
    logger.info("\n6. Computing aggregate metrics...")
    linear_agg = aggregate_metrics(linear_results)
    agentic_agg = aggregate_metrics(agentic_results)
    agentic_specific = aggregate_agentic_specific(agentic_results)

    # Compare
    comparison = compare_results(linear_agg, agentic_agg, agentic_specific)

    # Print table
    print_comparison_table(comparison)

    # Save results
    logger.info("\n7. Saving results...")

    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(output_dir / "linear_results.json", "w", encoding="utf-8") as f:
        json.dump(linear_results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "agentic_results.json", "w", encoding="utf-8") as f:
        json.dump(agentic_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/:")
    logger.info("  - comparison.json (aggregate metrics + deltas)")
    logger.info(f"  - linear_results.json ({len(linear_results)} per-query results)")
    logger.info(f"  - agentic_results.json ({len(agentic_results)} per-query results)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
