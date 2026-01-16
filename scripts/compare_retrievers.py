"""
Compare dense, sparse, and hybrid retrieval strategies.

This script:
1. Loads FAISS index (dense retriever)
2. Builds BM25 index (sparse retriever)
3. Creates hybrid retriever with RRF fusion
4. Compares all three on SQuAD queries
5. Calculates metrics for each
6. Visualizes results and analyzes improvements

Usage:
    python scripts/compare_retrievers.py \
        --index_dir index/squad \
        --queries data/squad/sample_queries.json \
        --output_dir outputs/retriever_comparison
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.embeddings import EmbeddingModel, FAISSIndex
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    calculate_retrieval_metrics,
)
from src.utils import LoggerConfig, ensure_dir


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    logger.info(f"Loading queries from {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def build_retrievers(
    index_dir: Path, model_name: str
) -> tuple[DenseRetriever, BM25Retriever, HybridRetriever]:
    """
    Build all three retriever types.

    Args:
        index_dir: Directory containing FAISS index
        model_name: Embedding model name

    Returns:
        Tuple of (dense, sparse, hybrid) retrievers
    """
    logger.info("Building retrievers...")

    # Load FAISS index
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load(str(index_dir))

    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    embed_model = EmbeddingModel(model_name=model_name, device="cuda")

    # Create dense retriever
    dense_retriever = DenseRetriever(faiss_index, embed_model)

    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25_retriever = BM25Retriever(k1=1.2, b=0.75)

    # Extract chunks from FAISS metadata
    from src.chunking import Chunk

    chunks = []
    for metadata in faiss_index.chunk_metadata:
        chunk = Chunk(
            content=metadata["content"],
            chunk_id=metadata["chunk_id"],
            doc_id=metadata["doc_id"],
            start_char=0,
            end_char=len(metadata["content"]),
            chunk_index=metadata["chunk_index"],
            metadata=metadata["metadata"],
        )
        chunks.append(chunk)

    bm25_retriever.index(chunks)

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        k_rrf=60,
        dense_weight=0.9,  # Favor dense (good on SQuAD)
        sparse_weight=0.1,  # Less weight to sparse (noisy on SQuAD)
    )

    logger.info("All retrievers built successfully")
    return dense_retriever, bm25_retriever, hybrid_retriever


def run_retrieval_comparison(
    queries: List[Dict[str, Any]],
    dense_retriever: DenseRetriever,
    sparse_retriever: BM25Retriever,
    hybrid_retriever: HybridRetriever,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run all retrievers on queries and collect results.

    Args:
        queries: List of query dictionaries
        dense_retriever: Dense retriever
        sparse_retriever: Sparse retriever
        hybrid_retriever: Hybrid retriever
        k_values: List of K values to evaluate

    Returns:
        Dictionary with results from all retrievers
    """
    logger.info(f"Running retrieval comparison on {len(queries)} queries...")
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {
        "dense": {"queries": [], "latencies": []},
        "sparse": {"queries": [], "latencies": []},
        "hybrid": {"queries": [], "latencies": []},
    }

    query_texts = [q["query"] for q in queries]
    ground_truth_docs = [q.get("doc_id") for q in queries]
    max_k = max(k_values)

    # Dense retrieval
    logger.info("Running dense retrieval...")
    start = time.perf_counter()
    dense_results = dense_retriever.batch_search(query_texts, k=max_k)
    dense_time = time.perf_counter() - start
    logger.info(
        f"Dense retrieval: {dense_time:.2f}s total, {dense_time/len(queries)*1000:.2f}ms/query"
    )

    for query_dict, retrieved in zip(queries, dense_results):
        results["dense"]["queries"].append(
            {
                "query": query_dict["query"],
                "ground_truth": query_dict.get("doc_id"),
                "retrieved": retrieved,
            }
        )

    results["dense"]["avg_latency_ms"] = dense_time / len(queries) * 1000

    # Sparse retrieval
    logger.info("Running sparse retrieval...")
    start = time.perf_counter()
    sparse_results = sparse_retriever.batch_search(query_texts, k=max_k)
    sparse_time = time.perf_counter() - start
    logger.info(
        f"Sparse retrieval: {sparse_time:.2f}s total, {sparse_time/len(queries)*1000:.2f}ms/query"
    )

    for query_dict, retrieved in zip(queries, sparse_results):
        results["sparse"]["queries"].append(
            {
                "query": query_dict["query"],
                "ground_truth": query_dict.get("doc_id"),
                "retrieved": retrieved,
            }
        )

    results["sparse"]["avg_latency_ms"] = sparse_time / len(queries) * 1000

    # Hybrid retrieval
    logger.info("Running hybrid retrieval...")
    start = time.perf_counter()
    hybrid_results = hybrid_retriever.batch_search(
        query_texts, k=max_k, k_retriever=50
    )  # Increased from 20
    hybrid_time = time.perf_counter() - start
    logger.info(
        f"Hybrid retrieval: {hybrid_time:.2f}s total, {hybrid_time/len(queries)*1000:.2f}ms/query"
    )

    for query_dict, retrieved in zip(queries, hybrid_results):
        results["hybrid"]["queries"].append(
            {
                "query": query_dict["query"],
                "ground_truth": query_dict.get("doc_id"),
                "retrieved": retrieved,
            }
        )

    results["hybrid"]["avg_latency_ms"] = hybrid_time / len(queries) * 1000

    # Calculate metrics for all retrievers
    for retriever_name in ["dense", "sparse", "hybrid"]:
        logger.info(f"\nCalculating metrics for {retriever_name}...")

        retriever_results = [q["retrieved"] for q in results[retriever_name]["queries"]]

        metrics = calculate_retrieval_metrics(retriever_results, ground_truth_docs, k_values)

        results[retriever_name]["metrics"] = metrics

        # Log metrics
        logger.info(f"{retriever_name.capitalize()} Results:")
        for k, recall in metrics["recall_at_k"].items():
            logger.info(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")

    return results


def visualize_comparison(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create visualizations comparing retrievers.

    Args:
        results: Results from run_retrieval_comparison
        output_dir: Directory to save plots
    """
    logger.info("Creating visualizations...")
    ensure_dir(output_dir)

    # Plot 1: Recall@K comparison
    plt.figure(figsize=(10, 6))

    retrievers = ["dense", "sparse", "hybrid"]
    colors = {"dense": "skyblue", "sparse": "lightcoral", "hybrid": "lightgreen"}

    for retriever in retrievers:
        metrics = results[retriever]["metrics"]
        k_values = sorted(metrics["recall_at_k"].keys())
        recall_values = [metrics["recall_at_k"][k] * 100 for k in k_values]

        plt.plot(
            k_values,
            recall_values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=retriever.capitalize(),
            color=colors[retriever],
        )

    plt.xlabel("K (Number of Retrieved Documents)", fontsize=12)
    plt.ylabel("Recall@K (%)", fontsize=12)
    plt.title("Retrieval Performance Comparison: Recall@K", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_comparison.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'recall_comparison.png'}")
    plt.close()

    # Plot 2: Metrics summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Recall@5 comparison
    ax1 = axes[0]
    recall_5 = [results[r]["metrics"]["recall_at_k"][5] * 100 for r in retrievers]
    bars1 = ax1.bar(retrievers, recall_5, color=[colors[r] for r in retrievers], edgecolor="black")
    ax1.set_ylabel("Recall@5 (%)", fontsize=12)
    ax1.set_title("Recall@5 Comparison", fontsize=13, fontweight="bold")
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars1, recall_5):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # MRR comparison
    ax2 = axes[1]
    mrr_values = [results[r]["metrics"]["mrr"] * 100 for r in retrievers]
    bars2 = ax2.bar(
        retrievers, mrr_values, color=[colors[r] for r in retrievers], edgecolor="black"
    )
    ax2.set_ylabel("MRR × 100", fontsize=12)
    ax2.set_title("Mean Reciprocal Rank Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars2, mrr_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'metrics_summary.png'}")
    plt.close()

    # Plot 3: Latency comparison
    plt.figure(figsize=(8, 6))

    latencies = [results[r]["avg_latency_ms"] for r in retrievers]
    bars = plt.bar(retrievers, latencies, color=[colors[r] for r in retrievers], edgecolor="black")
    plt.ylabel("Average Latency (ms)", fontsize=12)
    plt.title("Retrieval Latency Comparison", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars, latencies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}ms",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'latency_comparison.png'}")
    plt.close()

    # Plot 4: Improvement analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(results["dense"]["metrics"]["recall_at_k"].keys())
    dense_recalls = [results["dense"]["metrics"]["recall_at_k"][k] * 100 for k in k_values]
    hybrid_recalls = [results["hybrid"]["metrics"]["recall_at_k"][k] * 100 for k in k_values]
    improvements = [h - d for h, d in zip(hybrid_recalls, dense_recalls)]

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, dense_recalls, width, label="Dense", color="skyblue", edgecolor="black"
    )
    bars2 = ax.bar(
        x + width / 2, hybrid_recalls, width, label="Hybrid", color="lightgreen", edgecolor="black"
    )

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Recall@K (%)", fontsize=12)
    ax.set_title("Hybrid vs Dense: Absolute Improvement", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate improvements
    for _, (bar, improvement) in enumerate(zip(bars2, improvements)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"+{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="green",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_analysis.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'improvement_analysis.png'}")
    plt.close()


def analyze_failure_cases(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Analyze queries where dense fails but hybrid succeeds.

    Args:
        results: Results from run_retrieval_comparison
        output_dir: Directory to save analysis
    """
    logger.info("\nAnalyzing failure cases...")

    dense_queries = results["dense"]["queries"]
    hybrid_queries = results["hybrid"]["queries"]

    improvements = []
    degradations = []

    for dense_q, hybrid_q in zip(dense_queries, hybrid_queries):
        ground_truth = dense_q["ground_truth"]
        if not ground_truth:
            continue

        # Check if ground truth in top-5
        dense_docs = [r["doc_id"] for r in dense_q["retrieved"][:5]]
        hybrid_docs = [r["doc_id"] for r in hybrid_q["retrieved"][:5]]

        dense_found = ground_truth in dense_docs
        hybrid_found = ground_truth in hybrid_docs

        if not dense_found and hybrid_found:
            improvements.append(
                {
                    "query": dense_q["query"],
                    "ground_truth": ground_truth,
                    "dense_top5": dense_docs,
                    "hybrid_top5": hybrid_docs,
                }
            )
        elif dense_found and not hybrid_found:
            degradations.append(
                {
                    "query": dense_q["query"],
                    "ground_truth": ground_truth,
                    "dense_top5": dense_docs,
                    "hybrid_top5": hybrid_docs,
                }
            )

    logger.info(f"Found {len(improvements)} queries improved by hybrid")
    logger.info(f"Found {len(degradations)} queries degraded by hybrid")

    # Save analysis
    analysis = {
        "improvements": improvements,
        "degradations": degradations,
        "summary": {
            "total_improvements": len(improvements),
            "total_degradations": len(degradations),
            "net_improvement": len(improvements) - len(degradations),
        },
    }

    with open(output_dir / "failure_case_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved failure case analysis to {output_dir / 'failure_case_analysis.json'}")

    # Print sample improvements
    if improvements:
        logger.info("\nSample queries improved by hybrid (top 3):")
        for i, case in enumerate(improvements[:3], 1):
            logger.info(f"\n  {i}. Query: {case['query']}")
            logger.info(f"     Ground truth: {case['ground_truth']}")
            logger.info("     Dense found it: NO")
            logger.info("     Hybrid found it: YES")


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save detailed comparison results to JSON."""
    # Remove verbose retrieved docs to reduce file size
    save_data = {}
    for retriever in ["dense", "sparse", "hybrid"]:
        save_data[retriever] = {
            "metrics": results[retriever]["metrics"],
            "avg_latency_ms": results[retriever]["avg_latency_ms"],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare dense, sparse, and hybrid retrieval")
    parser.add_argument(
        "--index_dir", type=str, default="index/squad", help="Directory containing FAISS index"
    )
    parser.add_argument(
        "--queries", type=str, default="data/squad/sample_queries.json", help="Path to queries JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/retriever_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 3, 5, 10], help="K values to evaluate"
    )

    args = parser.parse_args()

    # Setup
    LoggerConfig.setup(level="INFO")

    logger.info("=" * 80)
    logger.info("RETRIEVAL STRATEGY COMPARISON")
    logger.info("=" * 80)

    # Load config
    index_dir = Path(args.index_dir)
    config_path = index_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]

    # Load queries
    queries_path = Path(args.queries)
    queries = load_queries(queries_path)

    # Build retrievers
    dense_retriever, sparse_retriever, hybrid_retriever = build_retrievers(index_dir, model_name)

    # Run comparison
    results = run_retrieval_comparison(
        queries, dense_retriever, sparse_retriever, hybrid_retriever, args.k_values
    )

    # Visualize
    output_dir = Path(args.output_dir)
    visualize_comparison(results, output_dir)

    # Analyze failure cases
    analyze_failure_cases(results, output_dir)

    # Save results
    save_results(results, output_dir / "comparison_results.json")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
