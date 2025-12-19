"""
Test retrieval quality on SQuAD queries.

This script:
1. Loads FAISS index
2. Loads sample queries from SQuAD
3. Performs retrieval for each query
4. Calculates Recall@K metrics
5. Measures latency
6. Visualizes results

Usage:
    python scripts/test_retrieval.py --index_dir index/squad --queries data/squad/sample_queries.json
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
from src.utils import LoggerConfig, Timer, ensure_dir


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """
    Load test queries from JSON file.

    Args:
        queries_path: Path to queries JSON

    Returns:
        List of query dictionaries
    """
    logger.info(f"Loading queries from {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def retrieve_for_queries(
    queries: List[Dict[str, Any]],
    index: FAISSIndex,
    embed_model: EmbeddingModel,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Perform retrieval for all queries and collect results.

    Args:
        queries: List of query dictionaries
        index: FAISS index
        embed_model: Embedding model
        k_values: List of K values to test

    Returns:
        Dictionary with retrieval results and metrics
    """
    logger.info(f"Running retrieval for {len(queries)} queries...")
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {"queries": [], "latencies": [], "max_k": max(k_values)}

    # Extract query texts
    query_texts = [q["query"] for q in queries]

    # Generate embeddings (batched for speed)
    logger.info("Generating query embeddings...")
    with Timer("Query embedding generation"):
        query_embeddings = embed_model.encode(query_texts, show_progress=True)

    logger.info(f"Query embeddings shape: {query_embeddings.shape}")

    # Perform retrieval for each query
    logger.info(f"Performing retrieval (top-{max(k_values)})...")

    for idx, (query_dict, query_emb) in enumerate(zip(queries, query_embeddings)):
        start_time = time.perf_counter()

        # Search
        retrieved = index.search_with_metadata(query_emb.reshape(1, -1), k=max(k_values))[
            0
        ]  # Get first (and only) query's results

        latency = time.perf_counter() - start_time

        # Store results
        results["queries"].append(
            {
                "query_text": query_dict["query"],
                "ground_truth_doc_id": query_dict.get("doc_id", None),
                "has_answer": query_dict.get("has_answer", False),
                "retrieved": retrieved,
                "latency_ms": latency * 1000,
            }
        )

        results["latencies"].append(latency * 1000)

        if (idx + 1) % 20 == 0:
            logger.info(f"Processed {idx + 1}/{len(queries)} queries")

    logger.info(f"Retrieval complete. Avg latency: {np.mean(results['latencies']):.2f}ms")

    return results


def calculate_recall_at_k(
    results: Dict[str, Any],
    k_values: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Calculate Recall@K metrics.

    Recall@K = fraction of queries where ground truth doc appears in top-K.

    Args:
        results: Results from retrieve_for_queries
        k_values: List of K values to evaluate

    Returns:
        Dictionary mapping K to Recall@K
    """
    logger.info("Calculating Recall@K metrics...")
    if k_values is None:
        k_values = [1, 3, 5, 10]

    recall_at_k = {k: 0.0 for k in k_values}
    valid_queries = 0

    for query_result in results["queries"]:
        ground_truth_doc = query_result.get("ground_truth_doc_id")

        if not ground_truth_doc:
            # Skip queries without ground truth
            continue

        valid_queries += 1

        # Extract retrieved doc IDs
        retrieved_docs = [r["doc_id"] for r in query_result["retrieved"]]

        # Check if ground truth appears in top-K for each K
        for k in k_values:
            if ground_truth_doc in retrieved_docs[:k]:
                recall_at_k[k] += 1.0

    # Normalize by number of valid queries
    if valid_queries > 0:
        for k in k_values:
            recall_at_k[k] /= valid_queries

    logger.info(f"Evaluated on {valid_queries} queries with ground truth")
    for k, recall in recall_at_k.items():
        logger.info(f"Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

    return recall_at_k


def calculate_mrr(results: Dict[str, Any]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = average of (1 / rank) where rank is position of first relevant doc.

    Args:
        results: Results from retrieve_for_queries

    Returns:
        MRR score
    """
    logger.info("Calculating MRR...")

    reciprocal_ranks = []

    for query_result in results["queries"]:
        ground_truth_doc = query_result.get("ground_truth_doc_id")

        if not ground_truth_doc:
            continue

        # Find rank of ground truth (1-indexed)
        retrieved_docs = [r["doc_id"] for r in query_result["retrieved"]]

        try:
            rank = retrieved_docs.index(ground_truth_doc) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Ground truth not in retrieved results
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    logger.info(f"MRR: {mrr:.4f}")
    return mrr


def analyze_latency(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze retrieval latency statistics.

    Args:
        results: Results from retrieve_for_queries

    Returns:
        Dictionary with latency statistics
    """
    latencies = results["latencies"]

    stats = {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }

    logger.info("Latency Statistics:")
    logger.info(f"  Mean: {stats['mean_ms']:.2f}ms")
    logger.info(f"  Median: {stats['median_ms']:.2f}ms")
    logger.info(f"  Std: {stats['std_ms']:.2f}ms")
    logger.info(f"  Range: [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}]ms")
    logger.info(f"  P95: {stats['p95_ms']:.2f}ms")
    logger.info(f"  P99: {stats['p99_ms']:.2f}ms")

    return stats


def visualize_results(
    results: Dict[str, Any],
    recall_at_k: Dict[int, float],
    mrr: float,
    latency_stats: Dict[str, float],
    output_dir: Path,
) -> None:
    """
    Create visualizations of retrieval results.

    Args:
        results: Results from retrieve_for_queries
        recall_at_k: Recall@K scores
        mrr: MRR score
        latency_stats: Latency statistics
        output_dir: Directory to save plots
    """
    logger.info("Creating visualizations...")
    ensure_dir(output_dir)

    # Plot 1: Recall@K curve
    plt.figure(figsize=(10, 6))

    k_values = sorted(recall_at_k.keys())
    recall_values = [recall_at_k[k] * 100 for k in k_values]

    plt.plot(k_values, recall_values, marker="o", linewidth=2, markersize=8)
    plt.xlabel("K (Number of Retrieved Documents)", fontsize=12)
    plt.ylabel("Recall@K (%)", fontsize=12)
    plt.title("Retrieval Performance: Recall@K", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Annotate points
    for k, recall in zip(k_values, recall_values):
        plt.annotate(
            f"{recall:.1f}%",
            xy=(k, recall),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "recall_at_k.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'recall_at_k.png'}")
    plt.close()

    # Plot 2: Latency distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(results["latencies"], bins=50, edgecolor="black", alpha=0.7, color="skyblue")
    plt.axvline(
        latency_stats["mean_ms"],
        color="red",
        linestyle="--",
        label=f"Mean: {latency_stats['mean_ms']:.2f}ms",
    )
    plt.axvline(
        latency_stats["median_ms"],
        color="green",
        linestyle="--",
        label=f"Median: {latency_stats['median_ms']:.2f}ms",
    )
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Retrieval Latency Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(results["latencies"], vert=True)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title("Latency Box Plot", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "latency_distribution.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'latency_distribution.png'}")
    plt.close()

    # Plot 3: Top-5 retrieval examples
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, ax in enumerate(axes):
        if idx >= len(results["queries"]):
            break

        query_result = results["queries"][idx]
        query_text = query_result["query_text"]
        retrieved = query_result["retrieved"][:5]

        # Get scores
        scores = [r["score"] for r in retrieved]
        labels = [f"Rank {i+1}" for i in range(len(scores))]

        # Check if ground truth is in results
        ground_truth = query_result.get("ground_truth_doc_id")
        colors = []
        for r in retrieved:
            if ground_truth and r["doc_id"] == ground_truth:
                colors.append("green")  # Ground truth
            else:
                colors.append("skyblue")

        ax.barh(labels, scores, color=colors, edgecolor="black")
        ax.set_xlabel("Similarity Score", fontsize=10)
        ax.set_title(f'Query {idx+1}: "{query_text[:60]}..."', fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add legend
        if "green" in colors:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="green", edgecolor="black", label="Ground Truth"),
                Patch(facecolor="skyblue", edgecolor="black", label="Other"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_examples.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'retrieval_examples.png'}")
    plt.close()

    # Plot 4: Summary metrics
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = {f"Recall@{k}": v * 100 for k, v in recall_at_k.items()}
    metrics["MRR × 100"] = mrr * 100

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    colors_map = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors_map, edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Retrieval Metrics Summary", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=300)
    logger.info(f"Saved: {output_dir / 'metrics_summary.png'}")
    plt.close()


def print_sample_results(results: Dict[str, Any], num_samples: int = 3) -> None:
    """
    Print sample retrieval results for manual inspection.

    Args:
        results: Results from retrieve_for_queries
        num_samples: Number of samples to print
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE RETRIEVAL RESULTS")
    logger.info("=" * 80)

    for idx in range(min(num_samples, len(results["queries"]))):
        query_result = results["queries"][idx]

        logger.info(f"\n--- Query {idx+1} ---")
        logger.info(f"Query: {query_result['query_text']}")
        logger.info(f"Ground Truth Doc: {query_result.get('ground_truth_doc_id', 'N/A')}")
        logger.info(f"Latency: {query_result['latency_ms']:.2f}ms")
        logger.info("\nTop-5 Retrieved:")

        for rank, result in enumerate(query_result["retrieved"][:5], 1):
            is_gt = result["doc_id"] == query_result.get("ground_truth_doc_id")
            marker = "✓ GROUND TRUTH" if is_gt else ""

            logger.info(f"\n  Rank {rank} (score={result['score']:.4f}) {marker}")
            logger.info(f"  Doc ID: {result['doc_id']}")
            content_preview = result["content"][:150] + "..."
            logger.info(f"  Content: {content_preview}")


def save_results(
    results: Dict[str, Any],
    recall_at_k: Dict[int, float],
    mrr: float,
    latency_stats: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Save all results to JSON file.

    Args:
        results: Results from retrieve_for_queries
        recall_at_k: Recall@K scores
        mrr: MRR score
        latency_stats: Latency statistics
        output_path: Path to save JSON
    """
    output = {
        "metrics": {"recall_at_k": recall_at_k, "mrr": mrr, "latency": latency_stats},
        "num_queries": len(results["queries"]),
        "retrieval_results": results["queries"],
    }

    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved detailed results to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Test retrieval quality on SQuAD queries")
    parser.add_argument(
        "--index_dir", type=str, default="index/squad", help="Directory containing FAISS index"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/squad/sample_queries.json",
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/retrieval_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values to evaluate (e.g., 1 3 5 10)",
    )

    args = parser.parse_args()

    # Setup
    LoggerConfig.setup(level="INFO")

    logger.info("=" * 80)
    logger.info("RETRIEVAL QUALITY TEST")
    logger.info("=" * 80)

    # Load index
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.error("Please run: python scripts/build_index.py first")
        return

    logger.info(f"Loading index from {index_dir}")
    index = FAISSIndex.load(str(index_dir))
    logger.info(f"Index loaded: {index.get_stats()}")

    # Load config to get model name
    config_path = index_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    logger.info(f"Loading embedding model: {model_name}")

    # Initialize embedding model
    embed_model = EmbeddingModel(model_name=model_name, device="cuda")

    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error(f"Queries file not found: {queries_path}")
        return

    queries = load_queries(queries_path)

    # Run retrieval
    results = retrieve_for_queries(queries, index, embed_model, args.k_values)

    # Calculate metrics
    recall_at_k = calculate_recall_at_k(results, args.k_values)
    mrr = calculate_mrr(results)
    latency_stats = analyze_latency(results)

    # Visualize
    output_dir = Path(args.output_dir)
    visualize_results(results, recall_at_k, mrr, latency_stats, output_dir)

    # Print samples
    print_sample_results(results, num_samples=3)

    # Save results
    save_results(results, recall_at_k, mrr, latency_stats, output_dir / "results.json")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
