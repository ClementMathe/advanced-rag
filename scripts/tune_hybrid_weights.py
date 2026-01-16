"""
Tune hybrid retriever weights to find optimal balance.

Tests different dense/sparse weight combinations and reports metrics.

Usage:
    python scripts/tune_hybrid_weights.py \
        --index_dir index/squad \
        --queries data/squad/sample_queries.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from loguru import logger

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    calculate_retrieval_metrics,
)
from src.utils import LoggerConfig, ensure_dir


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """Load queries from JSON."""
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_retrievers(index_dir: Path, model_name: str):
    """Build dense and sparse retrievers."""
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load(str(index_dir))

    logger.info(f"Loading embedding model: {model_name}")
    embed_model = EmbeddingModel(model_name=model_name, device="cuda")

    dense_retriever = DenseRetriever(faiss_index, embed_model)

    logger.info("Building BM25 index...")
    bm25_retriever = BM25Retriever(k1=1.2, b=0.75)

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

    return dense_retriever, bm25_retriever


def evaluate_weights(
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
    queries: List[Dict[str, Any]],
    dense_weight: float,
    sparse_weight: float,
) -> Dict[str, float]:
    """
    Evaluate hybrid retriever with specific weights.

    Returns:
        Dictionary with Recall@5 and MRR
    """
    hybrid = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        k_rrf=60,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    query_texts = [q["query"] for q in queries]
    ground_truth = [q.get("doc_id") for q in queries]

    results = hybrid.batch_search(query_texts, k=5, k_retriever=50)

    metrics = calculate_retrieval_metrics(results, ground_truth, k_values=[5])

    return {"recall_5": metrics["recall_at_k"][5], "mrr": metrics["mrr"]}


def main():
    parser = argparse.ArgumentParser(description="Tune hybrid retriever weights")
    parser.add_argument("--index_dir", type=str, default="index/squad")
    parser.add_argument("--queries", type=str, default="data/squad/sample_queries.json")
    parser.add_argument("--output_dir", type=str, default="outputs/weight_tuning")

    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")

    logger.info("=" * 80)
    logger.info("HYBRID WEIGHT TUNING")
    logger.info("=" * 80)

    # Load
    index_dir = Path(args.index_dir)
    config_path = index_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    queries = load_queries(Path(args.queries))
    dense_retriever, bm25_retriever = build_retrievers(index_dir, config["model_name"])

    # Test different weight combinations
    logger.info("\nTesting different weight combinations...")

    weight_configs = [
        (1.0, 0.0),  # Dense only
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),  # Equal
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),  # Sparse only
    ]

    results = []

    for dense_w, sparse_w in weight_configs:
        logger.info(f"Testing: dense={dense_w:.1f}, sparse={sparse_w:.1f}")

        metrics = evaluate_weights(dense_retriever, bm25_retriever, queries, dense_w, sparse_w)

        results.append(
            {
                "dense_weight": dense_w,
                "sparse_weight": sparse_w,
                "recall_5": metrics["recall_5"],
                "mrr": metrics["mrr"],
            }
        )

        logger.info(f"  → Recall@5: {metrics['recall_5']:.4f}, MRR: {metrics['mrr']:.4f}")

    # Find best
    best_recall = max(results, key=lambda x: x["recall_5"])
    best_mrr = max(results, key=lambda x: x["mrr"])

    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATIONS")
    logger.info("=" * 80)
    logger.info(
        f"Best Recall@5: dense={best_recall['dense_weight']:.1f}, "
        f"sparse={best_recall['sparse_weight']:.1f} → {best_recall['recall_5']:.4f}"
    )
    logger.info(
        f"Best MRR: dense={best_mrr['dense_weight']:.1f}, "
        f"sparse={best_mrr['sparse_weight']:.1f} → {best_mrr['mrr']:.4f}"
    )

    # Visualize
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dense_weights = [r["dense_weight"] for r in results]
    recall_values = [r["recall_5"] * 100 for r in results]
    mrr_values = [r["mrr"] * 100 for r in results]

    # Recall@5 plot
    ax1 = axes[0]
    ax1.plot(dense_weights, recall_values, marker="o", linewidth=2, markersize=8, color="skyblue")
    ax1.axhline(best_recall["recall_5"] * 100, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Dense Weight", fontsize=12)
    ax1.set_ylabel("Recall@5 (%)", fontsize=12)
    ax1.set_title("Recall@5 vs Dense Weight", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # MRR plot
    ax2 = axes[1]
    ax2.plot(dense_weights, mrr_values, marker="o", linewidth=2, markersize=8, color="lightcoral")
    ax2.axhline(best_mrr["mrr"] * 100, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Dense Weight", fontsize=12)
    ax2.set_ylabel("MRR × 100", fontsize=12)
    ax2.set_title("MRR vs Dense Weight", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "weight_tuning.png", dpi=300)
    logger.info(f"\nSaved plot: {output_dir / 'weight_tuning.png'}")
    plt.close()

    # Save results
    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results: {output_dir / 'tuning_results.json'}")


if __name__ == "__main__":
    main()
