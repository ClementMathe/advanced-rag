"""
compare_rerankers — Strict Parameter Control

Ensures all configurations use identical retrieval parameters,
enabling a fair apples-to-apples comparison.

Configurations evaluated:
    - Dense retrieval only
    - Hybrid (Dense 0.9 + BM25 0.1, RRF fusion)
    - Hybrid + BGE-reranker-base
    - Hybrid + QNLI-electra-base
    - Hybrid + ms-marco-MiniLM-L6-v2

Usage:
    python scripts/compare_rerankers.py
    python scripts/compare_rerankers.py --queries data/squad/queries_500.json
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from loguru import logger

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.reranker import CrossEncoderReranker
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    calculate_retrieval_metrics,
)
from src.utils import LoggerConfig

# --- Style Configuration ---
# Consistent color palette mapped to each configuration
COLORS = {
    "dense": "#4A90D9",  # Blue
    "hybrid": "#2ECC71",  # Green
    "hybrid_bge-reranker-base": "#E74C3C",  # Red
    "hybrid_qnli-electra": "#9B59B6",  # Purple
    "hybrid_ms-marco-MiniLM": "#F39C12",  # Orange
}

# Human-readable labels for the legend
LABELS = {
    "dense": "Dense",
    "hybrid": "Hybrid (0.9 / 0.1)",
    "hybrid_bge-reranker-base": "Hybrid + BGE-reranker-base",
    "hybrid_qnli-electra": "Hybrid + QNLI-electra",
    "hybrid_ms-marco-MiniLM": "Hybrid + MiniLM-L6-v2",
}
# Marker styles per config for print-friendly differentiation
MARKERS = {
    "dense": "o",
    "hybrid": "s",
    "hybrid_bge-reranker-base": "^",
    "hybrid_qnli-electra": "D",
    "hybrid_ms-marco-MiniLM": "P",
}
GT_COLOR = "#F1C40F"  # Gold for Ground Truth


# ---------------------------------------------------------------------------
# Monkey-patch: expose raw-pair scoring on CrossEncoderReranker
# ---------------------------------------------------------------------------
# The reranker's public API is rerank(query, docs), which loops internally.
# For batch reranking across many queries we need to score all (query, doc)
# pairs in a single pass to fully utilise GPU throughput.
# _compute_scores_from_pairs flattens the batch and delegates to the
# existing _score_batch method in chunks of self.batch_size.
# ---------------------------------------------------------------------------
def _compute_scores_from_pairs(self, pairs):
    """Score a flat list of [query, document] pairs using internal batching."""
    all_scores = []
    for i in range(0, len(pairs), self.batch_size):
        batch = pairs[i : i + self.batch_size]
        all_scores.extend(self._score_batch(batch))
    return all_scores


CrossEncoderReranker._compute_scores_from_pairs = _compute_scores_from_pairs


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
def warm_up(retriever, query_texts, n=5):
    """
    Run dummy searches to initialise CUDA kernels and internal caches.

    Without warm-up, the first few calls pay JIT compilation overhead,
    which skews latency measurements.
    """
    logger.info(f"Warming up ({n} iterations)...")
    for q in query_texts[: min(n, len(query_texts))]:
        _ = retriever.search(q, k=10)


# ---------------------------------------------------------------------------
# Batch reranking
# ---------------------------------------------------------------------------
def batch_rerank(reranker, queries, candidates_list, top_k=10):
    """
    Rerank candidates for multiple queries in a single GPU pass.

    Strategy:
        1. Flatten all (query, document) pairs across every query.
        2. Score the entire flat list via _compute_scores_from_pairs.
        3. Reconstruct per-query results using precomputed offsets.
        4. Sort each query's candidates by descending rerank score.

    Args:
        reranker: CrossEncoderReranker instance.
        queries: List of query strings.
        candidates_list: List of candidate-document lists, one per query.
        top_k: Maximum number of documents to keep per query after sorting.

    Returns:
        List of reranked candidate lists, one per query.
    """
    all_pairs = []
    offsets = [0]  # Cumulative pair count; used to slice scores back per query.

    for query, candidates in zip(queries, candidates_list):
        for cand in candidates:
            all_pairs.append([query, cand.get("content", "")])
        offsets.append(len(all_pairs))

    if not all_pairs:
        return [[] for _ in queries]

    # Single GPU pass over every pair
    all_scores = reranker._compute_scores_from_pairs(all_pairs)

    # Distribute scores back to each query's candidate list
    results = []
    for i in range(len(queries)):
        candidates = candidates_list[i]
        scores = all_scores[offsets[i] : offsets[i + 1]]

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        results.append(reranked[:top_k])

    return results


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def run_comparison(queries, dense_retriever, hybrid_retriever, k_values):
    """
    Execute every configuration and return a results dictionary.

    Fixed parameters (kept constant across all configs for fairness):
        k_final      = max(k_values)   — number of results returned to the user
        k_retriever  = 50              — candidates fetched by dense + sparse before RRF
        k_candidates = 20              — candidates passed to the cross-encoder

    Args:
        queries: List of query dicts loaded from JSON.
        dense_retriever: Initialised DenseRetriever.
        hybrid_retriever: Initialised HybridRetriever.
        k_values: List of K values for Recall@K evaluation (e.g. [1, 5, 10]).

    Returns:
        Ordered dict mapping config name → {metrics, avg_latency_ms}.
    """
    k_final = max(k_values)
    k_retriever = 50
    k_candidates = 20

    query_texts = [q["query"] for q in queries]
    ground_truth = [q.get("doc_id") for q in queries]
    results = {}

    # ------------------------------------------------------------------
    # 1. Dense
    # ------------------------------------------------------------------
    logger.info("Evaluating: DENSE")
    warm_up(dense_retriever, query_texts)

    start = time.perf_counter()
    dense_results = dense_retriever.batch_search(query_texts, k=k_final)
    dense_time_ms = (time.perf_counter() - start) * 1000 / len(queries)

    results["dense"] = {
        "metrics": calculate_retrieval_metrics(dense_results, ground_truth, k_values),
        "avg_latency_ms": round(dense_time_ms, 2),
    }

    # ------------------------------------------------------------------
    # 2. Hybrid
    # ------------------------------------------------------------------
    logger.info("Evaluating: HYBRID")
    warm_up(hybrid_retriever, query_texts)

    start = time.perf_counter()
    # Retrieve k_candidates (20) so the same list feeds the rerankers later.
    hybrid_candidates = hybrid_retriever.batch_search(
        query_texts, k=k_candidates, k_retriever=k_retriever
    )
    hybrid_time_ms = (time.perf_counter() - start) * 1000 / len(queries)

    # For Hybrid-only metrics, truncate to k_final.
    hybrid_eval = [res[:k_final] for res in hybrid_candidates]
    results["hybrid"] = {
        "metrics": calculate_retrieval_metrics(hybrid_eval, ground_truth, k_values),
        "avg_latency_ms": round(hybrid_time_ms, 2),
    }

    # ------------------------------------------------------------------
    # 3. Hybrid + Rerankers
    # ------------------------------------------------------------------
    reranker_models = {
        "hybrid_bge-reranker-base": "BAAI/bge-reranker-base",
        "hybrid_qnli-electra": "cross-encoder/qnli-electra-base",
        "hybrid_ms-marco-MiniLM": "cross-encoder/ms-marco-MiniLM-L6-v2",
    }

    for config_name, model_path in reranker_models.items():
        logger.info(f"Evaluating: {config_name.upper()}")

        reranker = CrossEncoderReranker(model_name=model_path, device="cuda", batch_size=32)

        # Warm up the reranker with a small batch
        _ = batch_rerank(reranker, query_texts[:2], hybrid_candidates[:2], top_k=k_final)

        start = time.perf_counter()
        reranked = batch_rerank(reranker, query_texts, hybrid_candidates, top_k=k_final)
        rerank_time_ms = (time.perf_counter() - start) * 1000 / len(queries)

        # Total latency = retrieval + reranking
        total_latency_ms = round(hybrid_time_ms + rerank_time_ms, 2)

        results[config_name] = {
            "metrics": calculate_retrieval_metrics(reranked, ground_truth, k_values),
            "avg_latency_ms": total_latency_ms,
            "model": model_path,
        }

        del reranker  # Free GPU memory before loading next model

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_table(results):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 95)
    print(f"{'Configuration':<32} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'Latency'}")
    print("-" * 95)

    for name, data in results.items():
        m = data["metrics"]
        r1 = m["recall_at_k"][1] * 100
        r5 = m["recall_at_k"][5] * 100
        r10 = m["recall_at_k"][10] * 100
        label = name.replace("_", " + ").title()
        print(
            f"{label:<32} {r1:>6.1f}% {r5:>6.1f}% {r10:>6.1f}% "
            f"{m['mrr']:>8.3f} {data['avg_latency_ms']:>6.1f}ms"
        )
    print("=" * 95)


def save_results(results, output_dir: Path, num_queries: int):
    """
    Persist results to a JSON file consumable by visualize_reranking.py.

    A _meta block is included so the visualisation script can reconstruct
    context (number of queries, k_values) without external knowledge.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a serialisable copy (convert numpy types if present)
    serialisable = {"_meta": {"num_queries": num_queries}}
    for config_name, data in results.items():
        entry = {
            "metrics": {
                "recall_at_k": {
                    str(k): float(v) for k, v in data["metrics"]["recall_at_k"].items()
                },
                "mrr": float(data["metrics"]["mrr"]),
            },
            "avg_latency_ms": float(data["avg_latency_ms"]),
        }
        if "model" in data:
            entry["model"] = data["model"]
        serialisable[config_name] = entry

    json_path = output_dir / f"results_{num_queries}q.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {json_path}")
    return json_path


def setup_plot_style():
    """Apply a clean, professional matplotlib style globally."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CCCCCC",
            "axes.grid": True,
            "grid.color": "#E8E8E8",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_recall_at_k(results: dict, output_path: Path, num_queries: int):
    """
    Plot Recall@K progression for all configurations.

    Each configuration is a line; X-axis is K (1, 3, 5, 10),
    Y-axis is Recall percentage.

    Args:
        results: Dictionary of {config_name: {"metrics": {...}, ...}}
        output_path: Path to save the PNG
        num_queries: Total number of queries evaluated (for the subtitle)
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name, data in results.items():
        recall_dict = data["metrics"]["recall_at_k"]

        # Sort by K numerically
        k_values = sorted(recall_dict.keys(), key=int)
        k_labels = [str(k) for k in k_values]
        recall_values = [recall_dict[k] * 100 for k in k_values]

        ax.plot(
            k_labels,
            recall_values,
            marker=MARKERS.get(config_name, "o"),
            color=COLORS.get(config_name, "#888888"),
            label=LABELS.get(config_name, config_name),
            linewidth=2.2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=2,
        )

    # Formatting
    ax.set_xlabel("K")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall@K — Retrieval Configurations Comparison")
    ax.set_ylim(40, 102)  # Leave headroom at the top
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Legend outside the plot area to avoid overlap
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Subtitle with evaluation context
    fig.text(
        0.5,
        -0.02,
        f"Evaluated on {num_queries} SQuAD queries | Index: 744 chunks, BGE-large-en-v1.5 embeddings",
        ha="center",
        fontsize=9,
        color="#666666",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_latency(results: dict, output_path: Path, num_queries: int):
    """
    Horizontal bar chart comparing average pipeline latency per configuration.

    Bars are color-coded consistently with the Recall@K chart.
    A vertical reference line marks the Hybrid baseline for quick comparison.

    Args:
        results: Dictionary of {config_name: {"avg_latency_ms": ..., ...}}
        output_path: Path to save the PNG
        num_queries: Total number of queries evaluated (for the subtitle)
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Reverse order so that the first config (Dense) is at the top
    config_names = list(results.keys())[::-1]
    latencies = [results[name]["avg_latency_ms"] for name in config_names]
    colors = [COLORS.get(name, "#888888") for name in config_names]
    labels = [LABELS.get(name, name) for name in config_names]

    bars = ax.barh(
        labels,
        latencies,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        height=0.55,
    )

    # Annotate each bar with its value
    for bar, lat in zip(bars, latencies):
        ax.text(
            bar.get_width() + 2,  # Small offset from bar end
            bar.get_y() + bar.get_height() / 2,
            f"{lat:.1f} ms",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
            fontweight="bold",
        )

    # Reference line: Hybrid baseline
    hybrid_lat = results.get("hybrid", {}).get("avg_latency_ms", None)
    if hybrid_lat:
        ax.axvline(
            hybrid_lat,
            color="#2ECC71",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Hybrid baseline ({hybrid_lat:.1f} ms)",
        )
        ax.legend(loc="lower right", fontsize=9)

    # Formatting
    ax.set_xlabel("Average Latency (ms)")
    ax.set_title("Pipeline Latency — Per-Query Average")
    ax.set_xlim(0, max(latencies) * 1.25)  # Headroom for labels
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.text(
        0.5,
        -0.02,
        f"Evaluated on {num_queries} SQuAD queries | Latency includes retrieval + reranking",
        ha="center",
        fontsize=9,
        color="#666666",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_all_charts(results: dict, output_dir: Path, num_queries: int):
    """
    Entry point: generate all Step 5 charts from a results dictionary.

    Args:
        results: Full output from compare_rerankers.run_comparison()
        output_dir: Directory where PNGs will be saved
        num_queries: Number of queries used in evaluation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts in {output_dir}/")

    plot_recall_at_k(results, output_dir / "step5_recall_at_k.png", num_queries)
    plot_latency(results, output_dir / "step5_latency.png", num_queries)

    print("Done.")


def plot_rank_migration(
    query_text,
    ground_truth_id,
    hybrid_results,
    reranked_results,
    model_label,
    output_path,
    top_n=20,
):
    """
    Generate a Slopegraph with GroundTruth (GT).
    """
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 13))

    # Map  hybride ranks
    hybrid_rank_map = {doc["doc_id"]: i + 1 for i, doc in enumerate(hybrid_results)}

    # 1. Prepare Data
    migration_data = []
    for final_rank_idx, doc in enumerate(reranked_results[:top_n]):
        doc_id = doc["doc_id"]
        is_gt = doc_id == ground_truth_id
        initial_rank = hybrid_rank_map.get(doc_id, 21)  # 21 = out top 20

        migration_data.append(
            {
                "id": doc_id,
                "initial": initial_rank,
                "final": final_rank_idx + 1,
                "content": doc.get("content", "")[:65] + "...",
                "is_gt": is_gt,
            }
        )

    # 2. Plotting (first non-GT, then GT above)
    for is_pass_gt in [False, True]:
        for item in [d for d in migration_data if d["is_gt"] == is_pass_gt]:
            if item["is_gt"]:
                color = GT_COLOR
                linewidth = 4
                alpha = 1.0
                zorder = 10
                label_suffix = " ⭐ [GROUND TRUTH]"
                weight = "bold"
            else:
                color = (
                    "#2ECC71"
                    if item["initial"] > item["final"]
                    else "#E74C3C" if item["initial"] < item["final"] else "#BDC3C7"
                )
                if item["initial"] > 20:
                    color = "#3498DB"  # Nouveau dans le top
                linewidth = 1.5
                alpha = 0.5
                zorder = 2
                label_suffix = ""
                weight = "normal"

            # Ligne
            ax.plot(
                [0, 1],
                [item["initial"], item["final"]],
                marker="o",
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

            # Left Label  (Hybrid)
            ax.text(
                -0.05,
                item["initial"],
                f"R{item['initial'] if item['initial'] <= 20 else '>20'}",
                ha="right",
                va="center",
                fontsize=9,
                fontweight=weight,
                color=color if item["is_gt"] else "black",
            )

            # Right Label (Reranked + Content)
            ax.text(
                1.05,
                item["final"],
                f"R{item['final']} - {item['content']}{label_suffix}",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight=weight,
                color=color if item["is_gt"] else "#333333",
            )

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Initial Hybrid Rank", f"Reranked ({model_label})"], fontweight="bold", fontsize=12
    )
    ax.set_yticks(range(1, 22))
    ax.set_yticklabels([str(i) for i in range(1, 21)] + [">20"])
    ax.invert_yaxis()

    plt.title(f'Rank Migration Analysis\nQuery: "{query_text}"', pad=40, fontsize=13, loc="center")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified reranking comparison with charting")
    parser.add_argument("--index_dir", type=str, default="index/squad")
    parser.add_argument("--queries", type=str, default="data/squad/sample_queries.json")
    parser.add_argument("--output_dir", type=str, default="outputs/compare_rerankers")
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")

    # --- Load index and build retrievers ---
    index_dir = Path(args.index_dir)
    with open(index_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    faiss_index = FAISSIndex.load(str(index_dir))
    embed_model = EmbeddingModel(model_name=config["model_name"], device="cuda")
    dense_retriever = DenseRetriever(faiss_index, embed_model)

    chunks = [
        Chunk(
            content=m["content"],
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            start_char=0,
            end_char=len(m["content"]),
            chunk_index=m["chunk_index"],
            metadata=m["metadata"],
        )
        for m in faiss_index.chunk_metadata
    ]

    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25, dense_weight=0.9, sparse_weight=0.1)

    # --- Load queries ---
    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {args.queries}")

    # --- Run comparison ---
    k_values = [1, 5, 10]
    results = run_comparison(queries, dense_retriever, hybrid_retriever, k_values)

    # --- Print table ---
    print_table(results)

    # --- Save JSON ---
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, len(queries))

    # --- Generate charts ---
    generate_all_charts(results, output_dir, len(queries))

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    query_texts = [q["query"] for q in queries_data]
    logger.info("Retrieving Hybrid candidates...")
    hybrid_candidates = hybrid_retriever.batch_search(query_texts, k=20, k_retriever=50)
    reranker_models = {
        "hybrid_bge-reranker-base": "BAAI/bge-reranker-base",
        "hybrid_qnli-electra": "cross-encoder/qnli-electra-base",
        "hybrid_ms-marco-MiniLM": "cross-encoder/ms-marco-MiniLM-L6-v2",
    }
    target_idx = 0
    q_text = queries_data[target_idx]["query"]
    gt_id = queries_data[target_idx]["doc_id"]

    for config_name, model_path in reranker_models.items():
        logger.info(f"Processing Reranker: {config_name}")
        reranker = CrossEncoderReranker(model_name=model_path, device="cuda", batch_size=32)

        # Rerank batch
        reranked_results = batch_rerank(
            reranker, [q_text], [hybrid_candidates[target_idx]], top_k=10
        )

        # Plot migration for target_query
        plot_rank_migration(
            q_text,
            gt_id,
            hybrid_candidates[target_idx],
            reranked_results[0],
            LABELS[config_name],
            output_dir / f"migration_{config_name}_gt.png",
        )
        del reranker  # Cleanup VRAM
        logger.info(f"Migration plot saved for {config_name}")

    logger.info(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
