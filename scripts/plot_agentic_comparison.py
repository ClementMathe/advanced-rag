"""
Generate comparison visualizations for agentic vs linear RAG evaluation.

Reads JSON results from scripts/evaluate_agentic.py and produces 3 plots:
1. Bar chart: Metrics comparison (Linear vs Agentic)
2. Histogram: Retry count distribution
3. Scatter: Per-query F1 improvement vs retry count

Usage:
    python scripts/plot_agentic_comparison.py

Requires outputs/agentic_eval/ to contain:
    - comparison.json
    - linear_results.json
    - agentic_results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

INPUT_DIR = Path("outputs/agentic_eval")
OUTPUT_DIR = INPUT_DIR  # Save plots alongside data


def load_results():
    """Load evaluation results from JSON files.

    Returns:
        Tuple of (comparison, linear_results, agentic_results) dicts.
    """
    with open(INPUT_DIR / "comparison.json", "r") as f:
        comparison = json.load(f)

    with open(INPUT_DIR / "linear_results.json", "r") as f:
        linear_results = json.load(f)

    with open(INPUT_DIR / "agentic_results.json", "r") as f:
        agentic_results = json.load(f)

    return comparison, linear_results, agentic_results


def plot_metrics_comparison(comparison: dict, output_dir: Path) -> None:
    """Plot 1: Side-by-side bar chart of Linear vs Agentic metrics.

    Args:
        comparison: Comparison dict with 'linear' and 'agentic' aggregate metrics.
        output_dir: Directory to save the plot.
    """
    metrics = ["exact_match", "f1", "rouge_l", "faithfulness"]
    labels = ["Exact Match", "F1 Score", "ROUGE-L", "Faithfulness"]

    linear_means = [comparison["linear"][m]["mean"] for m in metrics]
    linear_stds = [comparison["linear"][m]["std"] for m in metrics]
    agentic_means = [comparison["agentic"][m]["mean"] for m in metrics]
    agentic_stds = [comparison["agentic"][m]["std"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_linear = ax.bar(
        x - width / 2,
        linear_means,
        width,
        yerr=linear_stds,
        capsize=4,
        label="Linear (Step 6)",
        color="#4A90D9",
        alpha=0.85,
        edgecolor="black",
    )
    bars_agentic = ax.bar(
        x + width / 2,
        agentic_means,
        width,
        yerr=agentic_stds,
        capsize=4,
        label="Agentic (Step 6.5)",
        color="#2ECC71",
        alpha=0.85,
        edgecolor="black",
    )

    # Value labels on bars
    for bars in [bars_linear, bars_agentic]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Delta annotations
    for i, metric in enumerate(metrics):
        delta = comparison["delta"][metric]
        sign = "+" if delta >= 0 else ""
        color = "#27AE60" if delta >= 0 else "#E74C3C"
        ax.annotate(
            f"{sign}{delta:.2%}",
            xy=(x[i] + width / 2, agentic_means[i] + agentic_stds[i] + 0.05),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Generation Quality: Linear vs Agentic Pipeline",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_retry_distribution(agentic_results: list, output_dir: Path) -> None:
    """Plot 2: Histogram of retry counts across queries.

    Args:
        agentic_results: List of per-query agentic result dicts.
        output_dir: Directory to save the plot.
    """
    retry_counts = [r["retry_count"] for r in agentic_results]
    max_retry = max(retry_counts) if retry_counts else 3

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.arange(-0.5, max_retry + 1.5, 1)
    counts, _, bars = ax.hist(
        retry_counts,
        bins=bins,
        color="#9B59B6",
        alpha=0.85,
        edgecolor="black",
        rwidth=0.8,
    )

    # Value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                count + 0.5,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Percentage annotations
    total = len(retry_counts)
    for bar, count in zip(bars, counts):
        if count > 0:
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                count / 2,
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

    ax.set_xlabel("Number of Retries", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title("Retry Distribution (Agentic Pipeline)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(max_retry + 1))
    ax.set_xticklabels([str(i) for i in range(max_retry + 1)])
    ax.grid(True, axis="y", alpha=0.3)

    # Summary stats
    avg_retries = np.mean(retry_counts)
    retry_rate = sum(1 for r in retry_counts if r > 0) / total * 100
    ax.text(
        0.97,
        0.95,
        f"Retry rate: {retry_rate:.0f}%\nAvg retries: {avg_retries:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    path = output_dir / "retry_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_f1_vs_retries(linear_results: list, agentic_results: list, output_dir: Path) -> None:
    """Plot 3: Per-query F1 delta (agentic - linear) vs retry count.

    Shows whether queries that required retries benefited from them.

    Args:
        linear_results: List of per-query linear result dicts.
        agentic_results: List of per-query agentic result dicts.
        output_dir: Directory to save the plot.
    """
    # Build lookup by query_id for linear results
    linear_by_id = {r["query_id"]: r for r in linear_results}

    retry_counts = []
    f1_deltas = []

    for agentic_r in agentic_results:
        qid = agentic_r["query_id"]
        linear_r = linear_by_id.get(qid)
        if linear_r is None:
            continue

        retry = agentic_r["retry_count"]
        delta = agentic_r["metrics"]["f1"] - linear_r["metrics"]["f1"]
        retry_counts.append(retry)
        f1_deltas.append(delta)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Jitter x for visibility
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(retry_counts))
    x_jittered = np.array(retry_counts) + jitter

    # Color by improvement direction
    colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in f1_deltas]

    ax.scatter(x_jittered, f1_deltas, c=colors, alpha=0.6, s=40, edgecolors="black", linewidths=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Mean delta per retry bucket
    max_retry = max(retry_counts) if retry_counts else 0
    for r in range(max_retry + 1):
        bucket = [d for rc, d in zip(retry_counts, f1_deltas) if rc == r]
        if bucket:
            mean_d = np.mean(bucket)
            ax.plot(r, mean_d, marker="D", color="#2C3E50", markersize=10, zorder=5)
            ax.annotate(
                f"avg: {mean_d:+.3f}",
                xy=(r, mean_d),
                xytext=(r + 0.25, mean_d + 0.03),
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Retry Count", fontsize=12)
    ax.set_ylabel("F1 Delta (Agentic - Linear)", fontsize=12)
    ax.set_title(
        "Per-Query F1 Improvement by Retry Count",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(max_retry + 1))
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#27AE60",
            markersize=8,
            label="Agentic better",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#E74C3C",
            markersize=8,
            label="Linear better",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#2C3E50",
            markersize=8,
            label="Bucket mean",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

    plt.tight_layout()
    path = output_dir / "f1_vs_retries.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def main():
    """Load results and generate all 3 comparison plots."""
    logger.info("=" * 60)
    logger.info("AGENTIC RAG - Comparison Visualizations")
    logger.info("=" * 60)

    if not INPUT_DIR.exists():
        logger.error(
            f"Results directory not found: {INPUT_DIR}\n"
            "Run 'python scripts/evaluate_agentic.py' first."
        )
        return

    comparison, linear_results, agentic_results = load_results()
    logger.info(f"Loaded results: {len(linear_results)} linear, " f"{len(agentic_results)} agentic")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(comparison, OUTPUT_DIR)
    plot_retry_distribution(agentic_results, OUTPUT_DIR)
    plot_f1_vs_retries(linear_results, agentic_results, OUTPUT_DIR)

    logger.info("")
    logger.info("All plots saved to outputs/agentic_eval/:")
    logger.info("  - metrics_comparison.png")
    logger.info("  - retry_distribution.png")
    logger.info("  - f1_vs_retries.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
