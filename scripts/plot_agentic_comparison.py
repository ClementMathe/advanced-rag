"""
Generate comparison visualizations for 3-config agentic RAG ablation.

Reads JSON results from scripts/evaluate_agentic.py and produces 5 plots:
1. Bar chart: 3-config metrics comparison (Linear vs Adaptive vs Adaptive+Web)
2. Histogram: Rerank score distribution colored by fallback trigger
3. Scatter: Per-query F1 delta (adaptive - linear) by fallback group
4. Grouped bars: Fallback impact breakdown (F1 + faithfulness)
5. Box plot: Latency comparison across configs

Usage:
    python scripts/plot_agentic_comparison.py

Requires outputs/agentic_eval/ to contain:
    - ablation_comparison.json
    - linear_results.json
    - adaptive_results.json
    - adaptive_web_results.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.lines import Line2D

INPUT_DIR = Path("outputs/agentic_eval")
OUTPUT_DIR = INPUT_DIR

# Consistent color palette
COLOR_LINEAR = "#4A90D9"
COLOR_ADAPTIVE = "#2ECC71"
COLOR_WEB = "#E67E22"
COLOR_POSITIVE = "#27AE60"
COLOR_NEGATIVE = "#E74C3C"
COLOR_FALLBACK = "#E88D72"
COLOR_NO_FALLBACK = "#5DADE2"


def load_results() -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    """Load evaluation results from JSON files.

    Returns:
        Tuple of (comparison, linear_results, adaptive_results, adaptive_web_results).
    """
    with open(INPUT_DIR / "ablation_comparison.json", "r") as f:
        comparison = json.load(f)

    with open(INPUT_DIR / "linear_results.json", "r") as f:
        linear_results = json.load(f)

    with open(INPUT_DIR / "adaptive_results.json", "r") as f:
        adaptive_results = json.load(f)

    with open(INPUT_DIR / "adaptive_web_results.json", "r") as f:
        adaptive_web_results = json.load(f)

    return comparison, linear_results, adaptive_results, adaptive_web_results


def plot_metrics_comparison(comparison: Dict, output_dir: Path) -> None:
    """Plot 1: 3-config grouped bar chart of F1, ROUGE-L, Faithfulness.

    Args:
        comparison: Ablation comparison dict with configs and deltas.
        output_dir: Directory to save the plot.
    """
    metrics = ["f1", "rouge_l", "faithfulness"]
    labels = ["F1 Score", "ROUGE-L", "Faithfulness"]
    configs = ["linear", "adaptive", "adaptive_web"]
    config_labels = ["Linear", "Adaptive", "Adaptive+Web"]
    colors = [COLOR_LINEAR, COLOR_ADAPTIVE, COLOR_WEB]

    means = {}
    stds = {}
    for cfg in configs:
        means[cfg] = [comparison["configs"][cfg]["metrics"][m]["mean"] for m in metrics]
        stds[cfg] = [comparison["configs"][cfg]["metrics"][m]["std"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            means[cfg],
            width,
            yerr=stds[cfg],
            capsize=3,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Delta annotations for adaptive vs linear
    deltas = comparison["deltas_vs_linear"]
    for i, metric in enumerate(metrics):
        for j, cfg in enumerate(["adaptive", "adaptive_web"]):
            delta = deltas[cfg][metric]
            sign = "+" if delta >= 0 else ""
            color = COLOR_POSITIVE if delta >= 0 else COLOR_NEGATIVE
            bar_x = x[i] + (j) * width  # offset for adaptive (j=0→0), web (j=1→width)
            bar_y = means[cfg][i] + stds[cfg][i] + 0.04
            ax.annotate(
                f"{sign}{delta:.1%}",
                xy=(bar_x, bar_y),
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=color,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Generation Quality: Linear vs Adaptive vs Adaptive+Web",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_rerank_score_distribution(adaptive_results: List[Dict], output_dir: Path) -> None:
    """Plot 2: Histogram of rerank scores colored by fallback trigger.

    Args:
        adaptive_results: Per-query adaptive results with min_rerank_score.
        output_dir: Directory to save the plot.
    """
    fallback_scores = [
        r["min_rerank_score"] for r in adaptive_results if r["used_fallback_retrieval"]
    ]
    no_fallback_scores = [
        r["min_rerank_score"] for r in adaptive_results if not r["used_fallback_retrieval"]
    ]
    all_scores = [r["min_rerank_score"] for r in adaptive_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(-11, 7, 1)

    ax.hist(
        fallback_scores,
        bins=bins,
        color=COLOR_FALLBACK,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label=f"Fallback triggered (n={len(fallback_scores)})",
    )
    ax.hist(
        no_fallback_scores,
        bins=bins,
        color=COLOR_NO_FALLBACK,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label=f"No fallback (n={len(no_fallback_scores)})",
        bottom=np.histogram(fallback_scores, bins=bins)[0],
    )

    # Threshold line
    ax.axvline(
        x=0.0,
        color="#2C3E50",
        linestyle="--",
        linewidth=2,
        label="Threshold = 0.0",
    )

    # Stats box
    median = float(np.median(all_scores))
    mean = float(np.mean(all_scores))
    stats_text = (
        f"Mean: {mean:.2f}\n"
        f"Median: {median:.2f}\n"
        f"Range: [{min(all_scores):.1f}, {max(all_scores):.1f}]\n"
        f"Fallback rate: {len(fallback_scores)/len(all_scores):.0%}"
    )
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Min Rerank Score (per query)", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title(
        "Rerank Score Distribution & Fallback Trigger",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "rerank_score_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_f1_delta_by_fallback(
    linear_results: List[Dict],
    adaptive_results: List[Dict],
    output_dir: Path,
) -> None:
    """Plot 3: Per-query F1 delta scatter grouped by fallback trigger.

    Args:
        linear_results: Per-query linear baseline results.
        adaptive_results: Per-query adaptive results.
        output_dir: Directory to save the plot.
    """
    linear_by_id = {r["query_id"]: r for r in linear_results}

    fallback_deltas = []
    no_fallback_deltas = []

    for ar in adaptive_results:
        lr = linear_by_id.get(ar["query_id"])
        if lr is None:
            continue
        delta = ar["metrics"]["f1"] - lr["metrics"]["f1"]
        if ar["used_fallback_retrieval"]:
            fallback_deltas.append(delta)
        else:
            no_fallback_deltas.append(delta)

    fig, ax = plt.subplots(figsize=(9, 6))

    rng = np.random.default_rng(42)

    # No-fallback group (x=0)
    jitter_nf = rng.uniform(-0.15, 0.15, len(no_fallback_deltas))
    colors_nf = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in no_fallback_deltas]
    ax.scatter(
        0 + jitter_nf,
        no_fallback_deltas,
        c=colors_nf,
        alpha=0.6,
        s=40,
        edgecolors="black",
        linewidths=0.3,
    )

    # Fallback group (x=1)
    jitter_fb = rng.uniform(-0.15, 0.15, len(fallback_deltas))
    colors_fb = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in fallback_deltas]
    ax.scatter(
        1 + jitter_fb,
        fallback_deltas,
        c=colors_fb,
        alpha=0.6,
        s=40,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Group means
    for i, (group, _label) in enumerate(
        [(no_fallback_deltas, "No Fallback"), (fallback_deltas, "Fallback")]
    ):
        if group:
            mean_d = float(np.mean(group))
            ax.plot(i, mean_d, marker="D", color="#2C3E50", markersize=12, zorder=5)
            ax.annotate(
                f"mean: {mean_d:+.4f}",
                xy=(i, mean_d),
                xytext=(i + 0.25, mean_d + 0.02),
                fontsize=10,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2),
            )

    # Count annotations
    n_improved = sum(1 for d in fallback_deltas if d > 0)
    n_degraded = sum(1 for d in fallback_deltas if d < 0)
    n_same = sum(1 for d in fallback_deltas if d == 0)
    ax.text(
        0.97,
        0.95,
        f"Fallback queries (n={len(fallback_deltas)}):\n"
        f"  Improved: {n_improved}\n"
        f"  Same: {n_same}\n"
        f"  Degraded: {n_degraded}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"No Fallback\n(n={len(no_fallback_deltas)})", f"Fallback\n(n={len(fallback_deltas)})"],
        fontsize=11,
    )
    ax.set_ylabel("F1 Delta (Adaptive - Linear)", fontsize=12)
    ax.set_title(
        "Per-Query F1 Impact: Adaptive vs Linear",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, 1.5)
    ax.grid(True, axis="y", alpha=0.3)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_POSITIVE,
            markersize=8,
            label="Improved",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_NEGATIVE,
            markersize=8,
            label="Degraded",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#2C3E50",
            markersize=8,
            label="Group mean",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    plt.tight_layout()
    path = output_dir / "f1_delta_by_fallback.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_fallback_impact(
    linear_results: List[Dict],
    adaptive_results: List[Dict],
    output_dir: Path,
) -> None:
    """Plot 4: Fallback impact breakdown — F1 and Faithfulness by query group.

    Args:
        linear_results: Per-query linear baseline results.
        adaptive_results: Per-query adaptive results.
        output_dir: Directory to save the plot.
    """
    linear_by_id = {r["query_id"]: r for r in linear_results}

    # Split queries by fallback trigger
    fallback_ids = {r["query_id"] for r in adaptive_results if r["used_fallback_retrieval"]}
    no_fallback_ids = {r["query_id"] for r in adaptive_results if not r["used_fallback_retrieval"]}

    groups = {
        "No Fallback": no_fallback_ids,
        "Fallback": fallback_ids,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, metric, metric_label in zip(axes, ["f1", "faithfulness"], ["F1 Score", "Faithfulness"]):
        x = np.arange(len(groups))
        width = 0.3

        linear_means = []
        adaptive_means = []

        for _group_name, qids in groups.items():
            lin_scores = [
                linear_by_id[qid]["metrics"][metric] for qid in qids if qid in linear_by_id
            ]
            adp_scores = [r["metrics"][metric] for r in adaptive_results if r["query_id"] in qids]
            linear_means.append(float(np.mean(lin_scores)) if lin_scores else 0)
            adaptive_means.append(float(np.mean(adp_scores)) if adp_scores else 0)

        bars_lin = ax.bar(
            x - width / 2,
            linear_means,
            width,
            label="Linear",
            color=COLOR_LINEAR,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_adp = ax.bar(
            x + width / 2,
            adaptive_means,
            width,
            label="Adaptive",
            color=COLOR_ADAPTIVE,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        # Value labels
        for bars in [bars_lin, bars_adp]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        # Delta annotations
        for i in range(len(groups)):
            delta = adaptive_means[i] - linear_means[i]
            sign = "+" if delta >= 0 else ""
            color = COLOR_POSITIVE if delta >= 0 else COLOR_NEGATIVE
            ax.annotate(
                f"{sign}{delta:.3f}",
                xy=(x[i] + width / 2, adaptive_means[i] + 0.04),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{name}\n(n={len(qids)})" for name, qids in groups.items()],
            fontsize=10,
        )
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"{metric_label} by Query Group", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Fallback Impact: Does Adaptive Retrieval Help?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "fallback_impact.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_latency_comparison(
    linear_results: List[Dict],
    adaptive_results: List[Dict],
    adaptive_web_results: List[Dict],
    output_dir: Path,
) -> None:
    """Plot 5: Latency box plot across 3 configs with web search annotation.

    Args:
        linear_results: Per-query linear baseline results.
        adaptive_results: Per-query adaptive results.
        adaptive_web_results: Per-query adaptive+web results.
        output_dir: Directory to save the plot.
    """
    lin_lat = [r["latency_s"] for r in linear_results]
    adp_lat = [r["latency_s"] for r in adaptive_results]
    web_lat = [r["latency_s"] for r in adaptive_web_results]

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        [lin_lat, adp_lat, web_lat],
        tick_labels=["Linear", "Adaptive", "Adaptive+Web"],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=7),
    )

    colors = [COLOR_LINEAR, COLOR_ADAPTIVE, COLOR_WEB]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Mean annotations
    for i, (data, _label) in enumerate(
        [(lin_lat, "Linear"), (adp_lat, "Adaptive"), (web_lat, "Adpt+Web")]
    ):
        mean_val = float(np.mean(data))
        ax.text(
            i + 1,
            mean_val + 0.3,
            f"{mean_val:.1f}s",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Web search stats
    web_triggered = [r for r in adaptive_web_results if r.get("used_web_search", False)]
    web_not_triggered = [r for r in adaptive_web_results if not r.get("used_web_search", False)]
    web_rate = len(web_triggered) / len(adaptive_web_results) if adaptive_web_results else 0

    if web_triggered and web_not_triggered:
        web_overhead = float(np.mean([r["latency_s"] for r in web_triggered])) - float(
            np.mean([r["latency_s"] for r in web_not_triggered])
        )
        stats_text = (
            f"Web search rate: {web_rate:.0%}\n"
            f"Web overhead: +{web_overhead:.1f}s\n"
            f"(per triggered query)"
        )
    else:
        stats_text = f"Web search rate: {web_rate:.0%}"

    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_title(
        "Latency Comparison Across Configurations",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "latency_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def main():
    """Load results and generate all 5 comparison plots."""
    logger.info("=" * 60)
    logger.info("AGENTIC RAG - 3-Config Ablation Visualizations")
    logger.info("=" * 60)

    if not INPUT_DIR.exists():
        logger.error(
            f"Results directory not found: {INPUT_DIR}\n"
            "Run 'python scripts/evaluate_agentic.py' first."
        )
        return

    comparison, linear_results, adaptive_results, adaptive_web_results = load_results()
    logger.info(
        f"Loaded: {len(linear_results)} linear, "
        f"{len(adaptive_results)} adaptive, "
        f"{len(adaptive_web_results)} adaptive+web"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(comparison, OUTPUT_DIR)
    plot_rerank_score_distribution(adaptive_results, OUTPUT_DIR)
    plot_f1_delta_by_fallback(linear_results, adaptive_results, OUTPUT_DIR)
    plot_fallback_impact(linear_results, adaptive_results, OUTPUT_DIR)
    plot_latency_comparison(linear_results, adaptive_results, adaptive_web_results, OUTPUT_DIR)

    logger.info("")
    logger.info(f"All plots saved to {OUTPUT_DIR}/:")
    logger.info("  - metrics_comparison.png       (3-config metrics bar chart)")
    logger.info("  - rerank_score_distribution.png (score histogram + fallback)")
    logger.info("  - f1_delta_by_fallback.png      (per-query F1 delta scatter)")
    logger.info("  - fallback_impact.png           (F1 + faithfulness breakdown)")
    logger.info("  - latency_comparison.png        (latency box plot)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
