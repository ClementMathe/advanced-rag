"""
Visualization functions for RAG evaluation results.

Produces publication-quality matplotlib charts for:
- Error distribution (horizontal bar chart)
- Metric comparison across pipeline configurations (grouped bars)

Follows existing project conventions:
- matplotlib only (no seaborn)
- dpi=300, bbox_inches="tight"
- plt.close() after save
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from src.evaluation.error_taxonomy import ErrorAnalysis  # noqa: E402

# ---------------------------------------------------------------------------
# Color palette (consistent with project conventions)
# ---------------------------------------------------------------------------

COLOR_PRIMARY = "#4A90D9"
COLOR_SECONDARY = "#2ECC71"
COLOR_ACCENT = "#E67E22"
COLOR_NEGATIVE = "#E74C3C"
COLOR_NEUTRAL = "#95A5A6"

CATEGORY_COLORS: Dict[str, str] = {
    "correct": "#2ECC71",
    "empty_response": "#95A5A6",
    "retrieval_failure": "#E74C3C",
    "hallucination": "#E67E22",
    "low_context_relevance": "#F39C12",
    "incomplete_answer": "#9B59B6",
    "verbose_answer": "#3498DB",
    "wrong_answer": "#C0392B",
}


# ---------------------------------------------------------------------------
# Error distribution plot
# ---------------------------------------------------------------------------


def plot_error_distribution(
    analysis: ErrorAnalysis,
    output_path: str,
    figsize: tuple = (10, 6),
) -> Path:
    """
    Horizontal bar chart of error category counts, sorted descending.

    Args:
        analysis: ErrorAnalysis from ErrorAnalyzer.analyze().
        output_path: Path to save the figure.
        figsize: Figure dimensions.

    Returns:
        Path to saved figure.
    """
    dist = analysis.error_distribution
    if not dist:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # Sort by count descending
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    colors = [CATEGORY_COLORS.get(cat, COLOR_NEUTRAL) for cat in categories]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(
        f"Error Distribution ({analysis.total_predictions} predictions)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=10,
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Error distribution plot saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Metric comparison plot
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    configs: Dict[str, Dict[str, Any]],
    output_path: str,
    metrics: List[str] | None = None,
    figsize: tuple = (12, 6),
) -> Path:
    """
    Grouped bar chart comparing aggregate metrics across configurations.

    Args:
        configs: Dict of config_name -> aggregate_metrics dict.
            Each metric should be a dict with "mean" key, or a float.
        output_path: Path to save the figure.
        metrics: List of metric names to plot. If None, auto-detects
            common metrics across all configs.
        figsize: Figure dimensions.

    Returns:
        Path to saved figure.
    """
    config_names = list(configs.keys())

    # Auto-detect common metrics
    if metrics is None:
        common_keys: set = set()
        for cfg in configs.values():
            common_keys = (
                common_keys | set(cfg.keys()) if not common_keys else common_keys & set(cfg.keys())
            )
        # Filter to plottable metrics (exclude latency, etc.)
        metrics = sorted(
            k
            for k in common_keys
            if k in {"f1", "exact_match", "rouge_l", "faithfulness", "bert_score"}
        )

    if not metrics:
        metrics = sorted(set().union(*(cfg.keys() for cfg in configs.values())))[:6]

    # Extract values
    values: Dict[str, List[float]] = {name: [] for name in config_names}
    for name in config_names:
        cfg = configs[name]
        for metric in metrics:
            val = cfg.get(metric, 0)
            if isinstance(val, dict):
                val = val.get("mean", 0)
            values[name].append(float(val))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(metrics))
    width = 0.8 / len(config_names)
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_NEGATIVE]

    for i, name in enumerate(config_names):
        offset = (i - len(config_names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values[name],
            width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Metric Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.15, ax.get_ylim()[1] * 1.15))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metric comparison plot saved to {path}")
    return path
