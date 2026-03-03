"""
Phase 6: HotpotQA benchmark — 200 questions, 3 configs.

Requires: scripts/prepare_hotpot.py must be run first to build the index.

Three configs:
  A: Linear pipeline (hybrid + reranker + Qwen, no grading)
  B: Adaptive retrieval (rerank score threshold, from Step 6.5)
  C: Agentic + Mistral grader (function calling) + Qwen generator

HotpotQA-specific metrics (beyond standard F1/EM):
  - Supporting Facts Recall: fraction of gold supporting paragraphs in final context
  - Grading Accuracy (Config C): % of queries where grader correctly includes both
    gold supporting paragraphs

Outputs:
  outputs/hotpot_eval/results_linear.json
  outputs/hotpot_eval/results_adaptive.json
  outputs/hotpot_eval/results_mistral_grader.json
  outputs/hotpot_eval/summary.json
  outputs/hotpot_eval/plots/metrics_comparison.png
  outputs/hotpot_eval/plots/supporting_facts_recall.png
  outputs/hotpot_eval/plots/squad_vs_hotpot_comparison.png
  outputs/hotpot_eval/plots/grading_accuracy_breakdown.png

Usage:
  python scripts/evaluate_hotpot.py [--num-questions N] [--configs A,B,C]
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from mistralai import Mistral
from tqdm import tqdm

from src.agentic_pipeline import AgenticRAGPipeline
from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.evaluation.ab_testing import ABTestRunner
from src.evaluation.metrics import MetricsCalculator
from src.generator import LLMGenerator
from src.mistral_generator import MistralAPIGenerator
from src.mistral_grader import (
    MistralAnswerGrader,
    MistralDocumentGrader,
    MistralQueryRewriter,
    RateLimiter,
)
from src.pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import LoggerConfig, ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOTPOT_INDEX_DIR = "index/hotpot"
GT_PATH = Path("data/hotpot/sample_200q.json")
OUTPUT_DIR = Path("outputs/hotpot_eval")
SQUAD_SUMMARY_PATH = Path("outputs/mistral_grader_eval/summary.json")
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5
RETRIEVAL_THRESHOLD = -1.0  # Config B: adaptive retrieval threshold


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------


def load_components():
    """Load HotpotQA retriever + reranker + Qwen generator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load Qwen FIRST while the GPU is completely clean — avoids OOM from
    # VRAM fragmentation that occurs when loading a large 4-bit model after
    # smaller models (embed_model + reranker) have already allocated GPU memory.
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        device=device,
        temperature=0.1,
        max_new_tokens=80,
    )

    faiss_index = FAISSIndex.load(HOTPOT_INDEX_DIR)

    chunks = [
        Chunk(
            content=meta["content"],
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            start_char=0,
            end_char=len(meta["content"]),
            chunk_index=meta["chunk_index"],
            metadata=meta.get("metadata", {}),
        )
        for meta in faiss_index.chunk_metadata
    ]

    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    embed_model = EmbeddingModel(model_name="BAAI/bge-large-en-v1.5", device=device)
    dense_retriever = DenseRetriever(faiss_index, embed_model)
    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever, k_rrf=60, dense_weight=0.9, sparse_weight=0.1
    )

    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=device)

    return hybrid_retriever, reranker, generator, embed_model, device


# ---------------------------------------------------------------------------
# HotpotQA-specific metrics
# ---------------------------------------------------------------------------


def compute_supporting_facts_recall(
    context_docs: List[Dict],
    supporting_facts: List[tuple],
) -> float:
    """
    Compute fraction of gold supporting paragraphs present in context.

    Args:
        context_docs: Final context documents sent to generator.
            Each has a 'doc_id' key matching the paragraph title.
        supporting_facts: List of (title, sent_id) from ground truth.

    Returns:
        Recall in [0, 1]. 1.0 if all gold paragraphs are in context.
    """
    if not supporting_facts:
        return 1.0

    gold_titles = set(title for title, _ in supporting_facts)
    context_titles = set(doc.get("doc_id", "") for doc in context_docs)
    matched = gold_titles & context_titles
    return len(matched) / len(gold_titles)


def is_grading_accurate(
    context_docs: List[Dict],
    supporting_facts: List[tuple],
    min_gold_present: int = 2,
) -> bool:
    """Check if both gold supporting paragraphs appear in context."""
    gold_titles = set(title for title, _ in supporting_facts)
    context_titles = set(doc.get("doc_id", "") for doc in context_docs)
    matched = gold_titles & context_titles
    return len(matched) >= min(min_gold_present, len(gold_titles))


# ---------------------------------------------------------------------------
# Query runners
# ---------------------------------------------------------------------------


def run_pipeline_query(
    pipeline: RAGPipeline,
    query: str,
    gt: str,
    supporting_facts: List[tuple],
    metrics: MetricsCalculator,
    is_agentic: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    # AgenticRAGPipeline.query() only accepts (query: str) — no return_intermediate
    if is_agentic:
        result = pipeline.query(query)
    else:
        result = pipeline.query(query, return_intermediate=True)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "")

    # Get context docs (with doc_id for supporting facts recall)
    if is_agentic:
        context_docs = result.get("context_documents", [])
    else:
        context_docs = result.get("reranked_chunks", [])

    context_texts = [d.get("content", "") for d in context_docs[:5]]
    sf_recall = compute_supporting_facts_recall(context_docs[:5], supporting_facts)

    f1 = metrics.f1_score(answer, gt)
    em = metrics.exact_match(answer, gt)

    return {
        "answer": answer,
        "ground_truth": gt,
        "f1": f1,
        "em": em,
        "rouge_l": metrics.rouge_l(answer, gt),
        "faithfulness": metrics.faithfulness(answer, context_texts),
        "bertscore": (
            metrics.bert_score(answer, gt)
            if (metrics._use_bertscore and answer.strip() and gt.strip())
            else 0.0
        ),
        "supporting_facts_recall": sf_recall,
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
    }


def run_mistral_grader_query(
    pipeline: AgenticRAGPipeline,
    query: str,
    gt: str,
    supporting_facts: List[tuple],
    metrics: MetricsCalculator,
) -> Dict[str, Any]:
    t0 = time.time()
    result = pipeline.query(query)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "")
    context_docs = result.get("context_documents", [])
    context_texts = [d.get("content", "") for d in context_docs[:5]]

    sf_recall = compute_supporting_facts_recall(context_docs[:5], supporting_facts)
    grading_accurate = is_grading_accurate(context_docs[:5], supporting_facts)

    f1 = metrics.f1_score(answer, gt)
    em = metrics.exact_match(answer, gt)

    return {
        "answer": answer,
        "ground_truth": gt,
        "f1": f1,
        "em": em,
        "rouge_l": metrics.rouge_l(answer, gt),
        "faithfulness": metrics.faithfulness(answer, context_texts),
        "bertscore": (
            metrics.bert_score(answer, gt)
            if (metrics._use_bertscore and answer.strip() and gt.strip())
            else 0.0
        ),
        "supporting_facts_recall": sf_recall,
        "grading_accurate": grading_accurate,
        "num_docs_graded": result.get("num_docs_graded", -1),
        "retry_count": result.get("retry_count", 0),
        "answer_is_acceptable": result.get("answer_is_acceptable", True),
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_partial(path: Path) -> List[Dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def save_partial(results: List[Dict], path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    metrics_keys = [
        "f1",
        "em",
        "rouge_l",
        "faithfulness",
        "bertscore",
        "supporting_facts_recall",
        "latency_ms",
    ]
    agg = {"n": len(results)}
    for m in metrics_keys:
        vals = [r[m] for r in results if m in r]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))

    # Config C specific
    if any("grading_accurate" in r for r in results):
        agg["grading_accuracy"] = float(
            np.mean([r.get("grading_accurate", False) for r in results])
        )
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(summaries: Dict[str, Dict], output_dir: Path):
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)

    configs = list(summaries.keys())
    colors = {
        "linear": "#4C72B0",
        "adaptive": "#DD8452",
        "mistral_grader": "#55A868",
        "mistral_answer_grader": "#9467BD",
        "linear_mistral": "#C44E52",
        "doc_grader_v2_qwen": "#8172B2",
        "doc_grader_v2_mistral": "#937860",
    }
    cfg_colors = [colors.get(c, "#888") for c in configs]

    # 1. EM/F1 comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    metric_keys = ["em_mean", "f1_mean", "bertscore_mean"]
    labels = ["Exact Match", "F1", "BERTScore"]
    x = np.arange(len(labels))
    width = 0.25
    for i, (cfg, color) in enumerate(zip(configs, cfg_colors)):
        vals = [summaries[cfg].get(k, 0.0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=cfg, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("HotpotQA Metrics: Linear vs Adaptive vs MistralGrader")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Supporting facts recall
    fig, ax = plt.subplots(figsize=(7, 5))
    sf_vals = [summaries[c].get("supporting_facts_recall_mean", 0.0) for c in configs]
    bars = ax.bar(configs, sf_vals, color=cfg_colors, alpha=0.85)
    for bar, val in zip(bars, sf_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Supporting Facts Recall")
    ax.set_title(
        "Supporting Facts Recall by Config\n(fraction of gold paragraphs in final context)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "supporting_facts_recall.png", dpi=150)
    plt.close(fig)

    # 3. SQuAD vs HotpotQA comparison (if SQuAD summary available)
    if SQUAD_SUMMARY_PATH.exists():
        with open(SQUAD_SUMMARY_PATH, encoding="utf-8") as f:
            squad_data = json.load(f)
        squad_summaries = squad_data.get("summaries", {})

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metric = "f1_mean"

        for ax, (dataset_name, dataset_summaries) in zip(
            axes,
            [("SQuAD (100q)", squad_summaries), ("HotpotQA (200q)", summaries)],
        ):
            cfg_names = list(dataset_summaries.keys())
            vals = [dataset_summaries[c].get(metric, 0.0) for c in cfg_names]
            cfg_col = [colors.get(c, "#888") for c in cfg_names]
            ax.bar(cfg_names, vals, color=cfg_col, alpha=0.85)
            ax.set_title(f"F1 on {dataset_name}")
            ax.set_ylim(0, 0.7)
            ax.set_ylabel("F1 Score")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(ax.patches, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.suptitle("Cross-Dataset Comparison: SQuAD vs HotpotQA")
        fig.tight_layout()
        fig.savefig(plots_dir / "squad_vs_hotpot_comparison.png", dpi=150)
        plt.close(fig)

    # 4. Grading accuracy breakdown (Config C only)
    if "mistral_grader" in summaries:
        mg = summaries["mistral_grader"]
        grading_acc = mg.get("grading_accuracy", 0.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        cats = ["Both gold\nparagraphs found", "Gold paragraphs\nmissed"]
        vals = [grading_acc * 100, (1 - grading_acc) * 100]
        ax.bar(cats, vals, color=["#55A868", "#C44E52"], alpha=0.85)
        for bar, val in zip(ax.patches, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        ax.set_ylim(0, 110)
        ax.set_ylabel("% of queries")
        ax.set_title(
            "Grading Accuracy — MistralGrader on HotpotQA\n"
            "(did grader include both gold supporting paragraphs?)"
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "grading_accuracy_breakdown.png", dpi=150)
        plt.close(fig)

    logger.info(f"Saved plots to {plots_dir}")


# ---------------------------------------------------------------------------
# A/B testing helper
# ---------------------------------------------------------------------------


def _align_by_query(results_a: list, results_b: list):
    """Align two per-query result lists by the 'question_id' or 'query' key."""
    key = "question_id" if results_a and "question_id" in results_a[0] else "query"
    b_map = {r[key]: r for r in results_b}
    aligned_a, aligned_b = [], []
    for r in results_a:
        if r[key] in b_map:
            aligned_a.append(r)
            aligned_b.append(b_map[r[key]])
    return aligned_a, aligned_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HotpotQA 3-config benchmark")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument(
        "--configs",
        default="linear,adaptive,mistral_grader,mistral_answer_grader,linear_mistral,doc_grader_v2_qwen,doc_grader_v2_mistral",
        help="Comma-separated list of configs to run",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")
    ensure_dir(OUTPUT_DIR)
    requested_configs = [c.strip() for c in args.configs.split(",")]

    logger.info("=" * 60)
    logger.info("HotpotQA Evaluation — Phase 6")
    logger.info("=" * 60)

    # Load ground truth
    if not GT_PATH.exists():
        logger.error(f"Ground truth not found at {GT_PATH}. Run prepare_hotpot.py first.")
        return
    with open(GT_PATH, encoding="utf-8") as f:
        gt_records = json.load(f)
    gt_records = gt_records[: args.num_questions]
    logger.info(f"Loaded {len(gt_records)} ground truth records")

    # Load components
    logger.info("Loading components...")
    hybrid_retriever, reranker, generator, embed_model, device = load_components()
    metrics_calc = MetricsCalculator(embed_model=embed_model, use_bertscore=True)

    all_summaries = {}

    # -----------------------------------------------------------------------
    # Config A: Linear
    # -----------------------------------------------------------------------
    if "linear" in requested_configs:
        logger.info("\n[Config A] Linear pipeline")
        results_path = OUTPUT_DIR / "results_linear.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        linear_pipeline = RAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="linear"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_pipeline_query(
                    linear_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                    is_agentic=False,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.debug(f"linear | F1={r['f1']:.3f} SF={r['supporting_facts_recall']:.3f}")
            except Exception as e:
                logger.error(f"linear error on {qid}: {e}")

        all_summaries["linear"] = aggregate(results)
        logger.info(
            f"linear: F1={all_summaries['linear'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['linear'].get('supporting_facts_recall_mean', 0):.3f}"
        )

    # -----------------------------------------------------------------------
    # Config B: Adaptive retrieval
    # -----------------------------------------------------------------------
    if "adaptive" in requested_configs:
        logger.info("\n[Config B] Adaptive retrieval")
        results_path = OUTPUT_DIR / "results_adaptive.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        adaptive_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            enable_adaptive_retrieval=True,
            retrieval_threshold=RETRIEVAL_THRESHOLD,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="adaptive"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_pipeline_query(
                    adaptive_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                    is_agentic=True,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.debug(f"adaptive | F1={r['f1']:.3f} SF={r['supporting_facts_recall']:.3f}")
            except Exception as e:
                logger.error(f"adaptive error on {qid}: {e}")

        all_summaries["adaptive"] = aggregate(results)
        logger.info(
            f"adaptive: F1={all_summaries['adaptive'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['adaptive'].get('supporting_facts_recall_mean', 0):.3f}"
        )

    # -----------------------------------------------------------------------
    # Config C: Agentic + Mistral grader
    # -----------------------------------------------------------------------
    if "mistral_grader" in requested_configs:
        logger.info("\n[Config C] Agentic + MistralGrader")
        results_path = OUTPUT_DIR / "results_mistral_grader.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        mistral_model = "mistral-small-latest"
        mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        rate_limiter = RateLimiter(calls_per_second=0.5)

        mistral_doc_grader = MistralDocumentGrader(
            client=mistral_client, model_name=mistral_model, rate_limiter=rate_limiter
        )
        mistral_answer_grader = MistralAnswerGrader(
            client=mistral_client, model_name=mistral_model, rate_limiter=rate_limiter
        )
        mistral_query_rewriter = MistralQueryRewriter(
            client=mistral_client, model_name=mistral_model, rate_limiter=rate_limiter
        )

        mistral_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=mistral_answer_grader,
            doc_grader=mistral_doc_grader,
            query_rewriter=mistral_query_rewriter,
            enable_answer_grading=True,
            enable_doc_grading=True,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
            max_retries=1,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="mistral_grader"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_mistral_grader_query(
                    mistral_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"mistral_grader | F1={r['f1']:.3f} "
                    f"SF={r['supporting_facts_recall']:.3f} "
                    f"grading_ok={r['grading_accurate']}"
                )
            except Exception as e:
                logger.error(f"mistral_grader error on {qid}: {e}")

        all_summaries["mistral_grader"] = aggregate(results)
        logger.info(
            f"mistral_grader: F1={all_summaries['mistral_grader'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['mistral_grader'].get('supporting_facts_recall_mean', 0):.3f} "
            f"grading_accuracy={all_summaries['mistral_grader'].get('grading_accuracy', 0):.2%}"
        )

    # -----------------------------------------------------------------------
    # Config D: Mistral answer grader (answer grading only, no doc filtering)
    # -----------------------------------------------------------------------
    if "mistral_answer_grader" in requested_configs:
        logger.info("\n[Config D] Mistral answer grader (answer grading only)")
        results_path = OUTPUT_DIR / "results_mistral_answer_grader.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        mistral_model = "mistral-small-latest"
        mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        rate_limiter = RateLimiter(calls_per_second=0.5)

        answer_grader_inst = MistralAnswerGrader(
            client=mistral_client, model_name=mistral_model, rate_limiter=rate_limiter
        )

        answer_grader_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader_inst,
            enable_answer_grading=True,
            enable_doc_grading=False,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
            max_retries=1,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="mistral_answer_grader"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_pipeline_query(
                    answer_grader_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                    is_agentic=True,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"mistral_answer_grader | F1={r['f1']:.3f} "
                    f"SF={r['supporting_facts_recall']:.3f}"
                )
            except Exception as e:
                logger.error(f"mistral_answer_grader error on {qid}: {e}")

        all_summaries["mistral_answer_grader"] = aggregate(results)
        logger.info(
            f"mistral_answer_grader: F1={all_summaries['mistral_answer_grader'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['mistral_answer_grader'].get('supporting_facts_recall_mean', 0):.3f}"
        )

    # -----------------------------------------------------------------------
    # Config E: Linear + Mistral-small generator (generator effect only)
    # -----------------------------------------------------------------------
    if "linear_mistral" in requested_configs:
        logger.info("\n[Config E] Linear pipeline + Mistral-small generator")
        results_path = OUTPUT_DIR / "results_linear_mistral.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        mistral_gen = MistralAPIGenerator(
            model_name="mistral-small-latest",
            api_key=os.environ.get("MISTRAL_API_KEY"),
            max_new_tokens=150,
            temperature=0.1,
        )

        linear_mistral_pipeline = RAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=mistral_gen,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="linear_mistral"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_pipeline_query(
                    linear_mistral_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                    is_agentic=False,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"linear_mistral | F1={r['f1']:.3f} " f"SF={r['supporting_facts_recall']:.3f}"
                )
            except Exception as e:
                logger.error(f"linear_mistral error on {qid}: {e}")

        all_summaries["linear_mistral"] = aggregate(results)
        logger.info(
            f"linear_mistral: F1={all_summaries['linear_mistral'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['linear_mistral'].get('supporting_facts_recall_mean', 0):.3f}"
        )

    # -----------------------------------------------------------------------
    # Config F: Doc grader v2 (multi-hop prompt) + Qwen generator
    # -----------------------------------------------------------------------
    if "doc_grader_v2_qwen" in requested_configs:
        logger.info("\n[Config F] Multi-hop doc grader v2 + Qwen generator")
        results_path = OUTPUT_DIR / "results_doc_grader_v2_qwen.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        mistral_model = "mistral-small-latest"
        mistral_client_f = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        rate_limiter_f = RateLimiter(calls_per_second=0.5)

        doc_grader_v2 = MistralDocumentGrader(
            client=mistral_client_f,
            model_name=mistral_model,
            multi_hop=True,
            rate_limiter=rate_limiter_f,
        )
        answer_grader_f = MistralAnswerGrader(
            client=mistral_client_f, model_name=mistral_model, rate_limiter=rate_limiter_f
        )
        query_rewriter_f = MistralQueryRewriter(
            client=mistral_client_f, model_name=mistral_model, rate_limiter=rate_limiter_f
        )

        doc_grader_v2_qwen_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader_f,
            doc_grader=doc_grader_v2,
            query_rewriter=query_rewriter_f,
            enable_answer_grading=True,
            enable_doc_grading=True,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
            max_retries=1,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="doc_grader_v2_qwen"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_mistral_grader_query(
                    doc_grader_v2_qwen_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"doc_grader_v2_qwen | F1={r['f1']:.3f} "
                    f"SF={r['supporting_facts_recall']:.3f} "
                    f"grading_ok={r['grading_accurate']}"
                )
            except Exception as e:
                logger.error(f"doc_grader_v2_qwen error on {qid}: {e}")

        all_summaries["doc_grader_v2_qwen"] = aggregate(results)
        logger.info(
            f"doc_grader_v2_qwen: F1={all_summaries['doc_grader_v2_qwen'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['doc_grader_v2_qwen'].get('supporting_facts_recall_mean', 0):.3f} "
            f"grading_accuracy={all_summaries['doc_grader_v2_qwen'].get('grading_accuracy', 0):.2%}"
        )

    # -----------------------------------------------------------------------
    # Config G: Doc grader v2 (multi-hop prompt) + Mistral-small generator
    # -----------------------------------------------------------------------
    if "doc_grader_v2_mistral" in requested_configs:
        logger.info("\n[Config G] Multi-hop doc grader v2 + Mistral-small generator")
        results_path = OUTPUT_DIR / "results_doc_grader_v2_mistral.json"
        partial = load_partial(results_path)
        done_ids = {r["question_id"] for r in partial}

        mistral_model = "mistral-small-latest"
        mistral_client_g = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        rate_limiter_g = RateLimiter(calls_per_second=0.5)

        doc_grader_v2_g = MistralDocumentGrader(
            client=mistral_client_g,
            model_name=mistral_model,
            multi_hop=True,
            rate_limiter=rate_limiter_g,
        )
        answer_grader_g = MistralAnswerGrader(
            client=mistral_client_g, model_name=mistral_model, rate_limiter=rate_limiter_g
        )
        query_rewriter_g = MistralQueryRewriter(
            client=mistral_client_g, model_name=mistral_model, rate_limiter=rate_limiter_g
        )
        mistral_gen_g = MistralAPIGenerator(
            model_name=mistral_model,
            api_key=os.environ.get("MISTRAL_API_KEY"),
            max_new_tokens=150,
            temperature=0.1,
        )

        doc_grader_v2_mistral_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=mistral_gen_g,
            answer_grader=answer_grader_g,
            doc_grader=doc_grader_v2_g,
            query_rewriter=query_rewriter_g,
            enable_answer_grading=True,
            enable_doc_grading=True,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
            max_retries=1,
        )

        results = list(partial)
        for item in tqdm(gt_records, desc="doc_grader_v2_mistral"):
            qid = item["question_id"]
            if qid in done_ids:
                continue
            try:
                r = run_mistral_grader_query(
                    doc_grader_v2_mistral_pipeline,
                    item["question"],
                    item["answer"],
                    item.get("supporting_facts", []),
                    metrics_calc,
                )
                r["query"] = item["question"]
                r["question_id"] = qid
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"doc_grader_v2_mistral | F1={r['f1']:.3f} "
                    f"SF={r['supporting_facts_recall']:.3f} "
                    f"grading_ok={r['grading_accurate']}"
                )
            except Exception as e:
                logger.error(f"doc_grader_v2_mistral error on {qid}: {e}")

        all_summaries["doc_grader_v2_mistral"] = aggregate(results)
        logger.info(
            f"doc_grader_v2_mistral: F1={all_summaries['doc_grader_v2_mistral'].get('f1_mean', 0):.3f} "
            f"SF_recall={all_summaries['doc_grader_v2_mistral'].get('supporting_facts_recall_mean', 0):.3f} "
            f"grading_accuracy={all_summaries['doc_grader_v2_mistral'].get('grading_accuracy', 0):.2%}"
        )

    # Save summary
    summary = {"num_questions": args.num_questions, "summaries": all_summaries}
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plots
    if not args.skip_plots and all_summaries:
        make_plots(all_summaries, OUTPUT_DIR)

    # Final table
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    header = (
        f"{'Config':<18} {'F1':>6} {'EM':>5} {'BERTScore':>10} {'SF_Recall':>10} {'GradAcc':>8}"
    )
    logger.info(header)
    for cfg, s in all_summaries.items():
        logger.info(
            f"{cfg:<18} "
            f"{s.get('f1_mean', 0):.3f}  "
            f"{s.get('em_mean', 0):.3f}  "
            f"{s.get('bertscore_mean', 0):.3f}       "
            f"{s.get('supporting_facts_recall_mean', 0):.3f}       "
            f"{s.get('grading_accuracy', float('nan')):.3f}"
        )

    # -----------------------------------------------------------------------
    # A/B statistical testing (baseline: linear)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("A/B STATISTICAL TESTING (baseline: linear)")
    logger.info("=" * 60)

    ab_runner = ABTestRunner(primary_metric="f1", guard_metrics=["faithfulness"])
    ab_results = {}
    METRICS = ["f1", "em", "bertscore", "faithfulness", "supporting_facts_recall"]

    # --- Baseline: linear (Qwen generator) vs all challengers ---
    linear_res = load_partial(OUTPUT_DIR / "results_linear.json")
    challengers = [
        ("adaptive", "results_adaptive.json"),
        ("mistral_grader", "results_mistral_grader.json"),
        ("mistral_answer_grader", "results_mistral_answer_grader.json"),
        ("linear_mistral", "results_linear_mistral.json"),
        ("doc_grader_v2_qwen", "results_doc_grader_v2_qwen.json"),
        ("doc_grader_v2_mistral", "results_doc_grader_v2_mistral.json"),
    ]
    for challenger_name, fname in challengers:
        chal_res = load_partial(OUTPUT_DIR / fname)
        if not chal_res:
            continue
        aligned_l, aligned_c = _align_by_query(linear_res, chal_res)
        if len(aligned_l) < 2:
            logger.warning(f"Not enough aligned queries for A/B test: linear vs {challenger_name}")
            continue
        ab_result = ab_runner.compare(
            aligned_l,
            aligned_c,
            metrics=METRICS,
            champion_name="linear",
            challenger_name=challenger_name,
        )
        logger.info("\n" + ab_result.summary())
        ab_results[f"linear_vs_{challenger_name}"] = ab_result.to_dict()

    # --- Baseline: linear_mistral — isolates grader effect on top of Mistral ---
    linear_mistral_res = load_partial(OUTPUT_DIR / "results_linear_mistral.json")
    if linear_mistral_res:
        mistral_challengers = [
            ("doc_grader_v2_mistral", "results_doc_grader_v2_mistral.json"),
        ]
        for challenger_name, fname in mistral_challengers:
            chal_res = load_partial(OUTPUT_DIR / fname)
            if not chal_res:
                continue
            aligned_l, aligned_c = _align_by_query(linear_mistral_res, chal_res)
            if len(aligned_l) < 2:
                logger.warning(f"Not enough aligned queries: linear_mistral vs {challenger_name}")
                continue
            ab_result = ab_runner.compare(
                aligned_l,
                aligned_c,
                metrics=METRICS,
                champion_name="linear_mistral",
                challenger_name=challenger_name,
            )
            logger.info("\n" + ab_result.summary())
            ab_results[f"linear_mistral_vs_{challenger_name}"] = ab_result.to_dict()

    if ab_results:
        summary["ab_tests"] = ab_results
        with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nDone. Review {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
