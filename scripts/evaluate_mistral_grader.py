"""
Phase 4: Grader ablation — 100 SQuAD queries.

Three configs on identical retrieval:
  A: Linear pipeline (no grading)
  B: Agentic + QwenGrader (reproduce Step 6.5 failing baseline)
  C: Agentic + MistralGrader (function calling — the fix)

Generation is Qwen2.5-3B for all configs (isolates grading from generation).
Key metric: false rejection rate (answers graded as unacceptable when correct).

Outputs:
  outputs/mistral_grader_eval/results_linear.json
  outputs/mistral_grader_eval/results_qwen_grader.json
  outputs/mistral_grader_eval/results_mistral_grader.json
  outputs/mistral_grader_eval/summary.json
  outputs/mistral_grader_eval/plots/metrics_comparison.png
  outputs/mistral_grader_eval/plots/false_rejection_rate.png
  outputs/mistral_grader_eval/plots/latency_cost_comparison.png

Usage:
  python scripts/evaluate_mistral_grader.py [--num-queries N] [--configs A,B,C]
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
from src.evaluation.metrics import PRICING_TABLE, MetricsCalculator
from src.generator import LLMGenerator
from src.graders import AnswerGrader
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

INDEXES_DIR = "index/squad"
QUERIES_PATH = "data/squad/queries_500_with_answers.json"
OUTPUT_DIR = Path("outputs/mistral_grader_eval")
NUM_QUERIES = 100
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------


def load_components():
    """Load retriever, reranker, generator, embed_model."""
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

    faiss_index = FAISSIndex.load(INDEXES_DIR)

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
# Config A: Linear pipeline
# ---------------------------------------------------------------------------


def run_linear_query(
    pipeline: RAGPipeline,
    query: str,
    gt: str,
    metrics: MetricsCalculator,
) -> Dict[str, Any]:
    t0 = time.time()
    result = pipeline.query(query, return_intermediate=True)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "")
    context = [c.get("content", "") for c in result.get("reranked_chunks", [])[:5]]

    return {
        "query": query,
        "answer": answer,
        "ground_truth": gt,
        "f1": metrics.f1_score(answer, gt),
        "rouge_l": metrics.rouge_l(answer, gt),
        "faithfulness": metrics.faithfulness(answer, context),
        "bertscore": (
            metrics.bert_score(answer, gt)
            if (metrics._use_bertscore and answer.strip() and gt.strip())
            else 0.0
        ),
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
        "grading_calls": 0,
        "retry_count": 0,
        "answer_graded_acceptable": True,
        "false_rejection": False,
    }


# ---------------------------------------------------------------------------
# Config B: Agentic + QwenGrader
# ---------------------------------------------------------------------------


def run_agentic_qwen_query(
    pipeline: AgenticRAGPipeline,
    query: str,
    gt: str,
    metrics: MetricsCalculator,
) -> Dict[str, Any]:
    t0 = time.time()
    result = pipeline.query(query)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "")
    context = [c.get("content", "") for c in result.get("context_documents", [])[:5]]

    f1 = metrics.f1_score(answer, gt)
    # False rejection: answer was rejected by grader but was actually correct
    answer_graded_acceptable = result.get("answer_is_acceptable", True)
    false_rejection = (not answer_graded_acceptable) and (f1 >= 0.3)

    return {
        "query": query,
        "answer": answer,
        "ground_truth": gt,
        "f1": f1,
        "rouge_l": metrics.rouge_l(answer, gt),
        "faithfulness": metrics.faithfulness(answer, context),
        "bertscore": (
            metrics.bert_score(answer, gt)
            if (metrics._use_bertscore and answer.strip() and gt.strip())
            else 0.0
        ),
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
        "grading_calls": result.get("retry_count", 0) + 1,
        "retry_count": result.get("retry_count", 0),
        "answer_graded_acceptable": answer_graded_acceptable,
        "false_rejection": false_rejection,
    }


# ---------------------------------------------------------------------------
# Config C: Agentic + MistralGrader
# ---------------------------------------------------------------------------


def run_agentic_mistral_query(
    pipeline: AgenticRAGPipeline,
    query: str,
    gt: str,
    metrics: MetricsCalculator,
    mistral_model: str = "mistral-small-latest",
) -> Dict[str, Any]:
    t0 = time.time()
    result = pipeline.query(query)
    latency_ms = (time.time() - t0) * 1000

    answer = result.get("answer", "")
    context = [c.get("content", "") for c in result.get("context_documents", [])[:5]]

    f1 = metrics.f1_score(answer, gt)
    answer_graded_acceptable = result.get("answer_is_acceptable", True)
    # False rejection: answer was acceptable (F1 >= 0.3) but grader rejected it
    false_rejection = (not answer_graded_acceptable) and (f1 >= 0.3)

    # Estimate grading cost: ~1 answer grading + (retry_count * 1) calls
    grading_calls = result.get("retry_count", 0) + 1
    num_docs = result.get("num_docs_retrieved", 5)
    # Doc grading is done once (5 docs → 5 calls via grade_batch)
    total_api_calls = grading_calls + num_docs
    # Rough cost: 500 tokens input + 10 output per call
    pricing = PRICING_TABLE.get(mistral_model, {"input": 0.20, "output": 0.60})
    cost_usd = total_api_calls * (500 * pricing["input"] + 10 * pricing["output"]) / 1_000_000

    return {
        "query": query,
        "answer": answer,
        "ground_truth": gt,
        "f1": f1,
        "rouge_l": metrics.rouge_l(answer, gt),
        "faithfulness": metrics.faithfulness(answer, context),
        "bertscore": (
            metrics.bert_score(answer, gt)
            if (metrics._use_bertscore and answer.strip() and gt.strip())
            else 0.0
        ),
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "grading_calls": grading_calls,
        "num_docs_graded": result.get("num_docs_graded", -1),
        "retry_count": result.get("retry_count", 0),
        "answer_graded_acceptable": answer_graded_acceptable,
        "false_rejection": false_rejection,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_partial(path: Path) -> List[Dict]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def save_partial(results: List[Dict], path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    metrics = ["f1", "rouge_l", "faithfulness", "bertscore", "latency_ms", "cost_usd"]
    agg = {
        "n": len(results),
        "false_rejection_rate": float(np.mean([r.get("false_rejection", False) for r in results])),
        "retry_rate": float(np.mean([r.get("retry_count", 0) > 0 for r in results])),
        "total_grading_cost_usd": float(sum(r.get("cost_usd", 0) for r in results)),
    }
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
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
        "qwen_grader": "#DD8452",
        "mistral_grader": "#55A868",
    }
    cfg_colors = [colors.get(c, "#888") for c in configs]

    # 1. Metrics comparison (F1, BERTScore, Faithfulness)
    metric_keys = ["f1_mean", "bertscore_mean", "faithfulness_mean"]
    metric_labels = ["F1", "BERTScore", "Faithfulness"]
    x = np.arange(len(metric_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cfg, color) in enumerate(zip(configs, cfg_colors)):
        vals = [summaries[cfg].get(k, 0.0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=cfg, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison: Linear vs QwenGrader vs MistralGrader (SQuAD 100q)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    # 2. False rejection rate
    fig, ax = plt.subplots(figsize=(7, 5))
    fr_rates = [summaries[c].get("false_rejection_rate", 0.0) * 100 for c in configs]
    bars = ax.bar(configs, fr_rates, color=cfg_colors, alpha=0.85)
    for bar, val in zip(bars, fr_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("False Rejection Rate by Config\n(% of queries where correct answer was rejected)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "false_rejection_rate.png", dpi=150)
    plt.close(fig)

    # 3. Latency + grading cost per config
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lat_means = [summaries[c].get("latency_ms_mean", 0) for c in configs]
    ax1.bar(configs, lat_means, color=cfg_colors, alpha=0.85)
    ax1.set_ylabel("Avg Latency (ms)")
    ax1.set_title("Latency by Config")
    ax1.grid(axis="y", alpha=0.3)

    costs = [summaries[c].get("total_grading_cost_usd", 0.0) for c in configs]
    ax2.bar(configs, costs, color=cfg_colors, alpha=0.85)
    ax2.set_ylabel("Total Grading Cost (USD)")
    ax2.set_title("Grading Cost (100 queries)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Latency & Cost Comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_cost_comparison.png", dpi=150)
    plt.close(fig)

    logger.info(f"Saved plots to {plots_dir}")


# ---------------------------------------------------------------------------
# A/B testing helper
# ---------------------------------------------------------------------------


def _align_by_query(results_a: list, results_b: list):
    """Align two per-query result lists by the 'query' key for paired tests."""
    b_map = {r["query"]: r for r in results_b}
    aligned_a, aligned_b = [], []
    for r in results_a:
        if r["query"] in b_map:
            aligned_a.append(r)
            aligned_b.append(b_map[r["query"]])
    return aligned_a, aligned_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Grader ablation on 100 SQuAD queries")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument(
        "--configs",
        default="linear,qwen_grader,mistral_grader",
        help="Comma-separated list of configs to run",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")
    ensure_dir(OUTPUT_DIR)
    requested_configs = [c.strip() for c in args.configs.split(",")]

    logger.info("=" * 60)
    logger.info("Grader Ablation — Phase 4 (SQuAD)")
    logger.info("=" * 60)

    # Load queries
    with open(QUERIES_PATH) as f:
        all_queries = json.load(f)
    queries = all_queries[: args.num_queries]
    logger.info(f"Using {len(queries)} queries")

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
        done_queries = {r["query"] for r in partial}

        linear_pipeline = RAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
        )

        results = list(partial)
        for item in tqdm(queries, desc="linear"):
            query = item["query"]
            gt = item.get("answer", "")
            if query in done_queries:
                continue
            try:
                r = run_linear_query(linear_pipeline, query, gt, metrics_calc)
                results.append(r)
                save_partial(results, results_path)
                logger.debug(f"linear | F1={r['f1']:.3f} | {query[:50]}")
            except Exception as e:
                logger.error(f"linear error: {e}")

        all_summaries["linear"] = aggregate(results)
        logger.info(
            f"linear: F1={all_summaries['linear'].get('f1_mean', 0):.3f} "
            f"false_rejection={all_summaries['linear'].get('false_rejection_rate', 0):.2%}"
        )

    # -----------------------------------------------------------------------
    # Config B: Agentic + QwenGrader
    # -----------------------------------------------------------------------
    if "qwen_grader" in requested_configs:
        logger.info("\n[Config B] Agentic + QwenGrader")
        results_path = OUTPUT_DIR / "results_qwen_grader.json"
        partial = load_partial(results_path)
        done_queries = {r["query"] for r in partial}

        qwen_answer_grader = AnswerGrader(generator=generator)
        qwen_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=qwen_answer_grader,
            enable_answer_grading=True,
            k_retrieve=TOP_K_RETRIEVE,
            k_rerank=TOP_K_RERANK,
            max_retries=1,
        )

        results = list(partial)
        for item in tqdm(queries, desc="qwen_grader"):
            query = item["query"]
            gt = item.get("answer", "")
            if query in done_queries:
                continue
            try:
                r = run_agentic_qwen_query(qwen_pipeline, query, gt, metrics_calc)
                results.append(r)
                save_partial(results, results_path)
                logger.debug(
                    f"qwen_grader | F1={r['f1']:.3f} "
                    f"acceptable={r['answer_graded_acceptable']} "
                    f"false_rej={r['false_rejection']} | {query[:50]}"
                )
            except Exception as e:
                logger.error(f"qwen_grader error: {e}")

        all_summaries["qwen_grader"] = aggregate(results)
        logger.info(
            f"qwen_grader: F1={all_summaries['qwen_grader'].get('f1_mean', 0):.3f} "
            f"false_rejection={all_summaries['qwen_grader'].get('false_rejection_rate', 0):.2%}"
        )

    # -----------------------------------------------------------------------
    # Config C: Agentic + MistralGrader
    # -----------------------------------------------------------------------
    if "mistral_grader" in requested_configs:
        logger.info("\n[Config C] Agentic + MistralGrader (function calling)")
        results_path = OUTPUT_DIR / "results_mistral_grader.json"
        partial = load_partial(results_path)
        done_queries = {r["query"] for r in partial}

        mistral_model = "mistral-small-latest"
        mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        rate_limiter = RateLimiter(calls_per_second=0.85)

        mistral_doc_grader = MistralDocumentGrader(
            client=mistral_client,
            model_name=mistral_model,
            rate_limiter=rate_limiter,
        )
        mistral_answer_grader = MistralAnswerGrader(
            client=mistral_client,
            model_name=mistral_model,
            rate_limiter=rate_limiter,
        )
        mistral_query_rewriter = MistralQueryRewriter(
            client=mistral_client,
            model_name=mistral_model,
            rate_limiter=rate_limiter,
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
        for item in tqdm(queries, desc="mistral_grader"):
            query = item["query"]
            gt = item.get("answer", "")
            if query in done_queries:
                continue
            try:
                r = run_agentic_mistral_query(
                    mistral_pipeline, query, gt, metrics_calc, mistral_model
                )
                results.append(r)
                save_partial(results, results_path)
                logger.info(
                    f"mistral_grader | F1={r['f1']:.3f} "
                    f"acceptable={r['answer_graded_acceptable']} "
                    f"false_rej={r['false_rejection']} "
                    f"docs_graded={r.get('num_docs_graded', '?')} | {query[:50]}"
                )
            except Exception as e:
                logger.error(f"mistral_grader error: {e}")

        all_summaries["mistral_grader"] = aggregate(results)
        logger.info(
            f"mistral_grader: F1={all_summaries['mistral_grader'].get('f1_mean', 0):.3f} "
            f"false_rejection={all_summaries['mistral_grader'].get('false_rejection_rate', 0):.2%} "
            f"total_cost=${all_summaries['mistral_grader'].get('total_grading_cost_usd', 0):.4f}"
        )

    # Save summary
    summary = {"num_queries": args.num_queries, "summaries": all_summaries}
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
        f"{'Config':<18} {'F1':>6} {'BERTScore':>10} {'Faith':>8} {'FalseRej%':>10} {'Cost($)':>9}"
    )
    logger.info(header)
    for cfg, s in all_summaries.items():
        logger.info(
            f"{cfg:<18} "
            f"{s.get('f1_mean', 0):.3f}  "
            f"{s.get('bertscore_mean', 0):.3f}       "
            f"{s.get('faithfulness_mean', 0):.3f}    "
            f"{s.get('false_rejection_rate', 0) * 100:.1f}%         "
            f"${s.get('total_grading_cost_usd', 0):.4f}"
        )

    # -----------------------------------------------------------------------
    # A/B statistical testing (baseline: linear)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("A/B STATISTICAL TESTING (baseline: linear)")
    logger.info("=" * 60)

    ab_runner = ABTestRunner(primary_metric="f1", guard_metrics=["faithfulness"])
    ab_results = {}
    METRICS = ["f1", "bertscore", "faithfulness"]

    linear_res = load_partial(OUTPUT_DIR / "results_linear.json")
    challengers = [
        ("qwen_grader", "results_qwen_grader.json"),
        ("mistral_grader", "results_mistral_grader.json"),
    ]
    for challenger_name, fname in challengers:
        chal_res = load_partial(OUTPUT_DIR / fname)
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

    if ab_results:
        summary["ab_tests"] = ab_results
        with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nDone. Review {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
