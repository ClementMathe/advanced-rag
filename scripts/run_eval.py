"""
Step 7 Evaluation Runner — End-to-end RAG pipeline evaluation.

Two-phase approach:
  Phase 1 (smoke test): Run 10 queries, display metrics + cost estimate.
  Phase 2 (full run):   If approved, run 100 queries + error analysis + plots.

Usage:
    python scripts/run_eval.py                     # interactive (smoke → full)
    python scripts/run_eval.py --smoke-only        # just the 10-query smoke test
    python scripts/run_eval.py --full              # skip smoke, run 100 directly
    python scripts/run_eval.py --n-samples 50      # custom sample size for full run
    python scripts/run_eval.py --with-ragas         # enable RAGAS LLM-based metrics

Outputs (saved to outputs/eval_results/):
    - eval_results.json           — per-query results + aggregate metrics
    - cost_summary.json           — token usage + hypothetical API costs
    - error_analysis.json         — error distribution + worst cases
    - error_distribution.png      — bar chart of error categories
    - metric_comparison.png       — metric overview chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env (MISTRAL_API_KEY)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from src.chunking import Chunk  # noqa: E402
from src.embeddings import EmbeddingModel, FAISSIndex  # noqa: E402
from src.evaluation.benchmark_suite import BenchmarkSuite  # noqa: E402
from src.evaluation.error_taxonomy import ErrorAnalyzer  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    CostTracker,
    MetricsCalculator,
    RagasEvaluator,
)
from src.evaluation.visualizations import (  # noqa: E402
    plot_error_distribution,
    plot_metric_comparison,
)
from src.generator import LLMGenerator  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from src.utils import LoggerConfig, ensure_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEXES_DIR = "index/squad"
QUERIES_PATH = "data/squad/queries_500_with_answers.json"
OUTPUT_DIR = "outputs/eval_results"

SMOKE_SAMPLES = 10
FULL_SAMPLES = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_components():
    """Load all pipeline components (retriever, reranker, generator)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # FAISS index
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load(INDEXES_DIR)

    # BM25 index (rebuild from FAISS chunk metadata)
    logger.info("Building BM25 index...")
    chunks = []
    for meta in faiss_index.chunk_metadata:
        chunks.append(
            Chunk(
                content=meta["content"],
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                start_char=0,
                end_char=len(meta["content"]),
                chunk_index=meta["chunk_index"],
                metadata=meta.get("metadata", {}),
            )
        )
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    # Embedding model
    logger.info("Loading embedding model (bge-large-en-v1.5)...")
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
    logger.info("Loading reranker (ms-marco-MiniLM-L6-v2)...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
    )

    # Generator
    logger.info("Loading generator (Qwen2.5-3B-Instruct)...")
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        device=device,
        temperature=0.1,
        max_new_tokens=80,
    )

    # Pipeline
    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        k_retrieve=20,
        k_rerank=5,
        use_reranking=True,
        use_generation=True,
    )

    return pipeline, generator, embed_model


def load_queries(n_samples: int) -> list[dict]:
    """Load and prepare queries with ground truth."""
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # Remap 'answer' → 'ground_truth' for BenchmarkSuite
    queries = []
    for q in all_queries[:n_samples]:
        queries.append(
            {
                "query": q["query"],
                "ground_truth": q["answer"],
                "id": q.get("id", ""),
            }
        )
    return queries


def print_metrics_table(aggregate: dict[str, dict]) -> None:
    """Pretty-print aggregate metrics."""
    print("\n" + "=" * 65)
    print(f"  {'Metric':<20s}  {'Mean':>8s}  {'Std':>8s}  {'P50':>8s}  {'P95':>8s}")
    print("-" * 65)
    for metric in ["exact_match", "f1", "rouge_l", "faithfulness", "bert_score", "latency_s"]:
        if metric not in aggregate:
            continue
        m = aggregate[metric]
        print(
            f"  {metric:<20s}  {m['mean']:8.4f}  {m['std']:8.4f}  "
            f"{m['p50']:8.4f}  {m['p95']:8.4f}"
        )
    print("=" * 65)


def print_cost_summary(cost_summary: dict) -> None:
    """Print cost estimate table."""
    n = cost_summary.get("total_queries", 0)
    if n == 0:
        return

    print(f"\n  Token usage ({n} queries):")
    print(f"    Avg input:  {cost_summary['avg_input_tokens']:.0f} tokens")
    print(f"    Avg output: {cost_summary['avg_output_tokens']:.0f} tokens")
    print(
        f"    Total:      {cost_summary['total_input_tokens']} in "
        f"+ {cost_summary['total_output_tokens']} out"
    )

    print("\n  Hypothetical API cost (if using cloud models):")
    print(f"  {'Model':<28s}  {'Per query':>12s}  {'Total':>12s}  {'100 queries':>12s}")
    print("  " + "-" * 68)
    for model_name, costs in cost_summary["cost_by_model"].items():
        per_q = costs["avg_cost_per_query_usd"]
        total = costs["total_cost_usd"]
        proj_100 = per_q * 100
        print(f"  {model_name:<28s}  ${per_q:>10.6f}  ${total:>10.6f}  ${proj_100:>10.4f}")
    print()


def run_error_analysis(per_query_results: list[dict], output_dir: Path) -> dict:
    """Run error analysis and generate plots."""
    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze(per_query_results)

    # Print summary
    print("\n  Error Analysis:")
    print(f"    Total predictions: {analysis.total_predictions}")
    print(f"    Error rate: {analysis.error_rate:.1%}")
    print()
    for cat, count in sorted(analysis.error_distribution.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / analysis.total_predictions * 100
            print(f"    {cat:<25s}  {count:>3d}  ({pct:.1f}%)")

    # Save error analysis JSON
    analysis_path = output_dir / "error_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2, default=str)
    logger.info(f"Error analysis saved to {analysis_path}")

    # Generate error distribution plot
    plot_path = output_dir / "error_distribution.png"
    plot_error_distribution(analysis, str(plot_path))
    logger.info(f"Error distribution plot saved to {plot_path}")

    return analysis.to_dict()


def run_phase(
    pipeline,
    generator,
    embed_model,
    n_samples: int,
    output_dir: Path,
    phase_name: str,
    use_ragas: bool = False,
) -> dict:
    """Run evaluation for a given number of samples."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  {phase_name}: Evaluating on {n_samples} queries")
    logger.info(f"{'=' * 70}\n")

    queries = load_queries(n_samples)

    # Set up evaluation components
    calc = MetricsCalculator(
        embed_model=embed_model,
        use_bertscore=True,
        bertscore_model="distilbert-base-uncased",
    )
    cost_tracker = CostTracker(
        tokenizer=generator.tokenizer,
        model_name="qwen2.5-3b-instruct",
    )

    # Optional RAGAS evaluator
    ragas_evaluator = None
    if use_ragas:
        ragas_evaluator = RagasEvaluator()
        if ragas_evaluator.is_available:
            logger.info("RAGAS LLM-based evaluation enabled (Mistral API)")
        else:
            logger.warning("RAGAS requested but unavailable (missing API key or deps)")
            ragas_evaluator = None

    suite = BenchmarkSuite(
        metrics_calculator=calc,
        cost_tracker=cost_tracker,
        ragas_evaluator=ragas_evaluator,
        output_dir=str(output_dir),
    )

    # Run evaluation
    result = suite.evaluate_pipeline(
        pipeline=pipeline,
        queries=queries,
        config_name="linear_pipeline",
        pipeline_type="linear",
    )

    # Print results
    print_metrics_table(result["aggregate_metrics"])
    if "cost_summary" in result:
        print_cost_summary(result["cost_summary"])

    # Print RAGAS metrics if available
    if "ragas_metrics" in result:
        print("\n  RAGAS LLM-based metrics (Mistral):")
        for metric, value in result["ragas_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"    {metric:<30s}  {value:.4f}")
        print()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Runner")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run the 10-query smoke test",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Skip smoke test, run full evaluation directly",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=FULL_SAMPLES,
        help=f"Number of samples for full run (default: {FULL_SAMPLES})",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Enable RAGAS LLM-based metrics (requires MISTRAL_API_KEY in .env)",
    )
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    # --- Load components ---
    logger.info("Loading pipeline components...")
    pipeline, generator, embed_model = load_components()
    logger.info("All components loaded.\n")

    # --- Phase 1: Smoke test ---
    if not args.full:
        smoke_result = run_phase(
            pipeline,
            generator,
            embed_model,
            n_samples=SMOKE_SAMPLES,
            output_dir=output_dir,
            phase_name="PHASE 1 — Smoke Test",
            use_ragas=args.with_ragas,
        )

        if args.smoke_only:
            # Save smoke results and exit
            with open(output_dir / "smoke_results.json", "w") as f:
                json.dump(smoke_result, f, indent=2, default=str)
            logger.info("Smoke test complete. Results saved.")
            return

        # Ask for approval to continue
        print("\n" + "=" * 65)
        print(f"  Smoke test done. Ready to run full evaluation ({args.n_samples} queries).")
        print(
            f"  Estimated time: ~{args.n_samples * 5:.0f}s " f"(~{args.n_samples * 5 / 60:.1f} min)"
        )
        print("=" * 65)

        try:
            answer = input("\n  Proceed with full evaluation? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer not in ("y", "yes"):
            logger.info("Aborted. Smoke results available above.")
            with open(output_dir / "smoke_results.json", "w") as f:
                json.dump(smoke_result, f, indent=2, default=str)
            return

    # --- Phase 2: Full evaluation ---
    full_result = run_phase(
        pipeline,
        generator,
        embed_model,
        n_samples=args.n_samples,
        output_dir=output_dir,
        phase_name=f"PHASE 2 — Full Evaluation ({args.n_samples} queries)",
        use_ragas=args.with_ragas,
    )

    # Save full results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)
    logger.info(f"Full results saved to {results_path}")

    # Save cost summary separately
    if "cost_summary" in full_result:
        cost_path = output_dir / "cost_summary.json"
        with open(cost_path, "w") as f:
            json.dump(full_result["cost_summary"], f, indent=2, default=str)

    # --- Error analysis + plots ---
    logger.info("\nRunning error analysis...")
    run_error_analysis(full_result["per_query_results"], output_dir)

    # Metric comparison plot (single config view)
    agg = full_result["aggregate_metrics"]
    plot_configs = {
        "linear_pipeline": {
            k: v["mean"]
            for k, v in agg.items()
            if k in {"f1", "exact_match", "rouge_l", "faithfulness", "bert_score"}
        }
    }
    plot_metric_comparison(
        plot_configs,
        str(output_dir / "metric_comparison.png"),
    )
    logger.info(f"Metric comparison plot saved to {output_dir / 'metric_comparison.png'}")

    print("\n" + "=" * 65)
    print("  Evaluation complete!")
    print(f"  Results: {output_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
