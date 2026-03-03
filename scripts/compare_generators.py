"""
Phase 3: Generator comparison — 20 SQuAD queries.

Compares three generators on identical retrieved context:
  A: Qwen2.5-3B-Instruct (local, free)
  B: mistral-small-latest API
  C: mistral-large-latest API

Retrieval + reranking is done once per query (shared across generators).
Metrics: F1, ROUGE-L, BERTScore, Faithfulness, Latency, Token usage, Cost.

Outputs:
  outputs/generator_comparison/results.json
  outputs/generator_comparison/summary.json
  outputs/generator_comparison/plots/quality_cost_curve.png
  outputs/generator_comparison/plots/latency_comparison.png
  outputs/generator_comparison/plots/metric_comparison.png

Usage:
  python scripts/compare_generators.py [--num-queries N] [--skip-plots]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.evaluation.ab_testing import ABTestRunner
from src.evaluation.metrics import PRICING_TABLE, MetricsCalculator
from src.generator import LLMGenerator
from src.mistral_generator import MistralAPIGenerator
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
OUTPUT_DIR = Path("outputs/generator_comparison")
NUM_QUERIES = 20
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------


def load_retrieval_components():
    """Load retriever + reranker (shared across all generator configs)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load(INDEXES_DIR)

    logger.info("Building BM25 index...")
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

    logger.info("Loading embedding model (bge-large-en-v1.5)...")
    embed_model = EmbeddingModel(model_name="BAAI/bge-large-en-v1.5", device=device)

    dense_retriever = DenseRetriever(faiss_index, embed_model)
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        k_rrf=60,
        dense_weight=0.9,
        sparse_weight=0.1,
    )

    logger.info("Loading reranker (ms-marco-MiniLM-L6-v2)...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
    )

    return hybrid_retriever, reranker, embed_model, device


def load_qwen_generator(device: str) -> LLMGenerator:
    logger.info("Loading Qwen2.5-3B-Instruct...")
    return LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        device=device,
        temperature=0.1,
        max_new_tokens=80,
    )


def load_mistral_generator(model_name: str) -> MistralAPIGenerator:
    return MistralAPIGenerator(
        model_name=model_name,
        max_new_tokens=80,
        temperature=0.1,
        retry_delay=1.2,
    )


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------


def _tokens_for_qwen(generator: LLMGenerator, text: str) -> int:
    """Approximate token count using the Qwen tokenizer."""
    try:
        return len(generator.tokenizer.encode(text))
    except Exception:
        return len(text.split())


def run_query_with_generator(
    generator,
    query: str,
    reranked_chunks: List[Dict],
    metrics_calc: MetricsCalculator,
    ground_truth: str,
) -> Dict[str, Any]:
    """Run a single query through one generator and return metrics."""
    is_local = isinstance(generator, LLMGenerator)

    t0 = time.time()
    result = generator.generate(query, reranked_chunks, max_chunks=TOP_K_RERANK)
    latency_ms = (time.time() - t0) * 1000

    answer = result["answer"]
    context_texts = [c.get("content", "") for c in reranked_chunks[:TOP_K_RERANK]]

    # Metrics
    f1 = metrics_calc.f1_score(answer, ground_truth)
    rouge_l = metrics_calc.rouge_l(answer, ground_truth)
    faithfulness = metrics_calc.faithfulness(answer, context_texts)
    bertscore = (
        metrics_calc.bert_score(answer, ground_truth)
        if (metrics_calc._use_bertscore and answer.strip() and ground_truth.strip())
        else 0.0
    )

    # Token counts
    if is_local:
        prompt = result.get("prompt", "")
        input_tokens = _tokens_for_qwen(generator, prompt)
        output_tokens = _tokens_for_qwen(generator, answer)
        model_key = "qwen2.5-3b-instruct"
    else:
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        model_key = result.get("model", "mistral-small-latest")

    # Cost computation (USD per 1M tokens)
    pricing = PRICING_TABLE.get(model_key, {"input": 0.0, "output": 0.0})
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    return {
        "answer": answer,
        "ground_truth": ground_truth,
        "f1": f1,
        "rouge_l": rouge_l,
        "faithfulness": faithfulness,
        "bertscore": bertscore,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "model": model_key,
    }


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------


def load_partial_results(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_partial_results(results: list, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict]) -> Dict[str, float]:
    metrics = ["f1", "rouge_l", "faithfulness", "bertscore", "latency_ms", "cost_usd"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
    agg["total_cost_usd"] = float(sum(r.get("cost_usd", 0) for r in results))
    agg["total_input_tokens"] = int(sum(r.get("input_tokens", 0) for r in results))
    agg["total_output_tokens"] = int(sum(r.get("output_tokens", 0) for r in results))
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(summaries: Dict[str, Dict], output_dir: Path):
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)

    configs = list(summaries.keys())
    colors = {"qwen": "#4C72B0", "mistral-small": "#DD8452", "mistral-large": "#55A868"}
    config_colors = [colors.get(c, "#333") for c in configs]

    # 1. Quality-cost curve (BERTScore vs cost/query)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg, color in zip(configs, config_colors):
        s = summaries[cfg]
        x = s.get("cost_usd_mean", 0.0)
        y = s.get("bertscore_mean", 0.0)
        ax.scatter(x, y, s=150, color=color, label=cfg, zorder=5)
        ax.annotate(cfg, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Avg cost / query (USD)")
    ax.set_ylabel("BERTScore (mean)")
    ax.set_title("Quality–Cost Curve: Generators on SQuAD (20q)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_cost_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved quality_cost_curve.png")

    # 2. Latency comparison (box plots approximated as bar + error)
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [summaries[c].get("latency_ms_mean", 0) for c in configs]
    stds = [summaries[c].get("latency_ms_std", 0) for c in configs]
    ax.bar(configs, means, color=config_colors, yerr=stds, capsize=5, alpha=0.85)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Generation Latency by Config (SQuAD 20q)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved latency_comparison.png")

    # 3. Metric comparison bar chart
    metric_keys = ["f1_mean", "rouge_l_mean", "bertscore_mean", "faithfulness_mean"]
    metric_labels = ["F1", "ROUGE-L", "BERTScore", "Faithfulness"]
    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cfg, color) in enumerate(zip(configs, config_colors)):
        vals = [summaries[cfg].get(k, 0.0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=cfg, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison by Generator (SQuAD 20q)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metric_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved metric_comparison.png")


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
    parser = argparse.ArgumentParser(description="Compare generators on 20 SQuAD queries")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen (saves VRAM)")
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")
    ensure_dir(OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Generator Comparison — Phase 3")
    logger.info("=" * 60)

    # Load queries
    with open(QUERIES_PATH) as f:
        all_queries = json.load(f)
    queries = all_queries[: args.num_queries]
    logger.info(f"Using {len(queries)} queries from SQuAD")

    # Load retrieval components
    hybrid_retriever, reranker, embed_model, device = load_retrieval_components()

    # Retrieve + rerank once per query (shared)
    logger.info("Running retrieval + reranking for all queries...")
    retrieved = []
    for item in tqdm(queries, desc="Retrieving"):
        query = item["query"]
        candidates = hybrid_retriever.search(query, k=TOP_K_RETRIEVE, k_retriever=50)
        reranked = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
        retrieved.append(reranked)
    logger.info(f"Retrieval done for {len(retrieved)} queries")

    # Metrics calculator (BERTScore enabled)
    metrics_calc = MetricsCalculator(embed_model=embed_model, use_bertscore=True)

    # Free GPU memory before loading Qwen: reranker is no longer needed,
    # embed_model moves to CPU (BERTScore still works, just slightly slower).
    import gc

    reranker.model.cpu()
    del reranker
    embed_model.model.to("cpu")
    embed_model.device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Freed reranker + moved embed_model to CPU to make room for Qwen")

    # Define configs
    generator_configs = []
    if not args.skip_qwen:
        generator_configs.append(("qwen", None))  # loaded after retrieval to share VRAM
    generator_configs.append(("mistral-small", "mistral-small-latest"))
    generator_configs.append(("mistral-large", "mistral-large-latest"))

    all_summaries = {}

    for config_name, model_id in generator_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config_name}")
        logger.info(f"{'='*60}")

        results_path = OUTPUT_DIR / f"results_{config_name}.json"
        partial = load_partial_results(results_path)
        already_done = {r["query"]: r for r in partial} if isinstance(partial, list) else {}

        # Load generator
        if config_name == "qwen":
            generator = load_qwen_generator(device)
        else:
            generator = load_mistral_generator(model_id)

        config_results = list(already_done.values())

        for i, (item, reranked_chunks) in enumerate(
            tqdm(zip(queries, retrieved), total=len(queries), desc=config_name)
        ):
            query = item["query"]
            gt = item.get(
                "answer",
                item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else "",
            )

            if query in already_done:
                logger.debug(f"[{i+1}] Skipping (already computed): {query[:50]}")
                continue

            try:
                result = run_query_with_generator(
                    generator, query, reranked_chunks, metrics_calc, gt
                )
                result["query"] = query
                result["query_idx"] = i
                config_results.append(result)
                save_partial_results(config_results, results_path)

                logger.info(
                    f"[{i+1}/{len(queries)}] {config_name} | "
                    f"F1={result['f1']:.3f} BERTScore={result['bertscore']:.3f} "
                    f"cost=${result['cost_usd']:.5f} "
                    f"lat={result['latency_ms']:.0f}ms"
                )
            except Exception as e:
                logger.error(f"[{i+1}] Error on query '{query[:40]}': {e}")

        # Release GPU memory for Qwen before loading next generator
        if config_name == "qwen":
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        agg = aggregate(config_results)
        all_summaries[config_name] = agg
        logger.info(f"\nSummary [{config_name}]:")
        for k, v in agg.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save overall summary
    summary = {
        "num_queries": len(queries),
        "configs": list(all_summaries.keys()),
        "summaries": all_summaries,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved summary to {OUTPUT_DIR / 'summary.json'}")

    # Plots
    if not args.skip_plots and all_summaries:
        make_plots(all_summaries, OUTPUT_DIR)

    # Print final comparison table
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    header = f"{'Config':<20} {'F1':>6} {'BERTScore':>10} {'Faithfulness':>13} {'Cost/q':>10} {'Lat(ms)':>9}"
    logger.info(header)
    for cfg, s in all_summaries.items():
        row = (
            f"{cfg:<20} "
            f"{s.get('f1_mean', 0):.3f}  "
            f"{s.get('bertscore_mean', 0):.3f}       "
            f"{s.get('faithfulness_mean', 0):.3f}          "
            f"${s.get('cost_usd_mean', 0):.5f}  "
            f"{s.get('latency_ms_mean', 0):.0f}"
        )
        logger.info(row)

    # -----------------------------------------------------------------------
    # A/B statistical testing (baseline: qwen)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("A/B STATISTICAL TESTING (baseline: qwen)")
    logger.info("=" * 60)

    ab_runner = ABTestRunner(primary_metric="f1", guard_metrics=["faithfulness"])
    ab_results = {}
    METRICS = ["f1", "bertscore", "faithfulness"]

    def _load_list(path):
        raw = load_partial_results(path)
        return raw if isinstance(raw, list) else []

    qwen_res = _load_list(OUTPUT_DIR / "results_qwen.json")
    for challenger in ["mistral-small", "mistral-large"]:
        chal_res = _load_list(OUTPUT_DIR / f"results_{challenger}.json")
        aligned_q, aligned_c = _align_by_query(qwen_res, chal_res)
        if len(aligned_q) < 2:
            logger.warning(f"Not enough aligned queries for A/B test: qwen vs {challenger}")
            continue
        ab_result = ab_runner.compare(
            aligned_q,
            aligned_c,
            metrics=METRICS,
            champion_name="qwen",
            challenger_name=challenger,
        )
        logger.info("\n" + ab_result.summary())
        ab_results[f"qwen_vs_{challenger.replace('-', '_')}"] = ab_result.to_dict()

    if ab_results:
        summary["ab_tests"] = ab_results
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info("\nDone. Review outputs/generator_comparison/")


if __name__ == "__main__":
    main()
