"""
Ablation Study - Step 6 Pipeline Analysis

Evaluates 4 configurations:
1. Dense retrieval only
2. Hybrid retrieval (Dense + BM25)
3. Hybrid + Reranking
4. Full pipeline (Hybrid + Reranking + Generation)

Plus parameter variation study on number of chunks.

Outputs:
- JSON with all metrics
- 4 professional graphs
- Timing breakdown
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.generator import LLMGenerator
from src.pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    calculate_retrieval_metrics,
)
from src.utils import LoggerConfig, ensure_dir


class GenerationMetrics:
    """
    Metrics for evaluating generated answers.

    Computes:
    - Exact Match (EM)
    - F1 token overlap
    - ROUGE-L
    - Faithfulness (anti-hallucination via embeddings)
    """

    def __init__(self, embed_model):
        """
        Initialize metrics calculator.

        Args:
            embed_model: Embedding model for faithfulness computation
        """
        import re
        import string
        from collections import Counter

        from nltk.tokenize import sent_tokenize
        from rouge_score import rouge_scorer

        self.embed_model = embed_model
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._sent_tokenize = sent_tokenize
        self._Counter = Counter
        self._re = re
        self._string = string

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import re
        import string

        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match score (1.0 if match, 0.0 otherwise)."""
        pred_norm = self.normalize_text(prediction)
        gt_norm = self.normalize_text(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 token overlap score."""
        from collections import Counter

        pred_tokens = self.normalize_text(prediction).split()
        gt_tokens = self.normalize_text(ground_truth).split()

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)
        overlap = sum((pred_counts & gt_counts).values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def rouge_l(self, prediction: str, ground_truth: str) -> float:
        """Compute ROUGE-L F1 score."""
        scores = self.rouge_scorer.score(ground_truth, prediction)
        return scores["rougeL"].fmeasure

    def faithfulness(
        self, answer: str, context_chunks: List[str], threshold: float = 0.65
    ) -> float:
        """
        Compute faithfulness score (anti-hallucination).

        Uses embedding similarity between answer sentences and context chunks.
        """
        from nltk.tokenize import sent_tokenize

        if not answer.strip():
            return 0.0

        try:
            sentences = sent_tokenize(answer)
        except (LookupError, OSError):
            # Fallback if NLTK data not available
            sentences = [s.strip() for s in answer.split(".") if s.strip()]

        if not sentences:
            return 0.0

        sentence_embs = self.embed_model.encode(sentences)
        chunk_embs = self.embed_model.encode(context_chunks)

        supported_count = 0

        for sent_emb in sentence_embs:
            similarities = np.dot(chunk_embs, sent_emb) / (
                np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(sent_emb)
            )
            max_sim = np.max(similarities)

            if max_sim >= threshold:
                supported_count += 1

        return supported_count / len(sentences)

    def compute_all(
        self, prediction: str, ground_truth: str, context_chunks: List[str]
    ) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            "exact_match": self.exact_match(prediction, ground_truth),
            "f1": self.f1_score(prediction, ground_truth),
            "rouge_l": self.rouge_l(prediction, ground_truth),
            "faithfulness": self.faithfulness(prediction, context_chunks),
        }


class AblationStudy:
    """Complete ablation study of RAG pipeline."""

    def __init__(
        self,
        dense_retriever,
        hybrid_retriever,
        reranker,
        generator,
        embed_model,
    ):
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.embed_model = embed_model
        self.metrics_calc = GenerationMetrics(embed_model)

    def evaluate_config1_dense(self, queries: List[Dict], k_values: List[int] = None):
        """Config 1: Dense retrieval only."""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        logger.info("\n" + "=" * 60)
        logger.info("CONFIG 1: Dense Retrieval Only")
        logger.info("=" * 60)

        query_texts = [q["query"] for q in queries]
        ground_truth = [q.get("doc_id") for q in queries]

        # Retrieve
        results = self.dense_retriever.batch_search(query_texts, k=max(k_values))

        # Compute metrics
        metrics = calculate_retrieval_metrics(results, ground_truth, k_values)

        return {"metrics": metrics, "config": "dense_only", "has_generation": False}

    def evaluate_config2_hybrid(self, queries: List[Dict], k_values: List[int] = None):
        """Config 2: Hybrid retrieval."""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        logger.info("\n" + "=" * 60)
        logger.info("CONFIG 2: Hybrid Retrieval (Dense + BM25)")
        logger.info("=" * 60)

        query_texts = [q["query"] for q in queries]
        ground_truth = [q.get("doc_id") for q in queries]

        # Retrieve with hybrid
        results = self.hybrid_retriever.batch_search(query_texts, k=max(k_values), k_retriever=50)

        # Compute metrics
        metrics = calculate_retrieval_metrics(results, ground_truth, k_values)

        return {"metrics": metrics, "config": "hybrid", "has_generation": False}

    def evaluate_config3_reranking(self, queries: List[Dict], k_values: List[int] = None):
        """Config 3: Hybrid + Reranking."""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        logger.info("\n" + "=" * 60)
        logger.info("CONFIG 3: Hybrid + Reranking")
        logger.info("=" * 60)

        query_texts = [q["query"] for q in queries]
        ground_truth = [q.get("doc_id") for q in queries]

        # Retrieve candidates
        candidates = self.hybrid_retriever.batch_search(
            query_texts, k=20, k_retriever=50  # Get 20 for reranking
        )

        # Rerank
        reranked = self.reranker.batch_rerank(query_texts, candidates, top_k=max(k_values))

        # Compute metrics
        metrics = calculate_retrieval_metrics(reranked, ground_truth, k_values)

        return {"metrics": metrics, "config": "hybrid_rerank", "has_generation": False}

    def evaluate_config4_full(
        self,
        queries: List[Dict],
        k_values: List[int] = None,
        num_chunks: int = 5,
        save_individual_results: bool = False,
    ):
        """Config 4: Full pipeline with generation."""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        logger.info("\n" + "=" * 60)
        logger.info(f"CONFIG 4: Full Pipeline (num_chunks={num_chunks})")
        logger.info("=" * 60)

        # Create pipeline
        pipeline = RAGPipeline(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            generator=self.generator,
            k_retrieve=50,
            k_rerank=num_chunks,
            use_reranking=True,
            use_generation=True,
        )

        # Retrieval metrics (before generation)
        query_texts = [q["query"] for q in queries]
        ground_truth_ids = [q.get("doc_id") for q in queries]

        # Get reranked results for retrieval metrics
        candidates = self.hybrid_retriever.batch_search(query_texts, k=20, k_retriever=50)
        reranked = self.reranker.batch_rerank(query_texts, candidates, top_k=max(k_values))
        retrieval_metrics = calculate_retrieval_metrics(reranked, ground_truth_ids, k_values)

        # Generation metrics
        gen_metrics = {"exact_match": [], "f1": [], "rouge_l": [], "faithfulness": []}

        timings = []
        individual_results = []  # Store detailed results

        for query_data in tqdm(queries, desc="Generating"):
            question = query_data["query"]
            ground_truth_answer = query_data.get("answer", "")
            query_id = query_data.get("id", "unknown")

            # Run pipeline
            result = pipeline.query(question, return_intermediate=True)

            # Extract results
            prediction = result.get("answer", "")
            chunks = result.get("reranked_chunks", [])
            chunk_texts = [c.get("content", "") for c in chunks]

            # Compute generation metrics
            metrics = self.metrics_calc.compute_all(prediction, ground_truth_answer, chunk_texts)

            for metric_name, score in metrics.items():
                gen_metrics[metric_name].append(score)

            timings.append(result.get("timings", {}))

            # Save individual result if requested
            if save_individual_results:
                individual_results.append(
                    {
                        "query_id": query_id,
                        "question": question,
                        "ground_truth": ground_truth_answer,
                        "prediction": prediction,
                        "retrieved_chunks": [
                            {"content": c.get("content", ""), "score": c.get("score", 0.0)}
                            for c in chunks
                        ],
                        "metrics": metrics,
                        "timings": result.get("timings", {}),
                        "num_chunks": num_chunks,
                    }
                )

        # Aggregate generation metrics
        gen_metrics_agg = {}
        for metric_name, scores in gen_metrics.items():
            gen_metrics_agg[metric_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }

        # Average timings
        avg_timings = {}
        if timings:
            for key in timings[0].keys():
                avg_timings[key] = np.mean([t.get(key, 0) for t in timings])

        return {
            "metrics": retrieval_metrics,
            "generation_metrics": gen_metrics_agg,
            "timings": avg_timings,
            "config": "full_pipeline",
            "has_generation": True,
            "num_chunks": num_chunks,
            "individual_results": individual_results if save_individual_results else [],
        }

    def run_chunk_variation(self, queries: List[Dict], chunk_sizes: List[int] = None):
        """Vary number of chunks and measure impact."""
        if chunk_sizes is None:
            chunk_sizes = [1, 2, 3, 5, 7, 10]

        logger.info("\n" + "=" * 60)
        logger.info("CHUNK VARIATION STUDY")
        logger.info("=" * 60)

        results = []

        for num_chunks in chunk_sizes:
            logger.info(f"\nEvaluating with {num_chunks} chunks...")
            result = self.evaluate_config4_full(
                queries, k_values=[num_chunks], num_chunks=num_chunks  # Only need this K
            )
            results.append(result)

        return results


def plot_recall_comparison(results: Dict, output_dir: Path):
    """Graph 1: Recall@K comparison across configs 1-3."""
    configs = ["dense_only", "hybrid", "hybrid_rerank"]
    labels = ["Dense Only", "Hybrid (D+BM25)", "Hybrid + Reranking"]
    colors = ["#4A90D9", "#2ECC71", "#E74C3C"]

    k_values = [1, 3, 5, 10]

    plt.figure(figsize=(10, 6))

    for config, label, color in zip(configs, labels, colors):
        if config in results:
            # Extract from nested recall_at_k dict
            # Keys can be int or str depending on JSON loading
            recall_at_k = results[config]["metrics"].get("recall_at_k", {})
            recalls = []
            for k in k_values:
                # Try both int and str keys
                recall = recall_at_k.get(k, recall_at_k.get(str(k), 0))
                recalls.append(recall * 100)

            plt.plot(
                k_values, recalls, marker="o", label=label, color=color, linewidth=2, markersize=8
            )

    plt.xlabel("K (Number of Results)", fontsize=12)
    plt.ylabel("Recall (%)", fontsize=12)
    plt.title("Recall@K: Impact of Retrieval Strategy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()

    plt.savefig(output_dir / "recall_comparison.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'recall_comparison.png'}")
    plt.close()


def plot_generation_metrics(results: Dict, output_dir: Path):
    """Graph 2: Generation metrics comparison."""
    if "full_pipeline" not in results:
        logger.warning("No generation metrics to plot")
        return

    gen_metrics = results["full_pipeline"]["generation_metrics"]

    metrics = ["exact_match", "f1", "rouge_l", "faithfulness"]
    labels = ["Exact Match", "F1 Score", "ROUGE-L", "Faithfulness"]
    means = [gen_metrics[m]["mean"] for m in metrics]
    stds = [gen_metrics[m]["std"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x, means, width, yerr=stds, capsize=5, color="#9B59B6", alpha=0.8, edgecolor="black"
    )

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + stds[i] + 0.02,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Generation Quality Metrics (Full Pipeline)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "generation_metrics.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'generation_metrics.png'}")
    plt.close()


def plot_chunk_variation(chunk_results: List[Dict], output_dir: Path):
    """Graph 3: Impact of number of chunks."""
    chunk_sizes = [r["num_chunks"] for r in chunk_results]
    faithfulness = [r["generation_metrics"]["faithfulness"]["mean"] for r in chunk_results]
    latencies = [
        r["timings"].get("generation_ms", 0) / 1000 for r in chunk_results
    ]  # Convert to seconds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Faithfulness vs chunks
    ax1.plot(chunk_sizes, faithfulness, marker="o", color="#2ECC71", linewidth=2, markersize=10)
    ax1.set_xlabel("Number of Chunks", fontsize=12)
    ax1.set_ylabel("Faithfulness Score", fontsize=12)
    ax1.set_title("Faithfulness vs Number of Chunks", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Latency vs chunks
    ax2.plot(chunk_sizes, latencies, marker="s", color="#E74C3C", linewidth=2, markersize=10)
    ax2.set_xlabel("Number of Chunks", fontsize=12)
    ax2.set_ylabel("Generation Time (seconds)", fontsize=12)
    ax2.set_title("Latency vs Number of Chunks", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "chunk_variation.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'chunk_variation.png'}")
    plt.close()


def plot_latency_breakdown(results: Dict, output_dir: Path):
    """Graph 4: Latency breakdown stacked bar chart."""
    if "full_pipeline" not in results:
        logger.warning("No timing data to plot")
        return

    timings = results["full_pipeline"]["timings"]

    components = ["Retrieval", "Reranking", "Generation"]
    times = [
        timings.get("retrieval_ms", 0),
        timings.get("reranking_ms", 0),
        timings.get("generation_ms", 0),
    ]

    total = sum(times)
    percentages = [(t / total * 100) for t in times]

    colors = ["#3498DB", "#F39C12", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 6))

    cumsum = np.cumsum([0] + times[:-1])

    for i, (component, time, color, pct) in enumerate(zip(components, times, colors, percentages)):
        ax.barh(
            [0],
            [time],
            left=[cumsum[i]],
            color=color,
            label=f"{component}: {time:.0f}ms ({pct:.1f}%)",
            height=0.5,
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (milliseconds)", fontsize=12)
    ax.set_title("Pipeline Latency Breakdown (per query)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)

    # Add total time annotation
    ax.text(
        total / 2,
        0,
        f"Total: {total:.0f}ms ({total/1000:.1f}s)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "latency_breakdown.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'latency_breakdown.png'}")
    plt.close()


def main():
    """Run complete ablation study."""
    LoggerConfig.setup(level="INFO")

    # Configuration
    SAMPLE_SIZE = 100  # Full evaluation (use 10 for quick testing)
    INDEXES_DIR = "index/squad"
    QUERIES_PATH = "data/squad/queries_500_with_answers.json"  # With ground truth
    OUTPUT_DIR = "outputs/ablation_study"

    ensure_dir(OUTPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("ABLATION STUDY - Step 6 Pipeline Analysis")
    logger.info("=" * 80)

    # Load indexes and models
    logger.info("\n1. Loading indexes and models...")

    # FAISS
    faiss_index = FAISSIndex.load(INDEXES_DIR)

    # BM25
    chunks = []
    for meta in faiss_index.chunk_metadata:
        chunk = Chunk(
            content=meta["content"],
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            start_char=0,
            end_char=len(meta["content"]),
            chunk_index=meta["chunk_index"],
            metadata=meta.get("metadata", {}),
        )
        chunks.append(chunk)

    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    # Embedding model
    embed_model = EmbeddingModel(
        model_name="BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Retrievers
    dense_retriever = DenseRetriever(faiss_index, embed_model)
    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever, k_rrf=60, dense_weight=0.9, sparse_weight=0.1
    )

    # Reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Generator
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.1,
        max_new_tokens=80,
    )

    logger.info("All components loaded")

    # Load queries
    logger.info(f"\n2. Loading queries from {QUERIES_PATH}...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    queries = all_queries[:SAMPLE_SIZE]
    logger.info(f"Using {len(queries)} queries for ablation study")

    # Initialize ablation study
    study = AblationStudy(dense_retriever, hybrid_retriever, reranker, generator, embed_model)

    # Run 4 configurations
    results = {}

    results["dense_only"] = study.evaluate_config1_dense(queries)
    results["hybrid"] = study.evaluate_config2_hybrid(queries)
    results["hybrid_rerank"] = study.evaluate_config3_reranking(queries)

    # Config 4: Save individual results for detailed analysis
    logger.info("\nRunning Config 4 with detailed result tracking...")
    results["full_pipeline"] = study.evaluate_config4_full(
        queries, num_chunks=5, save_individual_results=True  # Enable detailed results
    )

    # Chunk variation study (optimized: only 1, 3, 5 chunks)
    logger.info("\n3. Running chunk variation study (1, 3, 5 chunks)...")
    chunk_results = study.run_chunk_variation(
        queries, chunk_sizes=[1, 3, 5]  # Reduced from [1,2,3,5,7,10] for efficiency
    )

    # Save results
    logger.info("\n4. Saving results...")

    # Extract and save individual results separately
    individual_results = results["full_pipeline"].pop("individual_results", [])

    # Save aggregate results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "chunk_variation_results.json", "w") as f:
        json.dump(chunk_results, f, indent=2)

    # Save individual Q/A results
    if individual_results:
        with open(output_dir / "individual_results.json", "w", encoding="utf-8") as f:
            json.dump(individual_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(individual_results)} individual Q/A results")

    # Generate plots
    logger.info("\n5. Generating visualizations...")
    plot_recall_comparison(results, output_dir)
    plot_generation_metrics(results, output_dir)
    plot_chunk_variation(chunk_results, output_dir)
    plot_latency_breakdown(results, output_dir)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("- ablation_results.json (aggregate metrics, 4 configs)")
    logger.info("- chunk_variation_results.json (chunk size impact)")
    logger.info(f"- individual_results.json ({len(individual_results)} Q/A examples)")
    logger.info("- recall_comparison.png")
    logger.info("- generation_metrics.png")
    logger.info("- chunk_variation.png")
    logger.info("- latency_breakdown.png")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
