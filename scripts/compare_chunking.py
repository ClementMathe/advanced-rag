"""
Compare different chunking strategies on SQuAD documents.

This script:
1. Loads SQuAD documents
2. Applies all three chunking strategies
3. Compares chunk counts, sizes, and characteristics
4. Visualizes the differences

Usage:
    python scripts/compare_chunking.py --input data/squad/squad_v2_train_documents.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from loguru import logger

from src.chunking import Chunk, FixedSizeChunker, SemanticChunker, SlidingWindowChunker
from src.loader import Document
from src.utils import LoggerConfig, ensure_dir


def load_squad_documents(file_path: Path, max_docs: int = None) -> List[Document]:
    """
    Load SQuAD documents from JSON file.

    Args:
        file_path: Path to the JSON file
        max_docs: Maximum number of documents to load (None = all)

    Returns:
        List of Document objects
    """
    logger.info(f"Loading documents from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:  # FIX: Added encoding
        data = json.load(f)

    documents = []
    for item in data[:max_docs] if max_docs else data:
        doc = Document(
            content=item["content"], doc_id=item["doc_id"], metadata=item["metadata"]
        )
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def apply_chunking_strategies(documents: List[Document]) -> Dict[str, List[Chunk]]:
    """
    Apply all three chunking strategies to documents.

    Args:
        documents: List of documents to chunk

    Returns:
        Dictionary mapping strategy name to list of chunks
    """
    logger.info("Applying chunking strategies...")

    strategies = {
        "Fixed (512, overlap=50)": FixedSizeChunker(chunk_size=512, overlap=50),
        "Semantic (target=500)": SemanticChunker(
            target_size=500, min_size=100, max_size=1000
        ),
        "Sliding (512, stride=400)": SlidingWindowChunker(window_size=512, stride=400),
    }

    results = {}

    for name, chunker in strategies.items():
        logger.info(f"Applying {name}...")
        chunks = chunker.chunk_batch(documents)
        results[name] = chunks
        logger.info(f"  → Created {len(chunks)} chunks")

    return results


def analyze_chunks(all_chunks: Dict[str, List[Chunk]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze chunk characteristics for each strategy.

    Args:
        all_chunks: Dictionary mapping strategy name to chunks

    Returns:
        Dictionary with analysis results for each strategy
    """
    logger.info("Analyzing chunk characteristics...")

    analysis = {}

    for strategy_name, chunks in all_chunks.items():
        # Calculate statistics
        lengths = [len(chunk.content) for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "std_length": 0,  # Will calculate below
            "chunk_lengths": lengths,
        }

        # Calculate standard deviation
        if len(lengths) > 1:
            mean = stats["avg_length"]
            variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
            stats["std_length"] = variance**0.5

        analysis[strategy_name] = stats

        logger.info(f"{strategy_name}:")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  Avg length: {stats['avg_length']:.1f} chars")
        logger.info(f"  Length range: [{stats['min_length']}, {stats['max_length']}]")
        logger.info(f"  Std dev: {stats['std_length']:.1f}")

    return analysis


def visualize_comparison(analysis: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Create visualizations comparing chunking strategies.

    Args:
        analysis: Analysis results from analyze_chunks()
        output_dir: Directory to save plots
    """
    logger.info("Creating visualizations...")
    ensure_dir(output_dir)

    # Plot 1: Chunk count comparison
    plt.figure(figsize=(10, 6))

    strategies = list(analysis.keys())
    chunk_counts = [analysis[s]["total_chunks"] for s in strategies]

    plt.bar(range(len(strategies)), chunk_counts, color="skyblue", edgecolor="black")
    plt.xticks(range(len(strategies)), strategies, rotation=15, ha="right")
    plt.ylabel("Number of Chunks")
    plt.title("Total Chunks Created by Each Strategy")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "chunk_count_comparison.png", dpi=300)
    logger.info(f"Saved plot: {output_dir / 'chunk_count_comparison.png'}")
    plt.close()

    # Plot 2: Chunk length distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (strategy_name, stats) in enumerate(analysis.items()):
        ax = axes[idx]
        lengths = stats["chunk_lengths"]

        ax.hist(lengths, bins=30, edgecolor="black", alpha=0.7, color="coral")
        ax.axvline(
            stats["avg_length"],
            color="red",
            linestyle="--",
            label=f"Mean: {stats['avg_length']:.0f}",
        )
        ax.set_xlabel("Chunk Length (characters)")
        ax.set_ylabel("Frequency")
        ax.set_title(strategy_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "chunk_length_distributions.png", dpi=300)
    logger.info(f"Saved plot: {output_dir / 'chunk_length_distributions.png'}")
    plt.close()

    # Plot 3: Box plot comparison
    plt.figure(figsize=(10, 6))

    data_for_boxplot = [analysis[s]["chunk_lengths"] for s in strategies]

    bp = plt.boxplot(data_for_boxplot, labels=strategies, patch_artist=True)

    # Color boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")

    plt.ylabel("Chunk Length (characters)")
    plt.title("Chunk Length Distribution Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "chunk_length_boxplot.png", dpi=300)
    logger.info(f"Saved plot: {output_dir / 'chunk_length_boxplot.png'}")
    plt.close()


def print_sample_chunks(
    all_chunks: Dict[str, List[Chunk]], num_samples: int = 3
) -> None:
    """
    Print sample chunks from each strategy for manual inspection.

    Args:
        all_chunks: Dictionary mapping strategy name to chunks
        num_samples: Number of sample chunks to print per strategy
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE CHUNKS FOR MANUAL INSPECTION")
    logger.info("=" * 80)

    for strategy_name, chunks in all_chunks.items():
        logger.info(f"\n{strategy_name}:")
        logger.info("-" * 80)

        for i in range(min(num_samples, len(chunks))):
            chunk = chunks[i]
            preview = (
                chunk.content[:200] + "..."
                if len(chunk.content) > 200
                else chunk.content
            )
            logger.info(f"\nChunk {i+1} (length={len(chunk.content)} chars):")
            logger.info(f"  {preview}")


def save_results(analysis: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Save analysis results to JSON file.

    Args:
        analysis: Analysis results
        output_path: Path to save JSON file
    """
    # Remove chunk_lengths (too large for JSON)
    analysis_copy = {}
    for strategy, stats in analysis.items():
        analysis_copy[strategy] = {
            k: v for k, v in stats.items() if k != "chunk_lengths"
        }

    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_copy, f, indent=2)

    logger.info(f"Saved analysis results to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on SQuAD documents"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/squad/squad_v2_train_documents.json",
        help="Path to SQuAD documents JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/chunking_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=100,
        help="Maximum number of documents to process (for speed)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of sample chunks to display per strategy",
    )

    args = parser.parse_args()

    # Setup
    LoggerConfig.setup(level="INFO")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run: python scripts/prepare_squad.py first")
        return

    # Load documents
    documents = load_squad_documents(input_path, max_docs=args.max_docs)

    # Apply chunking strategies
    all_chunks = apply_chunking_strategies(documents)

    # Analyze
    analysis = analyze_chunks(all_chunks)

    # Visualize
    visualize_comparison(analysis, output_dir)

    # Print samples
    print_sample_chunks(all_chunks, num_samples=args.num_samples)

    # Save results
    save_results(analysis, output_dir / "analysis_results.json")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
