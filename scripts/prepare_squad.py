"""
SQuAD 2.0 dataset preparation script.

Downloads SQuAD 2.0 dataset, converts to Document format,
and performs exploratory data analysis.

Usage:
    python scripts/prepare_squad.py --output_dir data/squad --sample_size 1000
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from datasets import load_dataset
from loguru import logger

from src.loader import Document
from src.utils import LoggerConfig, ensure_dir


def download_squad(
    split: str = "train", cache_dir: str = "data/cache"
) -> Dict[str, Any]:
    """
    Download SQuAD 2.0 dataset from HuggingFace.

    Args:
        split: Dataset split to download ('train' or 'validation')
        cache_dir: Directory to cache downloaded data

    Returns:
        Dataset dictionary
    """
    logger.info(f"Downloading SQuAD 2.0 ({split} split)...")

    dataset = load_dataset("squad_v2", split=split, cache_dir=cache_dir)

    logger.info(f"Downloaded {len(dataset)} examples")
    return dataset


def convert_to_documents(dataset: Any, max_samples: int = None) -> List[Document]:
    """
    Convert SQuAD dataset to Document objects.

    Each context paragraph becomes a Document.

    Args:
        dataset: HuggingFace dataset
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        List of Document objects
    """
    documents = []
    context_seen = set()  # Track unique contexts

    dataset_size = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )

    logger.info(f"Converting {dataset_size} examples to Documents...")

    for idx, example in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        context = example["context"]

        # Skip duplicate contexts
        if context in context_seen:
            continue
        context_seen.add(context)

        # Extract metadata
        metadata = {
            "source": "squad_v2",
            "title": example.get("title", ""),
            "question": example.get("question", ""),
            "has_answer": len(example.get("answers", {}).get("text", [])) > 0,
            "original_index": idx,
        }

        # Create document
        doc = Document(content=context, metadata=metadata)
        documents.append(doc)

        if (idx + 1) % 1000 == 0:
            logger.info(
                f"Processed {idx + 1} examples, {len(documents)} unique contexts"
            )

    logger.info(
        f"Created {len(documents)} documents from {dataset_size} examples "
        f"({len(context_seen)} unique contexts)"
    )

    return documents


def save_documents(documents: List[Document], output_path: Path) -> None:
    """
    Save documents to JSON file.

    Args:
        documents: List of documents to save
        output_path: Path to output file
    """
    ensure_dir(output_path.parent)

    # Convert to serializable format
    data = [
        {"doc_id": doc.doc_id, "content": doc.content, "metadata": doc.metadata}
        for doc in documents
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(documents)} documents to {output_path}")


def analyze_documents(documents: List[Document], output_dir: Path) -> Dict[str, Any]:
    """
    Perform exploratory data analysis on documents.

    Args:
        documents: List of documents to analyze
        output_dir: Directory to save analysis plots

    Returns:
        Dictionary with analysis statistics
    """
    logger.info("Performing exploratory data analysis...")

    # Calculate statistics
    lengths = [len(doc.content) for doc in documents]
    word_counts = [len(doc.content.split()) for doc in documents]

    stats = {
        "total_documents": len(documents),
        "total_characters": sum(lengths),
        "total_words": sum(word_counts),
        "avg_length_chars": sum(lengths) / len(lengths),
        "avg_length_words": sum(word_counts) / len(word_counts),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "min_length_words": min(word_counts),
        "max_length_words": max(word_counts),
    }

    # Analyze metadata
    has_answer_count = sum(
        1 for doc in documents if doc.metadata.get("has_answer", False)
    )
    stats["has_answer_percentage"] = (has_answer_count / len(documents)) * 100

    # Title distribution
    titles = [doc.metadata.get("title", "") for doc in documents]
    title_counter = Counter(titles)
    stats["unique_titles"] = len(title_counter)
    stats["top_5_titles"] = title_counter.most_common(5)

    # Print statistics
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(
        f"Average length: {stats['avg_length_chars']:.1f} chars, {stats['avg_length_words']:.1f} words"
    )
    logger.info(
        f"Length range: {stats['min_length_chars']}-{stats['max_length_chars']} chars"
    )
    logger.info(f"Documents with answers: {stats['has_answer_percentage']:.1f}%")
    logger.info(f"Unique titles: {stats['unique_titles']}")

    # Create visualizations
    ensure_dir(output_dir)

    # Plot 1: Document length distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(
        stats["avg_length_chars"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats['avg_length_chars']:.0f}",
    )
    plt.xlabel("Document Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Document Lengths")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(word_counts, bins=50, edgecolor="black", alpha=0.7, color="green")
    plt.axvline(
        stats["avg_length_words"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats['avg_length_words']:.0f}",
    )
    plt.xlabel("Document Length (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Word Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "document_length_distribution.png", dpi=300)
    logger.info(f"Saved plot: {output_dir / 'document_length_distribution.png'}")
    plt.close()

    # Plot 2: Top titles
    if stats["unique_titles"] > 1:
        top_titles = [title for title, _ in stats["top_5_titles"]]
        top_counts = [count for _, count in stats["top_5_titles"]]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_titles)), top_counts, color="skyblue", edgecolor="black")
        plt.yticks(range(len(top_titles)), top_titles)
        plt.xlabel("Number of Contexts")
        plt.title("Top 5 Titles by Context Count")
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(output_dir / "top_titles.png", dpi=300)
        logger.info(f"Saved plot: {output_dir / 'top_titles.png'}")
        plt.close()

    # Save statistics to JSON
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        # Convert top_titles to serializable format
        stats_copy = stats.copy()
        stats_copy["top_5_titles"] = [
            {"title": title, "count": count} for title, count in stats["top_5_titles"]
        ]
        json.dump(stats_copy, f, indent=2)

    logger.info(f"Saved statistics to {stats_path}")

    return stats


def create_sample_queries(
    documents: List[Document], output_path: Path, num_samples: int = 100
) -> None:
    """
    Create sample queries for testing from SQuAD questions.

    Args:
        documents: List of documents
        output_path: Path to save queries
        num_samples: Number of sample queries to create
    """
    queries = []

    for doc in documents[:num_samples]:
        if "question" in doc.metadata:
            queries.append(
                {
                    "query": doc.metadata["question"],
                    "doc_id": doc.doc_id,
                    "has_answer": doc.metadata.get("has_answer", False),
                }
            )

    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    logger.info(f"Created {len(queries)} sample queries at {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare SQuAD 2.0 dataset for RAG system"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/squad",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Cache directory for downloads",
    )

    args = parser.parse_args()

    # Setup logging
    LoggerConfig.setup(level="INFO")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Download dataset
    dataset = download_squad(split=args.split, cache_dir=args.cache_dir)

    # Convert to documents
    documents = convert_to_documents(dataset, max_samples=args.sample_size)

    # Save documents
    save_documents(documents, output_dir / f"squad_v2_{args.split}_documents.json")

    # Analyze
    analyze_documents(documents, output_dir / "analysis")

    # Create sample queries
    create_sample_queries(
        documents, output_dir / "sample_queries.json", num_samples=100
    )

    logger.info("=== Preparation Complete ===")
    logger.info(f"Documents saved to: {output_dir}")
    logger.info(f"Analysis plots saved to: {output_dir / 'analysis'}")
    logger.info("Next step: Run chunking experiments on these documents")


if __name__ == "__main__":
    main()
