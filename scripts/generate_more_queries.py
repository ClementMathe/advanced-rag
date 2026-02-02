"""
Generate 500 queries from already prepared SQuAD documents.

Uses the existing sample_queries.json as a template and expands
by downloading the raw SQuAD dataset to extract more questions.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from loguru import logger


def generate_queries_from_raw_squad(num_queries=500, seed=42):
    """
    Generate queries directly from raw SQuAD v2 dataset.

    Args:
        num_queries: Number of queries to generate
        seed: Random seed for reproducibility

    Returns:
        List of query dictionaries
    """
    random.seed(seed)

    logger.info("Downloading raw SQuAD v2 dataset...")
    dataset = load_dataset("squad_v2", split="train", cache_dir="data/cache")

    logger.info(f"Loaded {len(dataset)} examples from SQuAD v2")

    # Load our prepared documents to map contexts to doc_ids
    docs_path = Path("data/squad/squad_v2_train_documents.json")
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Create mapping: context -> doc_id
    context_to_docid = {doc["content"]: doc["doc_id"] for doc in documents}

    logger.info(f"Loaded {len(documents)} prepared documents")

    # Extract queries
    queries = []
    seen_questions = set()

    for example in dataset:
        question = example["question"]
        context = example["context"]
        is_impossible = example.get("is_impossible", False)

        # Skip unanswerable questions and duplicates
        if is_impossible or question in seen_questions:
            continue

        # Check if we have this context in our prepared docs
        doc_id = context_to_docid.get(context)
        if not doc_id:
            continue

        seen_questions.add(question)

        queries.append(
            {
                "query": question,
                "doc_id": doc_id,
                "title": example.get("title", ""),
                "has_answer": True,
            }
        )

        if len(queries) >= num_queries:
            break

    logger.info(f"Generated {len(queries)} queries")

    return queries


def main():
    """Main execution."""
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    output_path = Path("data/squad/queries_500.json")

    # Generate queries
    queries = generate_queries_from_raw_squad(num_queries=500)

    # Statistics
    logger.info("\n" + "=" * 80)
    logger.info("QUERY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total queries: {len(queries)}")

    unique_docs = len(set(q["doc_id"] for q in queries))
    logger.info(f"Unique documents: {unique_docs}")
    logger.info(f"Avg queries per doc: {len(queries) / unique_docs:.1f}")

    # Sample
    logger.info("\nSample queries:")
    for i, q in enumerate(queries[:3], 1):
        logger.info(f"\n{i}. {q['query']}")
        logger.info(f"   Doc ID: {q['doc_id']}")
        logger.info(f"   Title: {q['title']}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved {len(queries)} queries to {output_path}")


if __name__ == "__main__":
    main()
