"""
Phase 5: HotpotQA preparation — download, sample, index.

One-time setup script for HotpotQA distractor setting.
Uses HuggingFace datasets (reliable, no manual download needed).

Steps:
  1. Load HotpotQA distractor validation split (HF datasets)
  2. Sample 200 "bridge" questions (or bridge+comparison to reach 200)
  3. Pool all paragraphs (2,000 = 200q × 10 paragraphs each); deduplicate by title
  4. Chunk paragraphs with FixedSizeChunker(200 tokens, 20 overlap)
  5. Build FAISS index (BGE-large-en-v1.5 embeddings)
  6. Build BM25 index from same chunks
  7. Save index to index/hotpot/

Outputs:
  data/hotpot/sample_200q.json           — 200 sampled questions with metadata
  data/hotpot/paragraphs_pool.json       — deduplicated paragraphs pool
  index/hotpot/faiss.index               — FAISS index
  index/hotpot/chunk_metadata.pkl        — chunk metadata

Usage:
  python scripts/prepare_hotpot.py [--num-questions N] [--question-type bridge|all]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from src.chunking import Chunk, FixedSizeChunker
from src.embeddings import EmbeddingModel, FAISSIndex
from src.loader import Document
from src.retriever import BM25Retriever
from src.utils import LoggerConfig, Timer, ensure_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/hotpot")
INDEX_DIR = Path("index/hotpot")
CHUNK_SIZE = 200  # tokens
CHUNK_OVERLAP = 20
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_hotpot(split: str = "validation", cache_dir: str = "data/cache") -> Any:
    """Load HotpotQA distractor split from HuggingFace."""
    logger.info("Loading HotpotQA (distractor) from HuggingFace datasets...")
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
        cache_dir=cache_dir,
    )
    logger.info(f"Loaded {len(dataset)} examples from HotpotQA {split} split")
    return dataset


def sample_questions(
    dataset: Any,
    num_questions: int = 200,
    question_type: str = "bridge",
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """
    Sample questions from HotpotQA dataset.

    Args:
        dataset: HuggingFace HotpotQA dataset.
        num_questions: Number of questions to sample.
        question_type: "bridge", "comparison", or "all".
        seed: Random seed for reproducibility.

    Returns:
        List of sampled question dicts.
    """
    random.seed(seed)

    # Filter by type
    if question_type == "bridge":
        candidates = [ex for ex in dataset if ex.get("type") == "bridge"]
        logger.info(f"Bridge questions available: {len(candidates)}")
        if len(candidates) < num_questions:
            logger.warning(
                f"Only {len(candidates)} bridge questions — "
                f"adding comparison to reach {num_questions}"
            )
            comparison = [ex for ex in dataset if ex.get("type") == "comparison"]
            candidates = candidates + comparison
    elif question_type == "comparison":
        candidates = [ex for ex in dataset if ex.get("type") == "comparison"]
    else:
        candidates = list(dataset)

    random.shuffle(candidates)
    sampled = candidates[:num_questions]
    logger.info(
        f"Sampled {len(sampled)} questions (type distribution: "
        f"{dict((q['type'], 0) for q in sampled)})"
    )

    from collections import Counter

    type_dist = Counter(q.get("type", "unknown") for q in sampled)
    logger.info(f"Type distribution: {dict(type_dist)}")

    return sampled


def pool_paragraphs(questions: List[Dict]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Pool all paragraphs from the sampled questions.

    HotpotQA context format: {"title": [...], "sentences": [[sentences], ...]}
    Each question has ~10 context paragraphs.

    Returns:
        paragraphs: Deduplicated list of {"doc_id": title, "content": text}.
        q_to_titles: Map from question_id to list of context titles.
    """
    paragraphs: Dict[str, str] = {}  # title -> content (dedup by title)
    q_to_titles: Dict[str, List[str]] = {}

    for q in questions:
        qid = q.get("id", q.get("_id", ""))
        context = q.get("context", {})

        # HF format: {"title": [...], "sentences": [[...], ...]}
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        q_titles = []
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences).strip()
            if text:
                # Deduplicate by title (same article may appear in multiple questions)
                if title not in paragraphs:
                    paragraphs[title] = text
                q_titles.append(title)

        q_to_titles[qid] = q_titles

    paragraph_list = [
        {"doc_id": title, "content": content} for title, content in paragraphs.items()
    ]
    logger.info(
        f"Pooled {len(paragraph_list)} unique paragraphs " f"from {len(questions)} questions"
    )
    return paragraph_list, q_to_titles


def build_ground_truth(questions: List[Dict]) -> List[Dict]:
    """
    Build ground truth records for evaluation.

    Returns list of dicts with question, answer, type, supporting_facts.
    """
    records = []
    for q in questions:
        # supporting_facts format in HF: {"title": [...], "sent_id": [...]}
        sf = q.get("supporting_facts", {})
        sf_titles = sf.get("title", [])
        sf_sent_ids = sf.get("sent_id", [])
        supporting_pairs = list(zip(sf_titles, sf_sent_ids))

        records.append(
            {
                "question_id": q.get("id", q.get("_id", "")),
                "question": q["question"],
                "answer": q["answer"],
                "type": q.get("type", ""),
                "supporting_facts": supporting_pairs,
            }
        )
    return records


def chunk_paragraphs(
    paragraphs: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk paragraphs using FixedSizeChunker."""
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = []

    for para in tqdm(paragraphs, desc="Chunking paragraphs"):
        doc = Document(
            content=para["content"],
            doc_id=para["doc_id"],
            metadata={"source": "hotpot"},
        )
        para_chunks = chunker.chunk(doc)
        chunks.extend(para_chunks)

    logger.info(
        f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs "
        f"(avg {len(chunks) / max(len(paragraphs), 1):.1f} chunks/para)"
    )
    return chunks


def build_faiss_index(
    chunks: List[Chunk],
    embed_model: EmbeddingModel,
    index_dir: Path,
) -> FAISSIndex:
    """Build and save FAISS index from chunks."""
    logger.info(f"Building FAISS index ({len(chunks)} chunks)...")
    ensure_dir(index_dir)

    with Timer("FAISS embedding"):
        embeddings = embed_model.encode_chunks(chunks, show_progress=True)

    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)

    index.save(str(index_dir))
    logger.info(f"Saved FAISS index to {index_dir}")
    logger.info(f"Index stats: {index.get_stats()}")
    return index


def build_bm25_index(chunks: List[Chunk]) -> BM25Retriever:
    """Build BM25 index from chunks."""
    bm25 = BM25Retriever()
    bm25.index(chunks)
    logger.info(f"BM25 index built ({len(chunks)} chunks)")
    return bm25


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA dataset and FAISS index")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument(
        "--question-type",
        choices=["bridge", "comparison", "all"],
        default="bridge",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    LoggerConfig.setup(level="INFO")
    ensure_dir(DATA_DIR)
    ensure_dir(INDEX_DIR)

    logger.info("=" * 60)
    logger.info("HotpotQA Preparation — Phase 5")
    logger.info("=" * 60)

    # 1. Load dataset
    dataset = load_hotpot()

    # 2. Sample questions
    questions = sample_questions(
        dataset,
        num_questions=args.num_questions,
        question_type=args.question_type,
        seed=args.seed,
    )

    # 3. Build ground truth
    gt_records = build_ground_truth(questions)
    gt_path = DATA_DIR / "sample_200q.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(gt_records)} ground truth records to {gt_path}")

    # 4. Pool paragraphs
    paragraphs, q_to_titles = pool_paragraphs(questions)
    para_path = DATA_DIR / "paragraphs_pool.json"
    with open(para_path, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(paragraphs)} paragraphs to {para_path}")

    # 5. Chunk paragraphs
    chunks = chunk_paragraphs(
        paragraphs,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )

    # 6. Build FAISS index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model (bge-large-en-v1.5, device={device})...")
    embed_model = EmbeddingModel(model_name="BAAI/bge-large-en-v1.5", device=device)

    faiss_index = build_faiss_index(chunks, embed_model, INDEX_DIR)

    # 7. BM25 index (in-memory; will be rebuilt from FAISS metadata at eval time)
    _ = build_bm25_index(chunks)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Questions sampled : {len(questions)}")
    logger.info(f"Unique paragraphs : {len(paragraphs)}")
    logger.info(f"Chunks created    : {len(chunks)}")
    logger.info(f"FAISS index size  : {faiss_index.get_stats()}")
    logger.info(f"Index saved to    : {INDEX_DIR}")
    logger.info(f"Ground truth      : {gt_path}")
    logger.info(f"Paragraphs pool   : {para_path}")
    logger.info("\nNext: run scripts/evaluate_hotpot.py")


if __name__ == "__main__":
    main()
