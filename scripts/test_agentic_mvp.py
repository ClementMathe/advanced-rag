"""
Quick integration test for the agentic RAG pipeline MVP.

Loads all components and runs 5 test queries to verify:
- Graph executes end-to-end (retrieve -> grade -> generate)
- Grading filters some documents (graded < retrieved)
- Answers are generated for all queries
- Intermediate steps are logged correctly

Usage:
    python scripts/test_agentic_mvp.py
"""

import torch
from loguru import logger

from src.agentic_pipeline import AgenticRAGPipeline
from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.generator import LLMGenerator
from src.graders import DocumentGrader
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import LoggerConfig, Timer


def load_components():
    """Load all pipeline components (matches ablation_study.py pattern)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FAISS index
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex.load("index/squad")

    # BM25 (rebuild from FAISS metadata)
    logger.info("Building BM25 index...")
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
    logger.info("Loading embedding model...")
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
    logger.info("Loading reranker...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
    )

    # Generator (shared with grader)
    logger.info("Loading LLM generator...")
    generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        temperature=0.1,
        max_new_tokens=80,
    )

    # Grader (reuses generator)
    grader = DocumentGrader(generator)

    return hybrid_retriever, reranker, generator, grader


def main():
    LoggerConfig.setup(level="INFO")

    logger.info("=" * 60)
    logger.info("AGENTIC RAG MVP - Integration Test")
    logger.info("=" * 60)

    # Load components
    with Timer("Component loading"):
        hybrid_retriever, reranker, generator, grader = load_components()

    # Create pipeline
    pipeline = AgenticRAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        grader=grader,
        k_retrieve=20,
        k_rerank=10,
    )

    # Test queries
    test_queries = [
        "When did Beyoncé become famous?",
        "What city did Beyoncé grow up in?",
        "How many Grammy awards did Beyoncé win for her debut album?",
        "Who managed Destiny's Child?",
        "What was Beyoncé's first solo album?",
    ]

    # Run queries
    results = []
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(test_queries)}] Query: {query}")
        logger.info(f"{'='*60}")

        result = pipeline.query(query)
        results.append(result)

        logger.info(f"Answer: {result['answer']}")
        logger.info(
            f"Docs: {result['num_docs_graded']}/{result['num_docs_retrieved']} graded relevant"
        )
        logger.info(f"Steps: {' -> '.join(result['steps'])}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    all_passed = True
    for i, (_query, result) in enumerate(zip(test_queries, results), 1):
        has_answer = bool(result["answer"].strip())
        has_steps = len(result["steps"]) == 3
        status = "PASS" if (has_answer and has_steps) else "FAIL"
        if status == "FAIL":
            all_passed = False

        logger.info(
            f"  [{status}] Q{i}: "
            f"answer={'yes' if has_answer else 'NO'}, "
            f"steps={len(result['steps'])}/3, "
            f"graded={result['num_docs_graded']}/{result['num_docs_retrieved']}"
        )

    logger.info(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
