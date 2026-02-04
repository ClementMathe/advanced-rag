"""
Complete RAG pipeline orchestrating retrieval, re-ranking, and generation.

This module provides:
- RAGPipeline: End-to-end query → answer pipeline
- Integration of all RAG components
- Performance tracking and logging
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.generator import LLMGenerator
from src.reranker import CrossEncoderReranker
from src.retriever import HybridRetriever
from src.utils import Timer


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval → Re-ranking → Generation.

    Orchestrates the full flow from user query to final answer.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[LLMGenerator] = None,
        k_retrieve: int = 20,
        k_rerank: int = 5,
        use_reranking: bool = True,
        use_generation: bool = True,
    ):
        """
        Initialize RAG pipeline.

        Args:
            hybrid_retriever: HybridRetriever instance
            reranker: CrossEncoderReranker instance (optional)
            generator: LLMGenerator instance (optional)
            k_retrieve: Number of docs to retrieve before re-ranking
            k_rerank: Number of docs after re-ranking
            use_reranking: Whether to use re-ranking stage
            use_generation: Whether to use generation stage
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.k_retrieve = k_retrieve
        self.k_rerank = k_rerank
        self.use_reranking = use_reranking
        self.use_generation = use_generation

        # Validate configuration
        if use_reranking and reranker is None:
            logger.warning("Re-ranking enabled but no reranker provided. Disabling.")
            self.use_reranking = False

        if use_generation and generator is None:
            logger.warning("Generation enabled but no generator provided. Disabling.")
            self.use_generation = False

        logger.info(
            f"RAG Pipeline initialized: "
            f"k_retrieve={k_retrieve}, k_rerank={k_rerank}, "
            f"reranking={use_reranking}, generation={use_generation}"
        )

    def query(self, query: str, return_intermediate: bool = False) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline for a single query.

        Args:
            query: User question
            return_intermediate: Whether to return intermediate results

        Returns:
            Dictionary with answer and optional intermediate results
        """
        result = {"query": query}
        timings = {}

        # Stage 1: Hybrid Retrieval
        with Timer("Hybrid retrieval") as t:
            retrieved_chunks = self.hybrid_retriever.search(
                query, k=self.k_retrieve, k_retriever=50
            )
        timings["retrieval_ms"] = t.elapsed * 1000

        logger.debug(f"Retrieved {len(retrieved_chunks)} candidates")

        if return_intermediate:
            result["retrieved_chunks"] = retrieved_chunks

        # Stage 2: Re-ranking (optional)
        if self.use_reranking and self.reranker:
            with Timer("Re-ranking") as t:
                reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=self.k_rerank)
            timings["reranking_ms"] = t.elapsed * 1000

            logger.debug(f"Re-ranked to top-{self.k_rerank}")
            final_chunks = reranked_chunks
        else:
            final_chunks = retrieved_chunks[: self.k_rerank]

        if return_intermediate:
            result["reranked_chunks"] = final_chunks

        # Stage 3: Generation (optional)
        if self.use_generation and self.generator:
            with Timer("Generation") as t:
                generation_result = self.generator.generate(
                    query, final_chunks, max_chunks=self.k_rerank
                )
            timings["generation_ms"] = t.elapsed * 1000

            result["answer"] = generation_result["answer"]

            if return_intermediate:
                result["prompt"] = generation_result["prompt"]
        else:
            # Return chunks if no generation
            result["chunks"] = final_chunks

        # Add timing information
        result["timings"] = timings
        result["total_time_ms"] = sum(timings.values())

        logger.info(
            f"Pipeline completed in {result['total_time_ms']:.2f}ms "
            f"(retrieval: {timings.get('retrieval_ms', 0):.2f}ms, "
            f"rerank: {timings.get('reranking_ms', 0):.2f}ms, "
            f"generation: {timings.get('generation_ms', 0):.2f}ms)"
        )

        return result

    def batch_query(
        self, queries: List[str], return_intermediate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute pipeline for multiple queries.

        Args:
            queries: List of questions
            return_intermediate: Whether to return intermediate results

        Returns:
            List of result dictionaries
        """
        results = []

        logger.info(f"Processing {len(queries)} queries...")

        for i, query in enumerate(queries, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(queries)}")

            result = self.query(query, return_intermediate)
            results.append(result)

        logger.info(f"Batch processing completed: {len(results)} queries")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with component statistics
        """
        stats = {
            "k_retrieve": self.k_retrieve,
            "k_rerank": self.k_rerank,
            "use_reranking": self.use_reranking,
            "use_generation": self.use_generation,
        }

        # Add retriever stats if available
        if hasattr(self.hybrid_retriever, "get_stats"):
            stats["retriever"] = self.hybrid_retriever.get_stats()

        return stats


if __name__ == "__main__":
    # Example usage
    from src.chunking import FixedSizeChunker
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.loader import Document
    from src.reranker import CrossEncoderReranker
    from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Create sample documents
    docs = [
        Document(
            content="Paris is the capital of France. It is known for the Eiffel Tower.",
            doc_id="doc1",
        ),
        Document(
            content="The Eiffel Tower was completed in 1889 for the World's Fair.",
            doc_id="doc2",
        ),
        Document(
            content="France is a country in Western Europe with a rich history.",
            doc_id="doc3",
        ),
        Document(
            content="Python is a popular programming language for data science.",
            doc_id="doc4",
        ),
    ]

    # Build retrieval components
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_batch(docs)

    # Dense retriever
    embed_model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embeddings = embed_model.encode_chunks(chunks, show_progress=False)
    faiss_index = FAISSIndex(embed_model.embedding_dim, "Flat", "cosine")
    faiss_index.add(embeddings, chunks)
    dense_retriever = DenseRetriever(faiss_index, embed_model)

    # Sparse retriever
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    # Hybrid retriever
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    # Re-ranker
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base", device="cpu")

    # Generator (note: this will download Phi-3-mini ~2GB)
    logger.info("Loading LLM generator (this may take a minute)...")
    generator = LLMGenerator(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        load_in_4bit=False,  # Set to True if you have CUDA
        device="cpu",
    )

    # Complete pipeline
    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        k_retrieve=4,
        k_rerank=2,
        use_reranking=True,
        use_generation=True,
    )

    # Test query
    query = "What is the capital of France?"

    logger.info(f"\n{'='*60}")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*60}\n")

    result = pipeline.query(query, return_intermediate=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"ANSWER: {result['answer']}")
    logger.info(f"{'='*60}\n")

    logger.info(f"Total time: {result['total_time_ms']:.2f}ms")
    logger.info(f"Breakdown: {result['timings']}")

    logger.info("\nRetrieved chunks:")
    for i, chunk in enumerate(result["reranked_chunks"], 1):
        logger.info(f"{i}. (score={chunk.get('rerank_score', 0):.4f}) {chunk['content'][:80]}...")
