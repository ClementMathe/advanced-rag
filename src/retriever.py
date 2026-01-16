"""
Retrieval strategies for the Advanced RAG system.

This module provides:
- BM25Retriever: Sparse retrieval using BM25 algorithm
- DenseRetriever: Wrapper for FAISS-based dense retrieval
- HybridRetriever: Combines dense and sparse with RRF fusion
"""

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex


class BM25Retriever:
    """
    BM25 sparse retrieval implementation.

    Uses term frequency and inverse document frequency for ranking.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
            epsilon: Floor value for IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []

        logger.info(f"Initialized BM25Retriever: k1={k1}, b={b}, epsilon={epsilon}")

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        self.chunks = chunks

        # Tokenize corpus (simple whitespace + lowercase)
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)

        logger.info(f"BM25 index built: {len(self.chunks)} documents indexed")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Simple tokenization: lowercase + split on whitespace.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization (can be improved with stemming/lemmatization)
        return text.lower().split()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using BM25.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of results with scores and metadata
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "score": float(scores[idx]),
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )

        return results

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        return [self.search(query, k=k) for query in queries]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with statistics
        """
        if self.bm25 is None:
            return {"indexed": False}

        vocab_size = len(set(token for doc in self.tokenized_corpus for token in doc))
        avg_doc_len = np.mean([len(doc) for doc in self.tokenized_corpus])

        return {
            "indexed": True,
            "num_documents": len(self.chunks),
            "vocabulary_size": vocab_size,
            "avg_document_length": float(avg_doc_len),
            "k1": self.k1,
            "b": self.b,
        }


class DenseRetriever:
    """
    Wrapper for FAISS-based dense retrieval.

    Provides consistent interface with BM25Retriever.
    """

    def __init__(self, index: FAISSIndex, embed_model: EmbeddingModel):
        """
        Initialize dense retriever.

        Args:
            index: FAISS index
            embed_model: Embedding model
        """
        self.index = index
        self.embed_model = embed_model

        logger.info("Initialized DenseRetriever")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using dense embeddings.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of results with scores and metadata
        """
        # Encode query
        query_embedding = self.embed_model.encode([query])

        # Search
        results = self.index.search_with_metadata(query_embedding, k=k)

        return results[0]  # Return first query's results

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries (batched for efficiency).

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        # Batch encode queries
        query_embeddings = self.embed_model.encode(queries)

        # Search
        results = self.index.search_with_metadata(query_embeddings, k=k)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "type": "dense",
            "index_stats": self.index.get_stats(),
            "embedding_dim": self.embed_model.embedding_dim,
        }


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval with RRF fusion.

    Uses Reciprocal Rank Fusion to combine rankings from multiple retrievers.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        k_rrf: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            k_rrf: RRF constant (typically 60)
            dense_weight: Weight for dense retriever (0-1)
            sparse_weight: Weight for sparse retriever (0-1)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Normalize weights
        total_weight = dense_weight + sparse_weight
        self.dense_weight = dense_weight / total_weight
        self.sparse_weight = sparse_weight / total_weight

        logger.info(
            f"Initialized HybridRetriever: k_rrf={k_rrf}, "
            f"weights=(dense:{self.dense_weight:.2f}, sparse:{self.sparse_weight:.2f})"
        )

    def search(self, query: str, k: int = 5, k_retriever: int = 20) -> List[Dict[str, Any]]:
        """
        Search using hybrid retrieval with RRF fusion.

        Args:
            query: Query string
            k: Number of final results to return
            k_retriever: Number of results to retrieve from each retriever

        Returns:
            List of results sorted by RRF score
        """
        # Retrieve from both retrievers
        dense_results = self.dense_retriever.search(query, k=k_retriever)
        sparse_results = self.sparse_retriever.search(query, k=k_retriever)

        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, k=k)

        return fused_results

    def batch_search(
        self, queries: List[str], k: int = 5, k_retriever: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries.

        Args:
            queries: List of query strings
            k: Number of final results per query
            k_retriever: Number of results from each retriever

        Returns:
            List of result lists, one per query
        """
        # Batch retrieval
        dense_results_batch = self.dense_retriever.batch_search(queries, k=k_retriever)
        sparse_results_batch = self.sparse_retriever.batch_search(queries, k=k_retriever)

        # Fuse each query's results
        fused_results = []
        for dense_results, sparse_results in zip(dense_results_batch, sparse_results_batch):
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k=k)
            fused_results.append(fused)

        return fused_results

    def _reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fuse rankings using Reciprocal Rank Fusion.

        Formula: RRF_score(d) = Σ (weight / (k_rrf + rank))

        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            k: Number of final results

        Returns:
            Fused results sorted by RRF score
        """
        # Collect all unique documents
        all_docs = {}

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result["chunk_id"]

            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "chunk_id": result["chunk_id"],
                    "doc_id": result["doc_id"],
                    "content": result["content"],
                    "chunk_index": result["chunk_index"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": result["score"],
                    "sparse_score": None,
                }

            # Add RRF contribution from dense retriever
            rrf_contribution = self.dense_weight / (self.k_rrf + rank)
            all_docs[doc_id]["rrf_score"] += rrf_contribution
            all_docs[doc_id]["dense_rank"] = rank
            all_docs[doc_id]["dense_score"] = result["score"]

        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result["chunk_id"]

            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "chunk_id": result["chunk_id"],
                    "doc_id": result["doc_id"],
                    "content": result["content"],
                    "chunk_index": result["chunk_index"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "dense_score": None,
                    "sparse_score": result["score"],
                }

            # Add RRF contribution from sparse retriever
            rrf_contribution = self.sparse_weight / (self.k_rrf + rank)
            all_docs[doc_id]["rrf_score"] += rrf_contribution
            all_docs[doc_id]["sparse_rank"] = rank
            all_docs[doc_id]["sparse_score"] = result["score"]

        # Sort by RRF score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["rrf_score"], reverse=True)

        return sorted_docs[:k]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hybrid retriever statistics.

        Returns:
            Dictionary with statistics from both retrievers
        """
        return {
            "type": "hybrid",
            "k_rrf": self.k_rrf,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "dense_stats": self.dense_retriever.get_stats(),
            "sparse_stats": self.sparse_retriever.get_stats(),
        }


def calculate_retrieval_metrics(
    results: List[List[Dict[str, Any]]],
    ground_truth_docs: List[str],
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculate retrieval metrics (Recall@K, MRR).

    Args:
        results: List of result lists, one per query
        ground_truth_docs: List of ground truth doc IDs
        k_values: List of K values to evaluate

    Returns:
        Dictionary with metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    recall_at_k = {k: 0.0 for k in k_values}
    reciprocal_ranks = []

    valid_queries = 0

    for query_results, ground_truth in zip(results, ground_truth_docs):
        if not ground_truth:
            continue

        valid_queries += 1

        # Extract retrieved doc IDs
        retrieved_docs = [r["doc_id"] for r in query_results]

        # Calculate Recall@K
        for k in k_values:
            if ground_truth in retrieved_docs[:k]:
                recall_at_k[k] += 1.0

        # Calculate MRR
        try:
            rank = retrieved_docs.index(ground_truth) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    # Normalize
    if valid_queries > 0:
        for k in k_values:
            recall_at_k[k] /= valid_queries

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {"recall_at_k": recall_at_k, "mrr": mrr, "num_queries": valid_queries}


if __name__ == "__main__":
    # Example usage
    from src.chunking import FixedSizeChunker
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.loader import Document
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Create sample documents
    docs = [
        Document(
            content="Python is a programming language. Python 3.11 was released in 2022.",
            doc_id="doc1",
        ),
        Document(
            content="Machine learning uses Python for data science and AI applications.",
            doc_id="doc2",
        ),
        Document(
            content="Java is another popular programming language for enterprise software.",
            doc_id="doc3",
        ),
    ]

    # Chunk
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_batch(docs)

    logger.info(f"Created {len(chunks)} chunks")

    # Build BM25 index
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    # Build dense index
    embed_model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embeddings = embed_model.encode_chunks(chunks, show_progress=False)

    faiss_index = FAISSIndex(
        dimension=embed_model.embedding_dim, index_type="Flat", metric="cosine"
    )
    faiss_index.add(embeddings, chunks)

    dense_retriever = DenseRetriever(faiss_index, embed_model)

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    # Test queries
    queries = ["What is Python 3.11?", "Machine learning programming language"]

    for query in queries:
        logger.info(f"\nQuery: {query}")

        # Dense
        dense_results = dense_retriever.search(query, k=2)
        logger.info("Dense top-2:")
        for i, r in enumerate(dense_results, 1):
            logger.info(f"  {i}. (score={r['score']:.3f}) {r['content'][:60]}...")

        # Sparse
        sparse_results = bm25_retriever.search(query, k=2)
        logger.info("Sparse top-2:")
        for i, r in enumerate(sparse_results, 1):
            logger.info(f"  {i}. (score={r['score']:.3f}) {r['content'][:60]}...")

        # Hybrid
        hybrid_results = hybrid_retriever.search(query, k=2)
        logger.info("Hybrid top-2:")
        for i, r in enumerate(hybrid_results, 1):
            logger.info(f"  {i}. (RRF={r['rrf_score']:.4f}) {r['content'][:60]}...")
