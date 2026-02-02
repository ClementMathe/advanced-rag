"""
Cross-encoder re-ranking for the Advanced RAG system.

CRITICAL FIX: Use raw logits from BGE-reranker instead of sigmoid.
BGE-reranker models output calibrated logits that should be used directly
for ranking. Applying sigmoid compresses the score range and degrades performance.

This module provides:
- CrossEncoderReranker: BGE-reranker for precise relevance scoring
- Integration with retrieval pipeline
"""

from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import GPUManager, Timer


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using BGE-reranker model.

    Computes precise relevance scores by jointly encoding
    query and document pairs.

    IMPORTANT: Uses raw logits for scoring (no sigmoid applied).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for re-ranking
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Determine device
        if device is None:
            self.device = GPUManager.get_device(prefer_gpu=True)
        else:
            self.device = torch.device(device)

        logger.info(f"Loading re-ranker model: {model_name}")
        logger.info(f"Device: {self.device}")

        # Load model and tokenizer
        with Timer("Re-ranker model loading"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Inference mode

        # Optimize for inference
        if self.device.type == "cuda":
            try:
                self.model.half()  # FP16
                logger.info("Using FP16 precision for faster inference")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")

        logger.info(f"Re-ranker initialized: {model_name}")

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder scores.

        Args:
            query: Query string
            documents: List of document dictionaries (must have 'content' field)
            top_k: Number of top results to return (None = all)

        Returns:
            Re-ranked list of documents with 'rerank_score' field added
        """
        if not documents:
            return []

        # Compute scores
        scores = self._compute_scores(query, documents)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by score (descending)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # Return top-K if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Reorganise candidats batch.
        """
        all_pairs = []
        query_doc_counts = []

        # 1. Prepare all pairs (Flattening)
        for query, candidates in zip(queries, candidates_list):
            query_doc_counts.append(len(candidates))
            for cand in candidates:
                all_pairs.append([query, cand.get("content", "")])

        if not all_pairs:
            return [[] for _ in queries]

        # 2. Scoring massif
        all_scores = []
        for i in range(0, len(all_pairs), self.batch_size):
            batch_pairs = all_pairs[i : i + self.batch_size]
            all_scores.extend(self._score_batch(batch_pairs))

        # 3. Scores reallocation and sort
        results = []
        current_idx = 0
        for i, count in enumerate(query_doc_counts):
            query_candidates = candidates_list[i]
            query_scores = all_scores[current_idx : current_idx + count]

            for doc, score in zip(query_candidates, query_scores):
                doc["rerank_score"] = float(score)

            reranked = sorted(query_candidates, key=lambda x: x["rerank_score"], reverse=True)
            if top_k:
                reranked = reranked[:top_k]

            results.append(reranked)
            current_idx += count

        return results

    def _compute_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: Query string
            documents: List of documents

        Returns:
            List of relevance scores (raw logits)
        """
        # Prepare input pairs
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            pairs.append([query, content])

        # Batch process
        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_scores = self._score_batch(batch_pairs)
            all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(self, pairs: List[List[str]]) -> List[float]:
        """
        Score a batch of query-document pairs.

        Args:
            pairs: List of [query, document] pairs

        Returns:
            List of relevance scores (raw logits for BGE-reranker)
        """
        # Tokenize
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # CRITICAL FIX: BGE-reranker outputs raw logits - use directly for ranking
            # Higher logit = more relevant document
            # DO NOT apply sigmoid - logits are calibrated for direct comparison
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                # Multi-class output: use logit of positive class
                scores = logits[:, 1]

        return scores.cpu().tolist()

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        if self.device.type != "cuda":
            return {"device": "cpu", "memory_used": 0.0}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        return {
            "device": "cuda",
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "model_name": self.model_name,
        }


class RetrievalPipeline:
    """
    Complete retrieval pipeline: Hybrid retrieval + Re-ranking.

    Combines dense, sparse, and re-ranking for optimal quality.
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        k_retrieve: int = 20,
        k_final: int = 5,
        use_reranking: bool = True,
    ):
        """
        Initialize complete retrieval pipeline.

        Args:
            hybrid_retriever: HybridRetriever instance
            reranker: CrossEncoderReranker instance (optional)
            k_retrieve: Number of docs to retrieve before re-ranking
            k_final: Number of final results to return
            use_reranking: Whether to use re-ranking
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.k_retrieve = k_retrieve
        self.k_final = k_final
        self.use_reranking = use_reranking

        if use_reranking and reranker is None:
            logger.warning("Re-ranking enabled but no reranker provided. Disabling.")
            self.use_reranking = False

        logger.info(
            f"Initialized RetrievalPipeline: "
            f"k_retrieve={k_retrieve}, k_final={k_final}, "
            f"use_reranking={use_reranking}"
        )

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search using full pipeline.

        Args:
            query: Query string
            k: Number of results (overrides k_final if provided)

        Returns:
            List of ranked results
        """
        k_final = k if k is not None else self.k_final

        # Stage 1: Hybrid retrieval
        with Timer("Hybrid retrieval"):
            candidates = self.hybrid_retriever.search(
                query, k=self.k_retrieve, k_retriever=50  # Increased for better recall
            )

        logger.debug(f"Retrieved {len(candidates)} candidates from hybrid")

        # Stage 2: Re-ranking (optional)
        if self.use_reranking and self.reranker is not None:
            with Timer("Re-ranking"):
                results = self.reranker.rerank(query, candidates, top_k=k_final)
            logger.debug(f"Re-ranked to top-{k_final}")
        else:
            results = candidates[:k_final]

        return results

    def batch_search(
        self, queries: List[str], k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        return [self.search(query, k=k) for query in queries]


if __name__ == "__main__":
    # Example usage demonstrating the fix
    from src.chunking import FixedSizeChunker
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.loader import Document
    from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Create sample documents
    docs = [
        Document(
            content="Python is a programming language. Python 3.11 was released in 2022.",
            doc_id="doc1",
        ),
        Document(
            content="Machine learning uses Python for data science applications.", doc_id="doc2"
        ),
        Document(content="Java is another popular programming language.", doc_id="doc3"),
        Document(content="The capital of France is Paris, a beautiful city.", doc_id="doc4"),
    ]

    # Build retrieval pipeline
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
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

    # Re-ranker with FIXED logit handling
    logger.info("Initializing re-ranker...")
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base", device="cpu")

    # Complete pipeline
    pipeline = RetrievalPipeline(
        hybrid_retriever, reranker, k_retrieve=4, k_final=2, use_reranking=True
    )

    # Test query
    query = "What is Python 3.11?"

    logger.info(f"\nQuery: {query}")

    # Without re-ranking
    logger.info("\n=== Without Re-ranking ===")
    hybrid_results = hybrid_retriever.search(query, k=2)
    for i, r in enumerate(hybrid_results, 1):
        logger.info(f"{i}. (RRF={r.get('rrf_score', 0):.4f}) {r['content'][:60]}...")

    # With re-ranking (using raw logits)
    logger.info("\n=== With Re-ranking (FIXED) ===")
    pipeline_results = pipeline.search(query, k=2)
    for i, r in enumerate(pipeline_results, 1):
        logger.info(f"{i}. (rerank={r.get('rerank_score', 0):.4f}) {r['content'][:60]}...")
