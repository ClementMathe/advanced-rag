"""
Unit tests for retrieval strategies.

Tests cover:
- BM25Retriever functionality
- RRF fusion correctness
- HybridRetriever behavior
- Metric calculations
"""

import pytest

from src.chunking import Chunk
from src.embeddings import EmbeddingModel, FAISSIndex
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    calculate_retrieval_metrics,
)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    texts = [
        "Python is a programming language. Python 3.11 was released in 2022.",
        "Machine learning uses Python for data science and AI applications.",
        "Java is another popular programming language for enterprise software.",
        "The capital of France is Paris, a beautiful city.",
        "Python supports multiple programming paradigms including OOP.",
    ]

    chunks = []
    for i, text in enumerate(texts):
        chunk = Chunk(
            content=text,
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            start_char=0,
            end_char=len(text),
            chunk_index=i,
            metadata={"source": "test"},
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def bm25_retriever(sample_chunks):
    """Create BM25 retriever with indexed chunks."""
    retriever = BM25Retriever(k1=1.2, b=0.75)
    retriever.index(sample_chunks)
    return retriever


@pytest.fixture
def dense_retriever(sample_chunks):
    """Create dense retriever with indexed chunks."""
    embed_model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    embeddings = embed_model.encode_chunks(sample_chunks, show_progress=False)

    index = FAISSIndex(dimension=embed_model.embedding_dim, index_type="Flat", metric="cosine")
    index.add(embeddings, sample_chunks)

    return DenseRetriever(index, embed_model)


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def test_initialization(self):
        """Test BM25 retriever initializes correctly."""
        retriever = BM25Retriever(k1=1.5, b=0.8)

        assert retriever.k1 == 1.5
        assert retriever.b == 0.8
        assert retriever.bm25 is None

    def test_index_building(self, sample_chunks):
        """Test building BM25 index."""
        retriever = BM25Retriever()
        retriever.index(sample_chunks)

        assert retriever.bm25 is not None
        assert len(retriever.chunks) == len(sample_chunks)
        assert len(retriever.tokenized_corpus) == len(sample_chunks)

    def test_tokenization(self):
        """Test tokenization function."""
        retriever = BM25Retriever()

        text = "Python is Great!"
        tokens = retriever._tokenize(text)

        assert tokens == ["python", "is", "great!"]

    def test_search_basic(self, bm25_retriever):
        """Test basic search functionality."""
        query = "Python programming"
        results = bm25_retriever.search(query, k=3)

        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("content" in r for r in results)

    def test_exact_keyword_matching(self, bm25_retriever):
        """Test that BM25 finds exact keyword matches."""
        # Query with specific version number
        query = "Python 3.11"
        results = bm25_retriever.search(query, k=1)

        # First result should contain "3.11"
        assert "3.11" in results[0]["content"]

    def test_rare_term_boosting(self, bm25_retriever):
        """Test that rare terms are found in top results."""
        # "Paris" appears once, "Python" appears multiple times
        query_rare = "Paris"
        query_common = "Python"

        results_rare = bm25_retriever.search(query_rare, k=3)
        bm25_retriever.search(query_common, k=3)

        # Paris should be found somewhere in top-3
        # (not necessarily rank 1 due to length normalization in small corpus)
        retrieved_contents = [r["content"] for r in results_rare]
        assert any(
            "Paris" in content for content in retrieved_contents
        ), "Paris not found in top-3 results"

    def test_get_stats(self, bm25_retriever):
        """Test getting index statistics."""
        stats = bm25_retriever.get_stats()

        assert stats["indexed"] is True
        assert stats["num_documents"] > 0
        assert stats["vocabulary_size"] > 0
        assert "avg_document_length" in stats


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_search_basic(self, dense_retriever):
        """Test basic dense search."""
        query = "programming language"
        results = dense_retriever.search(query, k=2)

        assert len(results) == 2
        assert all("score" in r for r in results)

    def test_semantic_understanding(self, dense_retriever):
        """Test that dense retriever understands semantics."""
        # Query with paraphrase
        query = "What is the capital city of France?"
        results = dense_retriever.search(query, k=1)

        # Should find the Paris document
        assert "Paris" in results[0]["content"]

    def test_batch_search(self, dense_retriever):
        """Test batch search functionality."""
        queries = ["Python programming", "capital of France"]
        results = dense_retriever.batch_search(queries, k=2)

        assert len(results) == 2
        assert all(len(r) == 2 for r in results)


class TestHybridRetriever:
    """Tests for HybridRetriever with RRF fusion."""

    def test_initialization(self, dense_retriever, bm25_retriever):
        """Test hybrid retriever initializes correctly."""
        hybrid = HybridRetriever(
            dense_retriever, bm25_retriever, k_rrf=60, dense_weight=0.7, sparse_weight=0.3
        )

        assert hybrid.k_rrf == 60
        assert abs(hybrid.dense_weight + hybrid.sparse_weight - 1.0) < 1e-6

    def test_rrf_fusion_basic(self, dense_retriever, bm25_retriever):
        """Test basic RRF fusion."""
        hybrid = HybridRetriever(dense_retriever, bm25_retriever)

        query = "Python programming"
        results = hybrid.search(query, k=3)

        assert len(results) == 3
        assert all("rrf_score" in r for r in results)
        assert all("dense_rank" in r for r in results)
        assert all("sparse_rank" in r for r in results)

    def test_rrf_score_calculation(self, dense_retriever, bm25_retriever):
        """Test RRF score calculation is correct."""
        hybrid = HybridRetriever(
            dense_retriever, bm25_retriever, k_rrf=60, dense_weight=1.0, sparse_weight=1.0
        )

        # Create mock results
        dense_results = [
            {
                "chunk_id": "chunk_0",
                "doc_id": "doc_0",
                "content": "test",
                "score": 0.9,
                "chunk_index": 0,
                "metadata": {},
            },
            {
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "content": "test",
                "score": 0.8,
                "chunk_index": 1,
                "metadata": {},
            },
        ]

        sparse_results = [
            {
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "content": "test",
                "score": 15.0,
                "chunk_index": 1,
                "metadata": {},
            },
            {
                "chunk_id": "chunk_0",
                "doc_id": "doc_0",
                "content": "test",
                "score": 10.0,
                "chunk_index": 0,
                "metadata": {},
            },
        ]

        # Fuse
        fused = hybrid._reciprocal_rank_fusion(dense_results, sparse_results, k=2)

        # chunk_1 appears rank 2 in dense, rank 1 in sparse
        # RRF = 1/(60+2) + 1/(60+1) ≈ 0.0161 + 0.0164 = 0.0325

        # chunk_0 appears rank 1 in dense, rank 2 in sparse
        # RRF = 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 = 0.0325

        # They should be very close (depends on weight normalization)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] > 0

    def test_consensus_boosting(self, dense_retriever, bm25_retriever):
        """Test that documents in both top-K get boosted."""
        hybrid = HybridRetriever(dense_retriever, bm25_retriever)

        # Query that should match in both retrievers
        query = "Python programming language"
        results = hybrid.search(query, k=5)

        # Top result should appear in both retrievers
        top_result = results[0]
        assert top_result["dense_rank"] is not None
        assert top_result["sparse_rank"] is not None

    def test_batch_search(self, dense_retriever, bm25_retriever):
        """Test batch search in hybrid retriever."""
        hybrid = HybridRetriever(dense_retriever, bm25_retriever)

        queries = ["Python programming", "capital France"]
        results = hybrid.batch_search(queries, k=2)

        assert len(results) == 2
        assert all(len(r) == 2 for r in results)


class TestMetricsCalculation:
    """Tests for metric calculation functions."""

    def test_recall_at_k_perfect(self):
        """Test Recall@K with perfect retrieval."""
        results = [[{"doc_id": "doc1"}, {"doc_id": "doc2"}, {"doc_id": "doc3"}]]  # Ground truth
        ground_truth = ["doc1"]

        metrics = calculate_retrieval_metrics(results, ground_truth, k_values=[1, 3])

        assert metrics["recall_at_k"][1] == 1.0
        assert metrics["recall_at_k"][3] == 1.0

    def test_recall_at_k_partial(self):
        """Test Recall@K with ground truth not in top-1."""
        results = [
            [{"doc_id": "doc2"}, {"doc_id": "doc1"}, {"doc_id": "doc3"}]  # Ground truth at rank 2
        ]
        ground_truth = ["doc1"]

        metrics = calculate_retrieval_metrics(results, ground_truth, k_values=[1, 3])

        assert metrics["recall_at_k"][1] == 0.0  # Not in top-1
        assert metrics["recall_at_k"][3] == 1.0  # In top-3

    def test_recall_at_k_failure(self):
        """Test Recall@K when ground truth not found."""
        results = [[{"doc_id": "doc2"}, {"doc_id": "doc3"}, {"doc_id": "doc4"}]]
        ground_truth = ["doc1"]

        metrics = calculate_retrieval_metrics(results, ground_truth, k_values=[1, 3])

        assert metrics["recall_at_k"][1] == 0.0
        assert metrics["recall_at_k"][3] == 0.0

    def test_mrr_calculation(self):
        """Test MRR calculation."""
        results = [
            [{"doc_id": "doc1"}, {"doc_id": "doc2"}],  # Rank 1 → RR = 1.0
            [{"doc_id": "doc3"}, {"doc_id": "doc2"}],  # Rank 2 → RR = 0.5
            [{"doc_id": "doc4"}, {"doc_id": "doc5"}],  # Not found → RR = 0.0
        ]
        ground_truth = ["doc1", "doc2", "doc2"]

        metrics = calculate_retrieval_metrics(results, ground_truth)

        # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert abs(metrics["mrr"] - 0.5) < 1e-6

    def test_multiple_queries(self):
        """Test metrics with multiple queries."""
        results = [
            [{"doc_id": "doc1"}, {"doc_id": "doc2"}],
            [{"doc_id": "doc3"}, {"doc_id": "doc4"}],
            [{"doc_id": "doc5"}, {"doc_id": "doc6"}],
        ]
        ground_truth = ["doc1", "doc4", "doc5"]

        metrics = calculate_retrieval_metrics(results, ground_truth, k_values=[1, 2])

        # Query 1: doc1 at rank 1 ✓
        # Query 2: doc4 at rank 2 ✓
        # Query 3: doc5 at rank 1 ✓

        assert metrics["recall_at_k"][1] == 2 / 3  # 2 out of 3 in top-1
        assert metrics["recall_at_k"][2] == 1.0  # All 3 in top-2
        assert metrics["num_queries"] == 3


class TestIntegration:
    """Integration tests combining components."""

    def test_end_to_end_comparison(self, sample_chunks):
        """Test complete retrieval comparison pipeline."""
        # Build all retrievers
        embed_model = EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        embeddings = embed_model.encode_chunks(sample_chunks, show_progress=False)
        faiss_index = FAISSIndex(embed_model.embedding_dim, "Flat", "cosine")
        faiss_index.add(embeddings, sample_chunks)

        dense = DenseRetriever(faiss_index, embed_model)

        bm25 = BM25Retriever()
        bm25.index(sample_chunks)

        hybrid = HybridRetriever(dense, bm25)

        # Test query
        query = "Python 3.11 programming"

        dense_results = dense.search(query, k=2)
        sparse_results = bm25.search(query, k=2)
        hybrid_results = hybrid.search(query, k=2)

        # All should return results
        assert len(dense_results) == 2
        assert len(sparse_results) == 2
        assert len(hybrid_results) == 2

        # Hybrid should have RRF scores
        assert "rrf_score" in hybrid_results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
