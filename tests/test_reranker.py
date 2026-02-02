"""
Unit tests for cross-encoder re-ranking.

Tests cover:
- CrossEncoderReranker functionality
- Score improvements
- Integration with retrieval pipeline
"""

import pytest

from src.chunking import FixedSizeChunker
from src.embeddings import EmbeddingModel, FAISSIndex
from src.loader import Document
from src.reranker import CrossEncoderReranker, RetrievalPipeline
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever


@pytest.fixture
def sample_documents():
    """Create sample documents for re-ranking tests."""
    return [
        {
            "chunk_id": "1",
            "doc_id": "doc1",
            "content": "Paris is the capital of France",
            "score": 0.9,
        },
        {
            "chunk_id": "2",
            "doc_id": "doc2",
            "content": "France is a country in Europe",
            "score": 0.7,
        },
        {
            "chunk_id": "3",
            "doc_id": "doc3",
            "content": "Berlin is the capital of Germany",
            "score": 0.6,
        },
    ]


@pytest.fixture
def reranker():
    """Create re-ranker (CPU for testing speed)."""
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",  # Small model for tests
        device="cpu",
        batch_size=4,
    )


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_initialization(self):
        """Test re-ranker initializes correctly."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", device="cpu"
        )

        assert reranker.model is not None
        assert reranker.tokenizer is not None
        assert reranker.device.type in ["cuda", "cpu"]

    def test_rerank_basic(self, reranker, sample_documents):
        """Test basic re-ranking functionality."""
        query = "What is the capital of France?"

        reranked = reranker.rerank(query, sample_documents)

        # Should return all documents
        assert len(reranked) == len(sample_documents)

        # Should have rerank_score field
        assert all("rerank_score" in doc for doc in reranked)

        # Scores should be a number
        assert all(isinstance(doc["rerank_score"], float) for doc in reranked)

    def test_rerank_top_k(self, reranker, sample_documents):
        """Test re-ranking with top_k parameter."""
        query = "What is the capital of France?"

        reranked = reranker.rerank(query, sample_documents, top_k=2)

        assert len(reranked) == 2

    def test_rerank_improves_ranking(self, reranker):
        """Test that re-ranking improves document order."""
        query = "What is the capital of France?"

        # Documents in suboptimal order
        docs = [
            {"chunk_id": "1", "content": "France is in Europe", "score": 0.9},
            {"chunk_id": "2", "content": "Paris is the capital of France", "score": 0.7},
            {"chunk_id": "3", "content": "Berlin is a city", "score": 0.6},
        ]

        reranked = reranker.rerank(query, docs)

        # After re-ranking, "Paris is the capital" should be first
        assert "Paris" in reranked[0]["content"]

        # Rerank score of top result should be higher than others
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    def test_empty_documents(self, reranker):
        """Test handling of empty document list."""
        query = "test query"
        reranked = reranker.rerank(query, [])

        assert reranked == []

    def test_batch_processing(self, reranker):
        """Test that batch processing works."""
        query = "What is the capital?"

        # Create many documents to test batching
        docs = [{"chunk_id": str(i), "content": f"Document number {i}"} for i in range(10)]

        reranked = reranker.rerank(query, docs)

        assert len(reranked) == 10
        assert all("rerank_score" in doc for doc in reranked)


class TestRetrievalPipeline:
    """Tests for complete retrieval pipeline."""

    def test_pipeline_without_reranking(self):
        """Test pipeline without re-ranking."""
        # Build simple retrieval components
        docs = [
            Document(content="Python programming", doc_id="doc1"),
            Document(content="Java programming", doc_id="doc2"),
        ]

        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_batch(docs)

        embed_model = EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        embeddings = embed_model.encode_chunks(chunks, show_progress=False)

        index = FAISSIndex(embed_model.embedding_dim, "Flat", "cosine")
        index.add(embeddings, chunks)

        dense = DenseRetriever(index, embed_model)
        bm25 = BM25Retriever()
        bm25.index(chunks)

        hybrid = HybridRetriever(dense, bm25)

        # Pipeline without re-ranking
        pipeline = RetrievalPipeline(
            hybrid, reranker=None, k_retrieve=2, k_final=1, use_reranking=False
        )

        results = pipeline.search("Python programming")

        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_pipeline_with_reranking(self):
        """Test complete pipeline with re-ranking."""
        # Build components
        docs = [
            Document(content="Python is a programming language", doc_id="doc1"),
            Document(content="Java is also a programming language", doc_id="doc2"),
            Document(content="Machine learning uses Python", doc_id="doc3"),
        ]

        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_batch(docs)

        embed_model = EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        embeddings = embed_model.encode_chunks(chunks, show_progress=False)

        index = FAISSIndex(embed_model.embedding_dim, "Flat", "cosine")
        index.add(embeddings, chunks)

        dense = DenseRetriever(index, embed_model)
        bm25 = BM25Retriever()
        bm25.index(chunks)

        hybrid = HybridRetriever(dense, bm25)

        # Re-ranker
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", device="cpu"
        )

        # Pipeline with re-ranking
        pipeline = RetrievalPipeline(hybrid, reranker, k_retrieve=3, k_final=2, use_reranking=True)

        results = pipeline.search("What is Python?")

        assert len(results) == 2
        assert "rerank_score" in results[0]

        # Re-ranked result should be relevant
        assert "Python" in results[0]["content"]


class TestScoreImprovements:
    """Tests validating that re-ranking improves scores."""

    def test_reranking_changes_order(self):
        """Test that re-ranking can change document order."""
        # Create reranker
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", device="cpu"
        )

        query = "capital of France"

        # Documents in wrong order (by some initial score)
        docs = [
            {"chunk_id": "1", "content": "France is a country", "score": 0.9},
            {"chunk_id": "2", "content": "Paris is the capital of France", "score": 0.5},
        ]

        # Re-rank
        reranked = reranker.rerank(query, docs)

        # Check that order changed (most relevant doc should be first)
        # The doc mentioning "capital of France" should have higher rerank score
        capital_doc = next(d for d in reranked if "capital" in d["content"])
        assert capital_doc == reranked[0], "Re-ranking should put most relevant doc first"

    def test_score_distribution(self, reranker):
        """Test that rerank scores have reasonable distribution."""
        query = "programming language"

        docs = [
            {"chunk_id": "1", "content": "Python is a programming language"},
            {"chunk_id": "2", "content": "The weather is nice today"},
            {"chunk_id": "3", "content": "Java is also a programming language"},
        ]

        reranked = reranker.rerank(query, docs)

        [d["rerank_score"] for d in reranked]

        # Relevant docs should have higher scores than irrelevant
        relevant_scores = [reranked[0]["rerank_score"], reranked[2]["rerank_score"]]
        irrelevant_score = next(d["rerank_score"] for d in reranked if "weather" in d["content"])

        # At least one relevant doc should score higher than irrelevant
        assert max(relevant_scores) > irrelevant_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
