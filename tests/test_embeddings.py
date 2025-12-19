"""
Unit tests for embedding generation and FAISS indexing.

Tests cover:
- EmbeddingModel functionality
- FAISSIndex creation and operations
- Search quality and correctness
- Performance characteristics
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.chunking import Chunk, FixedSizeChunker
from src.embeddings import EmbeddingModel, FAISSIndex
from src.loader import Document


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "The cat sat on the mat.",
        "A dog played in the park.",
        "Machine learning is fascinating.",
        "Paris is the capital of France.",
        "The weather is nice today.",
    ]


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    texts = [
        "The cat sat on the mat.",
        "A dog played in the park.",
        "Machine learning is fascinating.",
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
def embedding_model():
    """Create embedding model for testing (small model for speed)."""
    return EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small fast model
        device="cpu",  # CPU for reproducibility in tests
        batch_size=8,
    )


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    def test_model_initialization(self, embedding_model):
        """Test that model initializes correctly."""
        assert embedding_model.model is not None
        assert embedding_model.embedding_dim > 0
        assert embedding_model.device.type in ["cuda", "cpu"]

    def test_encode_single_text(self, embedding_model, sample_texts):
        """Test encoding a single text."""
        text = sample_texts[0]
        embedding = embedding_model.encode([text])

        assert embedding.shape == (1, embedding_model.embedding_dim)
        assert embedding.dtype == np.float32

    def test_encode_multiple_texts(self, embedding_model, sample_texts):
        """Test encoding multiple texts."""
        embeddings = embedding_model.encode(sample_texts)

        assert embeddings.shape == (len(sample_texts), embedding_model.embedding_dim)
        assert embeddings.dtype == np.float32

    def test_encode_empty_list(self, embedding_model):
        """Test encoding empty list returns empty array."""
        embeddings = embedding_model.encode([])

        assert embeddings.shape == (0,)

    def test_embedding_similarity(self, embedding_model):
        """Test that similar texts have high cosine similarity."""
        text1 = "The cat is on the mat."
        text2 = "A feline sits on the rug."
        text3 = "Python is a programming language."

        embeddings = embedding_model.encode([text1, text2, text3])

        # Compute cosine similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_1_2 = cosine_sim(embeddings[0], embeddings[1])
        sim_1_3 = cosine_sim(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3
        assert sim_1_2 > 0.5  # Reasonable threshold for similar texts

    def test_encode_chunks(self, embedding_model, sample_chunks):
        """Test encoding Chunk objects."""
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        assert embeddings.shape == (len(sample_chunks), embedding_model.embedding_dim)

    def test_normalized_embeddings(self, embedding_model, sample_texts):
        """Test that embeddings are normalized when requested."""
        embeddings = embedding_model.encode(sample_texts)

        # Check normalization (L2 norm should be ~1)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_batch_processing(self, embedding_model):
        """Test that batch processing works correctly."""
        # Create many texts to test batching
        texts = [f"This is test sentence number {i}." for i in range(50)]

        # Encode with small batch size
        embeddings = embedding_model.encode(texts)

        assert embeddings.shape == (len(texts), embedding_model.embedding_dim)


class TestFAISSIndex:
    """Tests for FAISSIndex class."""

    def test_index_initialization(self):
        """Test index initializes correctly."""
        index = FAISSIndex(dimension=384, index_type="Flat", metric="cosine")

        assert index.dimension == 384
        assert index.index_type == "Flat"
        assert index.metric == "cosine"
        assert index.index.ntotal == 0

    def test_add_embeddings(self, embedding_model, sample_chunks):
        """Test adding embeddings to index."""
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )

        index.add(embeddings, sample_chunks)

        assert index.index.ntotal == len(sample_chunks)
        assert len(index.chunk_metadata) == len(sample_chunks)

    def test_search_basic(self, embedding_model, sample_chunks):
        """Test basic search functionality."""
        # Build index
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, sample_chunks)

        # Search with first embedding (should return itself as top result)
        query_emb = embeddings[0:1]
        distances, indices = index.search(query_emb, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert indices[0][0] == 0  # First result should be the query itself

    def test_search_with_metadata(self, embedding_model, sample_chunks):
        """Test search returns correct metadata."""
        # Build index
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, sample_chunks)

        # Search
        query_emb = embeddings[0:1]
        results = index.search_with_metadata(query_emb, k=2)

        assert len(results) == 1  # One query
        assert len(results[0]) == 2  # Top-2 results

        # Check metadata structure
        result = results[0][0]
        assert "score" in result
        assert "chunk_id" in result
        assert "content" in result
        assert "doc_id" in result

    def test_search_relevance(self, embedding_model):
        """Test that search returns semantically similar results."""
        # Create chunks with clear semantic relationships
        texts = [
            "Cats are domestic animals.",
            "Dogs are loyal pets.",
            "Machine learning uses neural networks.",
            "Deep learning is a subset of AI.",
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
                metadata={},
            )
            chunks.append(chunk)

        # Build index
        embeddings = embedding_model.encode_chunks(chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, chunks)

        # Search with ML-related query
        query = "What is artificial intelligence?"
        query_emb = embedding_model.encode([query])
        results = index.search_with_metadata(query_emb, k=2)

        # Top results should be ML/AI related (indices 2 or 3)
        top_indices = [2, 3]  # ML and AI chunks
        retrieved_ids = [r["chunk_id"] for r in results[0]]

        # At least one of top 2 should be ML/AI related
        assert any(f"chunk_{i}" in retrieved_ids for i in top_indices)

    def test_save_and_load(self, embedding_model, sample_chunks):
        """Test saving and loading index."""
        # Build index
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, sample_chunks)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            index.save(str(save_path))

            # Load
            loaded_index = FAISSIndex.load(str(save_path))

            # Verify
            assert loaded_index.index.ntotal == index.index.ntotal
            assert loaded_index.dimension == index.dimension
            assert len(loaded_index.chunk_metadata) == len(index.chunk_metadata)

            # Test search works on loaded index
            query_emb = embeddings[0:1]
            results = loaded_index.search_with_metadata(query_emb, k=2)
            assert len(results[0]) == 2

    def test_get_stats(self, embedding_model, sample_chunks):
        """Test getting index statistics."""
        embeddings = embedding_model.encode_chunks(sample_chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, sample_chunks)

        stats = index.get_stats()

        assert stats["ntotal"] == len(sample_chunks)
        assert stats["dimension"] == embedding_model.embedding_dim
        assert stats["index_type"] == "Flat"
        assert stats["metric"] == "cosine"

    def test_dimension_mismatch(self):
        """Test that adding wrong dimension embeddings raises error."""
        index = FAISSIndex(dimension=384, index_type="Flat", metric="cosine")

        # Create embeddings with wrong dimension
        wrong_embeddings = np.random.randn(5, 512).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            index.add(wrong_embeddings)

    def test_empty_search(self):
        """Test that searching empty index raises error."""
        index = FAISSIndex(dimension=384, index_type="Flat", metric="cosine")

        query = np.random.randn(1, 384).astype(np.float32)

        with pytest.raises(ValueError, match="empty"):
            index.search(query, k=5)


class TestPerformance:
    """Performance-related tests."""

    def test_embedding_speed(self, embedding_model):
        """Test embedding generation speed."""
        import time

        texts = ["This is a test sentence."] * 100

        start = time.perf_counter()
        embeddings = embedding_model.encode(texts, show_progress=False)
        duration = time.perf_counter() - start

        # Should process 100 sentences in reasonable time
        assert duration < 10.0  # 10 seconds max on CPU
        assert embeddings.shape == (100, embedding_model.embedding_dim)

    def test_search_speed(self, embedding_model):
        """Test search speed."""
        import time

        # Create larger index
        n_chunks = 1000
        texts = [f"This is test sentence number {i}." for i in range(n_chunks)]

        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i}",
                start_char=0,
                end_char=len(text),
                chunk_index=i,
                metadata={},
            )
            chunks.append(chunk)

        # Build index
        embeddings = embedding_model.encode_chunks(chunks, show_progress=False)

        index = FAISSIndex(
            dimension=embedding_model.embedding_dim, index_type="Flat", metric="cosine"
        )
        index.add(embeddings, chunks)

        # Search
        query_emb = embeddings[0:1]

        start = time.perf_counter()
        results = index.search_with_metadata(query_emb, k=10)
        duration = time.perf_counter() - start

        # Search should be fast (<100ms for 1000 vectors)
        assert duration < 0.1
        assert len(results[0]) == 10


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from documents to search."""
        # Create documents
        docs = [
            Document(content="The cat sat on the mat.", doc_id="doc1"),
            Document(content="A dog played in the park.", doc_id="doc2"),
            Document(content="Machine learning is fascinating.", doc_id="doc3"),
        ]

        # Chunk documents
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_batch(docs)

        # Create embeddings
        embed_model = EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        embeddings = embed_model.encode_chunks(chunks, show_progress=False)

        # Build index
        index = FAISSIndex(dimension=embed_model.embedding_dim, index_type="Flat", metric="cosine")
        index.add(embeddings, chunks)

        # Search
        query = "What animal was on the mat?"
        query_emb = embed_model.encode([query])
        results = index.search_with_metadata(query_emb, k=2)

        # Verify results
        assert len(results[0]) > 0

        # First result should be about the cat
        top_result = results[0][0]
        assert "cat" in top_result["content"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
