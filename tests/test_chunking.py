"""
Unit tests for chunking strategies.

Tests cover:
- Correct chunk generation
- Overlap behavior
- Edge cases (empty docs, very long docs)
- Metadata preservation
"""

import pytest

from src.chunking import Chunk, FixedSizeChunker, SemanticChunker, SlidingWindowChunker
from src.loader import Document


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    content = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence. "
        "This is the fourth sentence. "
        "This is the fifth sentence."
    )
    return Document(content=content, doc_id="test_doc", metadata={"source": "test"})


@pytest.fixture
def long_document():
    """Create a long document for stress testing."""
    content = " ".join([f"Sentence number {i}." for i in range(1000)])
    return Document(content=content, doc_id="long_doc", metadata={"source": "test"})


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self, sample_document):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_id == sample_document.doc_id for c in chunks)

    def test_no_overlap(self, sample_document):
        """Test chunking without overlap."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=0)
        chunks = chunker.chunk(sample_document)

        # With no overlap, chunks should not share content
        # (except potentially at token boundaries)
        assert len(chunks) > 0

    def test_chunk_size_respected(self, sample_document):
        """Test that chunks don't exceed specified size."""
        chunk_size = 30
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=5)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            token_count = chunker.count_tokens(chunk.content)
            # Allow slight overflow due to tokenization
            assert token_count <= chunk_size + 2

    def test_overlap_creates_more_chunks(self, sample_document):
        """Test that overlap increases chunk count."""
        no_overlap = FixedSizeChunker(chunk_size=30, overlap=0)
        with_overlap = FixedSizeChunker(chunk_size=30, overlap=10)

        chunks_no_overlap = no_overlap.chunk(sample_document)
        chunks_with_overlap = with_overlap.chunk(sample_document)

        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_metadata_preserved(self, sample_document):
        """Test that document metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test"

    def test_chunk_indices_sequential(self, sample_document):
        """Test that chunk indices are sequential."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=-10)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=-5)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_empty_document(self):
        """Test handling of empty document."""
        empty_doc = Document(content="", doc_id="empty")
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(empty_doc)

        # Should return empty list or single empty chunk
        assert len(chunks) <= 1


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_basic_chunking(self, sample_document):
        """Test basic semantic chunking."""
        chunker = SemanticChunker(target_size=30, min_size=10, max_size=100)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_sentence_boundaries(self):
        """Test that chunks don't split sentences."""
        content = (
            "This is sentence one. " "This is sentence two. " "This is sentence three."
        )
        doc = Document(content=content, doc_id="test")

        chunker = SemanticChunker(target_size=20, min_size=5, max_size=50)
        chunks = chunker.chunk(doc)

        # Each chunk should end with sentence-ending punctuation
        # or be at the document end
        for chunk in chunks[:-1]:  # Check all but last chunk
            assert chunk.content.rstrip().endswith((".", "!", "?"))

    def test_size_constraints(self, sample_document):
        """Test that chunks respect size constraints."""
        min_size = 10
        max_size = 100
        chunker = SemanticChunker(target_size=30, min_size=min_size, max_size=max_size)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            token_count = chunker.count_tokens(chunk.content)
            # Chunks should be within bounds (with some tolerance)
            assert token_count <= max_size + 5

    def test_long_sentence_handling(self):
        """Test handling of sentences exceeding max_size."""
        # Create document with one very long sentence
        long_sentence = " ".join([f"word{i}" for i in range(500)]) + "."
        doc = Document(content=long_sentence, doc_id="long_sentence")

        chunker = SemanticChunker(target_size=50, min_size=10, max_size=100)
        chunks = chunker.chunk(doc)

        # Should split the long sentence
        assert len(chunks) > 1

    def test_metadata_preserved(self, sample_document):
        """Test metadata preservation."""
        chunker = SemanticChunker(target_size=30, min_size=10, max_size=100)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert "source" in chunk.metadata


class TestSlidingWindowChunker:
    """Tests for SlidingWindowChunker."""

    def test_basic_chunking(self, sample_document):
        """Test basic sliding window chunking."""
        chunker = SlidingWindowChunker(window_size=30, stride=20)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_overlap_calculation(self, sample_document):
        """Test that overlap is correctly calculated."""
        window_size = 30
        stride = 20
        chunker = SlidingWindowChunker(window_size=window_size, stride=stride)

        expected_overlap = window_size - stride
        assert chunker.overlap == expected_overlap

    def test_window_coverage(self, long_document):
        """Test that windows cover the entire document."""
        chunker = SlidingWindowChunker(window_size=50, stride=40)
        chunks = chunker.chunk(long_document)

        # First chunk should start at beginning
        assert chunks[0].start_char == 0

        # Last chunk should extend to (near) end
        # (May not be exact due to tokenization)
        doc_length = len(long_document.content)
        assert chunks[-1].end_char >= doc_length * 0.95

    def test_no_gaps_between_windows(self, sample_document):
        """Test that there are no gaps between windows."""
        chunker = SlidingWindowChunker(window_size=30, stride=25)
        chunks = chunker.chunk(sample_document)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # End of chunk i should be after or at start of chunk i+1
            assert chunks[i].end_char >= chunks[i + 1].start_char

    def test_stride_equals_window_no_overlap(self, sample_document):
        """Test that stride==window_size means no overlap."""
        window_size = 30
        chunker = SlidingWindowChunker(window_size=window_size, stride=window_size)
        chunks = chunker.chunk(sample_document)

        assert chunker.overlap == 0
        assert len(chunks) > 0

    def test_metadata_preserved(self, sample_document):
        """Test metadata preservation."""
        chunker = SlidingWindowChunker(window_size=30, stride=20)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert "source" in chunk.metadata


class TestChunkerComparison:
    """Compare behavior across different chunkers."""

    def test_all_chunkers_produce_chunks(self, sample_document):
        """Test that all chunkers produce chunks for same document."""
        fixed = FixedSizeChunker(chunk_size=30, overlap=10)
        semantic = SemanticChunker(target_size=30, min_size=10, max_size=100)
        sliding = SlidingWindowChunker(window_size=30, stride=20)

        fixed_chunks = fixed.chunk(sample_document)
        semantic_chunks = semantic.chunk(sample_document)
        sliding_chunks = sliding.chunk(sample_document)

        assert len(fixed_chunks) > 0
        assert len(semantic_chunks) > 0
        assert len(sliding_chunks) > 0

    def test_sliding_produces_most_chunks(self, long_document):
        """Test that sliding window typically produces most chunks."""
        fixed = FixedSizeChunker(chunk_size=50, overlap=10)
        sliding = SlidingWindowChunker(window_size=50, stride=30)

        fixed_chunks = fixed.chunk(long_document)
        sliding_chunks = sliding.chunk(long_document)

        # Sliding window with stride < window should produce more chunks
        assert len(sliding_chunks) >= len(fixed_chunks)

    def test_unique_chunk_ids(self, sample_document):
        """Test that all chunkers generate unique IDs."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_document)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
