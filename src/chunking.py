"""
Text chunking strategies for the Advanced RAG system.

Implements three chunking strategies:
1. Fixed-size chunking: Simple, predictable chunks
2. Semantic chunking: Respects sentence/paragraph boundaries
3. Sliding window chunking: Overlapping windows for maximum coverage

Each strategy returns Chunk objects with metadata for tracking.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from src.loader import Document


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.

    Attributes:
        content: The chunk text content
        chunk_id: Unique identifier for this chunk
        doc_id: ID of parent document
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        chunk_index: Index of this chunk in the document's chunk sequence
        metadata: Additional information from parent document
    """

    content: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return character count."""
        return len(self.content)

    def __repr__(self) -> str:
        """String representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"Chunk(id={self.chunk_id}, doc={self.doc_id}, "
            f"index={self.chunk_index}, length={len(self)}, "
            f"preview='{preview}')"
        )


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    Provides common functionality and interface.
    """

    def __init__(self, tokenizer_name: str = "gpt2"):
        """
        Initialize chunker.

        Args:
            tokenizer_name: Name of tokenizer to use for token counting
        """
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self.chunks_created = 0

    @property
    def tokenizer(self):
        """Lazy-load tokenizer to avoid import overhead."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.

        Must be implemented by subclasses.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        pass

    def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
            f"using {self.__class__.__name__}"
        )

        return all_chunks

    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        metadata: dict,
    ) -> Chunk:
        """
        Create a Chunk object with generated ID.

        Args:
            content: Chunk text
            doc_id: Parent document ID
            start_char: Start position in document
            end_char: End position in document
            chunk_index: Chunk index in sequence
            metadata: Metadata from parent document

        Returns:
            Chunk object
        """
        import hashlib

        # Generate unique chunk ID from content and position
        chunk_id = hashlib.md5(
            f"{doc_id}_{chunk_index}_{start_char}".encode()
        ).hexdigest()[:16]

        self.chunks_created += 1

        return Chunk(
            content=content,
            chunk_id=chunk_id,
            doc_id=doc_id,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            metadata=metadata.copy(),
        )


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking strategy.

    Splits text into chunks of approximately equal size, measured in tokens.
    Optionally adds overlap between consecutive chunks.

    Advantages:
    - Simple and fast
    - Predictable chunk sizes
    - Easy to tune

    Disadvantages:
    - Ignores semantic boundaries
    - May split sentences/paragraphs
    """

    def __init__(
        self, chunk_size: int = 512, overlap: int = 50, tokenizer_name: str = "gpt2"
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target size in tokens
            overlap: Overlap between chunks in tokens
            tokenizer_name: Tokenizer to use
        """
        super().__init__(tokenizer_name)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

        logger.info(
            f"Initialized FixedSizeChunker: "
            f"size={chunk_size}, overlap={overlap}, stride={self.stride}"
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk document into fixed-size pieces.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        chunk_index = 0
        position = 0

        while position < len(tokens):
            # Extract chunk tokens
            end_position = min(position + self.chunk_size, len(tokens))
            chunk_tokens = tokens[position:end_position]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Estimate character positions (approximate)
            # Note: Token-to-character mapping is not perfect
            char_start = int(position / len(tokens) * len(text))
            char_end = int(end_position / len(tokens) * len(text))

            # Create chunk
            chunk = self._create_chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                start_char=char_start,
                end_char=char_end,
                chunk_index=chunk_index,
                metadata=document.metadata,
            )
            chunks.append(chunk)

            chunk_index += 1
            position += self.stride

        logger.debug(
            f"Document {document.doc_id}: "
            f"{len(tokens)} tokens → {len(chunks)} chunks"
        )

        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy.

    Splits text at natural boundaries (sentences, paragraphs) while
    respecting a target size. Preserves semantic coherence.

    Advantages:
    - Chunks are semantically meaningful
    - Respects document structure
    - Better for retrieval quality

    Disadvantages:
    - Variable chunk sizes
    - Slightly slower than fixed-size
    - Requires sentence detection
    """

    def __init__(
        self,
        target_size: int = 512,
        min_size: int = 100,
        max_size: int = 1000,
        tokenizer_name: str = "gpt2",
    ):
        """
        Initialize semantic chunker.

        Args:
            target_size: Target chunk size in tokens
            min_size: Minimum chunk size in tokens
            max_size: Maximum chunk size in tokens (hard limit)
            tokenizer_name: Tokenizer to use
        """
        super().__init__(tokenizer_name)

        if not (min_size < target_size < max_size):
            raise ValueError("Must have: min_size < target_size < max_size")

        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size

        # Sentence splitting regex
        # Matches: . ! ? followed by space and capital letter
        self.sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

        logger.info(
            f"Initialized SemanticChunker: "
            f"target={target_size}, min={min_size}, max={max_size}"
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk document at semantic boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content

        # Split into paragraphs first (double newline)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_tokens = 0
        char_position = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Split paragraph into sentences
            sentences = self._split_sentences(paragraph)

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)

                # If single sentence exceeds max_size, split it forcefully
                if sentence_tokens > self.max_size:
                    # Flush current chunk if any
                    if current_chunk:
                        chunk = self._create_chunk_from_sentences(
                            current_chunk, document, char_position, chunk_index
                        )
                        chunks.append(chunk)
                        char_position += len("".join(current_chunk))
                        chunk_index += 1
                        current_chunk = []
                        current_tokens = 0

                    # Force-split long sentence
                    sub_chunks = self._force_split_sentence(
                        sentence, document, char_position, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    char_position += len(sentence)
                    chunk_index += len(sub_chunks)
                    continue

                # Check if adding this sentence exceeds target
                if (
                    current_tokens + sentence_tokens > self.target_size
                    and current_chunk
                ):
                    # Current chunk is ready, save it
                    chunk = self._create_chunk_from_sentences(
                        current_chunk, document, char_position, chunk_index
                    )
                    chunks.append(chunk)
                    char_position += len("".join(current_chunk))
                    chunk_index += 1

                    # Start new chunk with current sentence
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

        # Don't forget last chunk
        if current_chunk:
            chunk = self._create_chunk_from_sentences(
                current_chunk, document, char_position, chunk_index
            )
            chunks.append(chunk)

        logger.debug(
            f"Document {document.doc_id}: "
            f"{len(text)} chars → {len(chunks)} semantic chunks"
        )

        return chunks

    def _split_sentences(self, paragraph: str) -> List[str]:
        """
        Split paragraph into sentences.

        Args:
            paragraph: Text to split

        Returns:
            List of sentences
        """
        sentences = self.sentence_pattern.split(paragraph)
        # Clean and filter empty
        return [s.strip() for s in sentences if s.strip()]

    def _force_split_sentence(
        self, sentence: str, document: Document, char_position: int, chunk_index: int
    ) -> List[Chunk]:
        """
        Force-split a very long sentence that exceeds max_size.

        Args:
            sentence: Long sentence to split
            document: Parent document
            char_position: Current character position
            chunk_index: Starting chunk index

        Returns:
            List of chunks
        """
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        chunks = []

        position = 0
        while position < len(tokens):
            end_position = min(position + self.max_size, len(tokens))
            chunk_tokens = tokens[position:end_position]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            char_start = char_position + int(position / len(tokens) * len(sentence))
            char_end = char_position + int(end_position / len(tokens) * len(sentence))

            chunk = self._create_chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                start_char=char_start,
                end_char=char_end,
                chunk_index=chunk_index + len(chunks),
                metadata=document.metadata,
            )
            chunks.append(chunk)

            position += self.max_size

        return chunks

    def _create_chunk_from_sentences(
        self,
        sentences: List[str],
        document: Document,
        start_char: int,
        chunk_index: int,
    ) -> Chunk:
        """
        Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to combine
            document: Parent document
            start_char: Starting character position
            chunk_index: Chunk index

        Returns:
            Chunk object
        """
        content = " ".join(sentences)
        end_char = start_char + len(content)

        return self._create_chunk(
            content=content,
            doc_id=document.doc_id,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            metadata=document.metadata,
        )


class SlidingWindowChunker(BaseChunker):
    """
    Sliding window chunking strategy.

    Creates overlapping windows that slide across the text.
    Ensures no information is lost at chunk boundaries.

    Advantages:
    - Maximum coverage (no gaps)
    - Robust to boundary effects
    - Good for dense information

    Disadvantages:
    - High redundancy
    - Larger index size
    - May retrieve duplicate content
    """

    def __init__(
        self, window_size: int = 512, stride: int = 384, tokenizer_name: str = "gpt2"
    ):
        """
        Initialize sliding window chunker.

        Args:
            window_size: Size of each window in tokens
            stride: Step size between windows in tokens
            tokenizer_name: Tokenizer to use
        """
        super().__init__(tokenizer_name)

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride >= window_size:
            logger.warning("stride >= window_size means no overlap between windows")

        self.window_size = window_size
        self.stride = stride
        self.overlap = window_size - stride

        logger.info(
            f"Initialized SlidingWindowChunker: "
            f"window={window_size}, stride={stride}, overlap={self.overlap}"
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk document using sliding windows.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        chunk_index = 0
        position = 0

        while position < len(tokens):
            # Extract window tokens
            end_position = min(position + self.window_size, len(tokens))
            window_tokens = tokens[position:end_position]

            # Decode back to text
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)

            # Estimate character positions
            char_start = int(position / len(tokens) * len(text))
            char_end = int(end_position / len(tokens) * len(text))

            # Create chunk
            chunk = self._create_chunk(
                content=window_text,
                doc_id=document.doc_id,
                start_char=char_start,
                end_char=char_end,
                chunk_index=chunk_index,
                metadata=document.metadata,
            )
            chunks.append(chunk)

            chunk_index += 1

            # Move window by stride
            position += self.stride

            # Stop if we've reached the end
            if end_position == len(tokens):
                break

        logger.debug(
            f"Document {document.doc_id}: "
            f"{len(tokens)} tokens → {len(chunks)} windows "
            f"(overlap={self.overlap} tokens)"
        )

        return chunks


if __name__ == "__main__":
    # Example usage
    from src.loader import Document
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="DEBUG")

    # Sample document
    doc = Document(
        content=(
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence. Here is another one. "
            "And yet another sentence to make this longer. "
            "We need enough text to demonstrate chunking strategies. "
            "Each strategy will split this differently."
        ),
        doc_id="test_doc_1",
    )

    # Test all strategies
    print("=== Fixed-Size Chunking ===")
    fixed_chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    fixed_chunks = fixed_chunker.chunk(doc)
    for chunk in fixed_chunks:
        print(chunk)

    print("\n=== Semantic Chunking ===")
    semantic_chunker = SemanticChunker(target_size=50, min_size=20, max_size=100)
    semantic_chunks = semantic_chunker.chunk(doc)
    for chunk in semantic_chunks:
        print(chunk)

    print("\n=== Sliding Window Chunking ===")
    sliding_chunker = SlidingWindowChunker(window_size=50, stride=30)
    sliding_chunks = sliding_chunker.chunk(doc)
    for chunk in sliding_chunks:
        print(chunk)
