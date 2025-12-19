"""
Embedding generation and vector indexing for the Advanced RAG system.

This module provides:
- EmbeddingModel: Wrapper for sentence-transformers models (BGE)
- FAISSIndex: Vector index for similarity search
- Batch processing and GPU acceleration
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk
from src.utils import GPUManager, Timer, ensure_dir


class EmbeddingModel:
    """
    Wrapper for sentence embedding models.

    Handles model loading, batch embedding generation,
    and GPU/CPU management.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        # Determine device
        if device is None:
            self.device = GPUManager.get_device(prefer_gpu=True)
        else:
            self.device = torch.device(device)

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}")

        # Load model
        with Timer("Model loading"):
            self.model = SentenceTransformer(model_name, device=str(self.device))

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Optimize for inference
        if self.device.type == "cuda":
            try:
                # Use FP16 for 2x speedup
                self.model.half()
                logger.info("Using FP16 precision for faster inference")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")

    def encode(
        self, texts: List[str], show_progress: bool = False, convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Embeddings as numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        with Timer(f"Encoding {len(texts)} texts"):
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=self.normalize_embeddings,
            )

        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def encode_queries(self, queries: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """
        Encode queries with optional instruction prefix.

        Some models (like BGE) benefit from task-specific instructions.

        Args:
            queries: List of query strings
            instruction: Optional instruction prefix

        Returns:
            Query embeddings
        """
        if instruction:
            # Add instruction prefix to each query
            queries = [f"{instruction} {q}" for q in queries]

        return self.encode(queries)

    def encode_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> np.ndarray:
        """
        Encode chunks into embeddings.

        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array
        """
        texts = [chunk.content for chunk in chunks]
        return self.encode(texts, show_progress=show_progress)

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


class FAISSIndex:
    """
    FAISS vector index for similarity search.

    Supports flat (exact) and approximate indexes.
    """

    def __init__(self, dimension: int, index_type: str = "Flat", metric: str = "cosine"):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        # Create index
        self.index = self._create_index()

        # Store chunk metadata (FAISS only stores vectors)
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.is_trained = False

        logger.info(
            f"Initialized FAISS index: type={index_type}, " f"metric={metric}, dim={dimension}"
        )

    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on type and metric.

        Returns:
            FAISS index object
        """
        if self.metric == "cosine":
            # For cosine similarity, use inner product
            # (assumes embeddings are normalized)
            if self.index_type == "Flat":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                raise NotImplementedError(f"Index type {self.index_type} not implemented")

        elif self.metric == "euclidean":
            if self.index_type == "Flat":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                raise NotImplementedError(f"Index type {self.index_type} not implemented")

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return index

    def add(self, embeddings: np.ndarray, chunks: Optional[List[Chunk]] = None) -> None:
        """
        Add embeddings to the index.

        Args:
            embeddings: Embeddings array (N, dimension)
            chunks: Optional list of corresponding Chunk objects
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # Ensure float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Add to index
        with Timer(f"Adding {len(embeddings)} vectors to index"):
            self.index.add(embeddings)

        # Store metadata
        if chunks:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")

            for chunk in chunks:
                self.chunk_metadata.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    }
                )

        logger.info(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query embedding(s) (N, dimension) or (dimension,)
            k: Number of neighbors to return

        Returns:
            Tuple of (distances, indices)
            - distances: (N, k) array of similarity scores
            - indices: (N, k) array of vector indices
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors before searching.")

        # Ensure 2D array
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Ensure float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        # Search
        with Timer(f"Searching for top-{k} neighbors"):
            distances, indices = self.index.search(query_embeddings, k)

        return distances, indices

    def search_with_metadata(
        self, query_embeddings: np.ndarray, k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Search and return results with chunk metadata.

        Args:
            query_embeddings: Query embedding(s)
            k: Number of neighbors to return

        Returns:
            List of result lists, one per query.
            Each result is a dict with 'score', 'chunk_id', 'content', etc.
        """
        distances, indices = self.search(query_embeddings, k)

        results = []

        for query_dists, query_indices in zip(distances, indices):
            query_results = []

            for score, idx in zip(query_dists, query_indices):
                if idx < 0 or idx >= len(self.chunk_metadata):
                    # Invalid index (can happen if k > ntotal)
                    continue

                metadata = self.chunk_metadata[idx].copy()
                metadata["score"] = float(score)
                query_results.append(metadata)

            results.append(query_results)

        return results

    def save(self, save_dir: str) -> None:
        """
        Save index and metadata to disk.

        Args:
            save_dir: Directory to save index files
        """
        save_path = Path(save_dir)
        ensure_dir(save_path)

        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        logger.info(f"Saved FAISS index to {index_file}")

        # Save metadata
        metadata_file = save_path / "metadata.pkl"
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "ntotal": self.index.ntotal,
            "chunk_metadata": self.chunk_metadata,
        }

        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved metadata to {metadata_file}")
        logger.info(f"Index contains {self.index.ntotal} vectors")

    @classmethod
    def load(cls, load_dir: str) -> "FAISSIndex":
        """
        Load index and metadata from disk.

        Args:
            load_dir: Directory containing saved index

        Returns:
            Loaded FAISSIndex object
        """
        load_path = Path(load_dir)

        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        # Create index instance
        index_obj = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
        )

        # Load FAISS index
        index_file = load_path / "index.faiss"
        index_obj.index = faiss.read_index(str(index_file))

        # Restore chunk metadata
        index_obj.chunk_metadata = metadata["chunk_metadata"]

        logger.info(f"Loaded FAISS index from {load_dir}")
        logger.info(f"Index contains {index_obj.index.ntotal} vectors")

        return index_obj

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "ntotal": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": self.is_trained,
            "metadata_count": len(self.chunk_metadata),
        }


if __name__ == "__main__":
    # Example usage
    from src.chunking import FixedSizeChunker
    from src.loader import Document
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Create sample documents
    docs = [
        Document(content="The cat sat on the mat.", doc_id="doc1"),
        Document(content="A dog played in the park.", doc_id="doc2"),
        Document(content="Machine learning is fascinating.", doc_id="doc3"),
    ]

    # Chunk documents
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_batch(docs)

    logger.info(f"Created {len(chunks)} chunks")

    # Initialize embedding model
    embed_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5",  # Smaller model for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Generate embeddings
    embeddings = embed_model.encode_chunks(chunks, show_progress=True)
    logger.info(f"Generated embeddings: {embeddings.shape}")

    # Create and populate index
    index = FAISSIndex(dimension=embed_model.embedding_dim, index_type="Flat", metric="cosine")

    index.add(embeddings, chunks)

    # Search
    query = "What animal was on the mat?"
    query_embedding = embed_model.encode([query])

    results = index.search_with_metadata(query_embedding, k=2)

    logger.info("\nSearch results:")
    for i, result in enumerate(results[0]):
        logger.info(f"Rank {i+1}: (score={result['score']:.4f})")
        logger.info(f"  Content: {result['content']}")

    # Save index
    index.save("outputs/test_index")
    logger.info("\nIndex saved successfully")

    # Load index
    loaded_index = FAISSIndex.load("outputs/test_index")
    logger.info(f"Index loaded: {loaded_index.get_stats()}")
