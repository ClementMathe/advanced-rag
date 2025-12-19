"""
Build FAISS index from SQuAD documents.

This script:
1. Loads SQuAD documents
2. Applies chunking strategy
3. Generates embeddings using BGE
4. Builds FAISS index
5. Saves index and metadata

Usage:
    python scripts/build_index.py --input data/squad/squad_v2_train_documents.json --output index/squad
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from loguru import logger

from src.chunking import Chunk, FixedSizeChunker, SemanticChunker, SlidingWindowChunker
from src.embeddings import EmbeddingModel, FAISSIndex
from src.loader import Document
from src.utils import GPUManager, LoggerConfig, Timer, ensure_dir


def load_documents(file_path: Path) -> List[Document]:
    """
    Load documents from JSON file.

    Args:
        file_path: Path to documents JSON

    Returns:
        List of Document objects
    """
    logger.info(f"Loading documents from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(content=item["content"], doc_id=item["doc_id"], metadata=item["metadata"])
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents")

    # Statistics
    lengths = [len(doc.content) for doc in documents]
    avg_length = sum(lengths) / len(lengths)
    logger.info(f"Average document length: {avg_length:.0f} chars")

    return documents


def apply_chunking(
    documents: List[Document], strategy: str, chunk_size: int, overlap: int
) -> List[Chunk]:
    """
    Apply chunking strategy to documents.

    Args:
        documents: List of documents
        strategy: Chunking strategy ('fixed', 'semantic', 'sliding')
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens (for fixed/sliding)

    Returns:
        List of Chunk objects
    """
    logger.info(f"Applying {strategy} chunking (size={chunk_size}, overlap={overlap})")

    if strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy == "semantic":
        chunker = SemanticChunker(
            target_size=chunk_size, min_size=chunk_size // 4, max_size=chunk_size * 2
        )
    elif strategy == "sliding":
        stride = chunk_size - overlap
        chunker = SlidingWindowChunker(window_size=chunk_size, stride=stride)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    with Timer("Chunking documents"):
        chunks = chunker.chunk_batch(documents)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    logger.info(f"Average chunks per document: {len(chunks) / len(documents):.2f}")

    # Statistics
    lengths = [len(chunk.content) for chunk in chunks]
    logger.info(f"Chunk size range: [{min(lengths)}, {max(lengths)}] chars")
    logger.info(f"Average chunk size: {sum(lengths) / len(lengths):.0f} chars")

    return chunks


def build_index(
    chunks: List[Chunk], model_name: str, batch_size: int, device: str
) -> tuple[FAISSIndex, EmbeddingModel]:
    """
    Build FAISS index from chunks.

    Args:
        chunks: List of Chunk objects
        model_name: HuggingFace model identifier
        batch_size: Batch size for embedding generation
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Tuple of (FAISSIndex, EmbeddingModel)
    """
    logger.info(f"Initializing embedding model: {model_name}")

    # Initialize model
    embed_model = EmbeddingModel(model_name=model_name, device=device, batch_size=batch_size)

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    with Timer("Embedding generation"):
        embeddings = embed_model.encode_chunks(chunks, show_progress=True)

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Show memory usage
    if device == "cuda":
        memory_info = embed_model.get_memory_usage()
        logger.info(f"GPU memory: {memory_info['allocated_gb']:.2f} GB allocated")

    # Create index
    logger.info("Building FAISS index...")
    index = FAISSIndex(dimension=embed_model.embedding_dim, index_type="Flat", metric="cosine")

    # Add embeddings to index
    index.add(embeddings, chunks)

    logger.info(f"Index built: {index.get_stats()}")

    return index, embed_model


def save_index_and_config(index: FAISSIndex, output_dir: Path, config: dict) -> None:
    """
    Save index and configuration.

    Args:
        index: FAISSIndex to save
        output_dir: Directory to save files
        config: Configuration dictionary
    """
    ensure_dir(output_dir)

    # Save index
    index.save(str(output_dir))

    # Save configuration
    config_file = output_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved configuration to {config_file}")

    # Save chunk statistics
    stats_file = output_dir / "stats.json"
    stats = {"index": index.get_stats(), "config": config}

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics to {stats_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build FAISS index from SQuAD documents")
    parser.add_argument(
        "--input",
        type=str,
        default="data/squad/squad_v2_train_documents.json",
        help="Path to input documents JSON",
    )
    parser.add_argument(
        "--output", type=str, default="index/squad", help="Output directory for index"
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-large-en-v1.5", help="HuggingFace model name"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=["fixed", "semantic", "sliding"],
        help="Chunking strategy",
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap in tokens (for fixed/sliding)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Setup logging
    LoggerConfig.setup(level="INFO")

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("=" * 80)
    logger.info("BUILDING FAISS INDEX")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Chunk size: {args.chunk_size} tokens")
    logger.info(f"Overlap: {args.overlap} tokens")
    logger.info(f"Device: {device}")

    # Check GPU
    if device == "cuda":
        logger.info(f"GPU detected: {GPUManager.get_gpu_memory_info()}")

    # Load documents
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run: python scripts/prepare_squad.py first")
        return

    documents = load_documents(input_path)

    # Apply chunking
    chunks = apply_chunking(
        documents, strategy=args.strategy, chunk_size=args.chunk_size, overlap=args.overlap
    )

    # Build index
    index, embed_model = build_index(
        chunks, model_name=args.model, batch_size=args.batch_size, device=device
    )

    # Save
    output_dir = Path(args.output)
    config = {
        "model_name": args.model,
        "embedding_dim": embed_model.embedding_dim,
        "chunking_strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "device": device,
    }

    save_index_and_config(index, output_dir, config)

    logger.info("=" * 80)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Index saved to: {output_dir}")
    logger.info(f"Total vectors: {index.index.ntotal}")
    logger.info(f"Dimension: {index.dimension}")
    logger.info("")
    logger.info("Next step: Test retrieval with sample queries")
    logger.info("  python scripts/test_retrieval.py --index_dir {output_dir}")


if __name__ == "__main__":
    main()
