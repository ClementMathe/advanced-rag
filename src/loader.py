"""
Document loading utilities for the Advanced RAG system.

This module provides loaders for various document formats:
- Plain text (.txt)
- JSON (.json, .jsonl)
- CSV (.csv)
- PDF (.pdf)
- Markdown (.md)

Each loader returns a standardized Document object with metadata.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


@dataclass
class Document:
    """
    Standardized document representation.

    Attributes:
        content: Main text content of the document
        metadata: Additional information (source, date, etc.)
        doc_id: Unique identifier for the document
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        """Generate doc_id if not provided."""
        if self.doc_id is None:
            # Generate ID from content hash
            import hashlib

            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        """Return character count of content."""
        return len(self.content)

    def __repr__(self) -> str:
        """String representation showing content preview."""
        preview = (
            self.content[:100] + "..." if len(self.content) > 100 else self.content
        )
        return f"Document(id={self.doc_id}, length={len(self)}, preview='{preview}')"


class BaseLoader:
    """
    Base class for all document loaders.

    Provides common functionality and interface for specific loaders.
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize loader.

        Args:
            encoding: Text encoding to use when reading files
        """
        self.encoding = encoding
        self.documents_loaded = 0

    def load(self, path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a file or directory.

        Args:
            path: File or directory path

        Returns:
            List of Document objects
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            docs = self._load_file(path)
        elif path.is_dir():
            docs = self._load_directory(path)
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

        self.documents_loaded += len(docs)
        logger.info(f"Loaded {len(docs)} documents from {path}")

        return docs

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load documents from a single file.

        Must be implemented by subclasses.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        raise NotImplementedError("Subclasses must implement _load_file")

    def _load_directory(self, dir_path: Path) -> List[Document]:
        """
        Load documents from all compatible files in a directory.

        Args:
            dir_path: Path to the directory

        Returns:
            List of Document objects from all files
        """
        documents = []

        for file_path in dir_path.rglob(f"*{self.file_extension}"):
            try:
                docs = self._load_file(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return documents

    @property
    def file_extension(self) -> str:
        """
        File extension this loader handles.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement file_extension")


class TextLoader(BaseLoader):
    """
    Loader for plain text files (.txt).

    Treats each file as a single document.
    """

    @property
    def file_extension(self) -> str:
        return ".txt"

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            List containing a single Document
        """
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "txt",
            "loaded_at": datetime.now().isoformat(),
        }

        return [Document(content=content, metadata=metadata)]


class JSONLoader(BaseLoader):
    """
    Loader for JSON and JSONL files.

    Supports:
    - Single JSON object
    - JSON array of objects
    - JSONL (one JSON object per line)

    Extracts text from specified fields.
    """

    def __init__(
        self,
        content_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Initialize JSON loader.

        Args:
            content_field: Field name containing main text content
            metadata_fields: List of fields to include in metadata
            encoding: Text encoding
        """
        super().__init__(encoding)
        self.content_field = content_field
        self.metadata_fields = metadata_fields or []

    @property
    def file_extension(self) -> str:
        return ".json"

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a JSON or JSONL file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of Documents (one per JSON object)
        """
        documents = []

        # Check if JSONL (one JSON per line)
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            doc = self._parse_json_object(obj, file_path, line_num)
                            if doc:
                                documents.append(doc)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Invalid JSON at {file_path}:{line_num}: {e}"
                            )
        else:
            # Standard JSON
            with open(file_path, "r", encoding=self.encoding) as f:
                data = json.load(f)

            # Handle both single object and array
            if isinstance(data, list):
                for idx, obj in enumerate(data):
                    doc = self._parse_json_object(obj, file_path, idx)
                    if doc:
                        documents.append(doc)
            else:
                doc = self._parse_json_object(data, file_path, 0)
                if doc:
                    documents.append(doc)

        return documents

    def _parse_json_object(
        self, obj: Dict[str, Any], file_path: Path, index: int
    ) -> Optional[Document]:
        """
        Parse a single JSON object into a Document.

        Args:
            obj: JSON object dictionary
            file_path: Source file path
            index: Index of object in file

        Returns:
            Document or None if content field not found
        """
        # Extract content
        if self.content_field not in obj:
            logger.warning(
                f"Content field '{self.content_field}' not found in object "
                f"at {file_path}:{index}"
            )
            return None

        content = str(obj[self.content_field])

        # Extract metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "json",
            "index": index,
            "loaded_at": datetime.now().isoformat(),
        }

        # Add requested metadata fields
        for meta_field in self.metadata_fields:
            if meta_field in obj:
                metadata[meta_field] = obj[meta_field]

        return Document(content=content, metadata=metadata)


class CSVLoader(BaseLoader):
    """
    Loader for CSV files.

    Each row becomes a document. Text fields are concatenated.
    """

    def __init__(
        self,
        content_columns: Optional[List[str]] = None,
        separator: str = ", ",
        encoding: str = "utf-8",
    ):
        """
        Initialize CSV loader.

        Args:
            content_columns: Columns to concatenate for content (None = all)
            separator: Separator for concatenating columns
            encoding: Text encoding
        """
        super().__init__(encoding)
        self.content_columns = content_columns
        self.separator = separator

    @property
    def file_extension(self) -> str:
        return ".csv"

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of Documents (one per row)
        """
        import csv

        documents = []

        with open(file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, 1):
                # Determine which columns to use
                if self.content_columns:
                    columns = [col for col in self.content_columns if col in row]
                else:
                    columns = list(row.keys())

                # Concatenate column values
                content = self.separator.join(str(row[col]) for col in columns)

                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": "csv",
                    "row_number": row_num,
                    "loaded_at": datetime.now().isoformat(),
                }

                documents.append(Document(content=content, metadata=metadata))

        return documents


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files (.md).

    Preserves Markdown formatting for better chunking.
    """

    @property
    def file_extension(self) -> str:
        return ".md"

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a Markdown file.

        Args:
            file_path: Path to the Markdown file

        Returns:
            List containing a single Document
        """
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "markdown",
            "loaded_at": datetime.now().isoformat(),
        }

        return [Document(content=content, metadata=metadata)]


def get_loader(file_path: Union[str, Path], **kwargs) -> BaseLoader:
    """
    Get appropriate loader for a file based on extension.

    Args:
        file_path: Path to the file
        **kwargs: Additional arguments to pass to loader

    Returns:
        Appropriate loader instance

    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    loaders = {
        ".txt": TextLoader,
        ".json": JSONLoader,
        ".jsonl": JSONLoader,
        ".csv": CSVLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }

    if extension not in loaders:
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported: {', '.join(loaders.keys())}"
        )

    return loaders[extension](**kwargs)


if __name__ == "__main__":
    # Example usage
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Example: Load a text file
    text_loader = TextLoader()
    docs = text_loader.load("data/example.txt")

    for doc in docs:
        print(doc)
        print(f"Metadata: {doc.metadata}")
        print()
