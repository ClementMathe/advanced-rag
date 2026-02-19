"""
Production API layer for the Advanced RAG system.

Provides:
- create_app: FastAPI application factory
- Settings: Configuration via environment variables
"""

from src.api.app import create_app
from src.api.config import Settings

__all__ = [
    "create_app",
    "Settings",
]
