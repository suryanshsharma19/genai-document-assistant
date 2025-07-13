"""
GenAI Document Assistant Backend
A comprehensive document analysis and AI-powered Q&A system.
"""

__version__ = "1.0.0"
__author__ = "GenAI Team"
__description__ = "Advanced document processing and AI assistant backend"

# Import main components for easier access
from .core.config import settings
from .core.database import get_db, create_tables
from .core.exceptions import *

__all__ = [
    'settings',
    'get_db',
    'create_tables',
    'DocumentProcessingError',
    'EmbeddingServiceError',
    'LLMServiceError',
    'VectorStoreError',
    'ValidationError'
]
