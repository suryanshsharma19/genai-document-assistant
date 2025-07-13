"""
Core backend components including configuration, database, and exceptions.
"""

from .config import settings
from .database import get_db, create_tables, Base, SessionLocal
from .exceptions import *

__all__ = [
    'settings',
    'get_db',
    'create_tables',
    'Base',
    'SessionLocal',
    'BaseCustomException',
    'DocumentProcessingError',
    'EmbeddingServiceError',
    'LLMServiceError',
    'VectorStoreError',
    'ValidationError',
    'AuthenticationError',
    'RateLimitError',
    'ConfigurationError'
]
