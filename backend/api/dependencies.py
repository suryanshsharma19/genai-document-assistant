from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import logging
import asyncio
from functools import wraps
import time

from ..core.config import settings
from ..core.database import get_db
from ..core.exceptions import AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}


async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current user from JWT token."""
    if not credentials:
        # For demo purposes, return anonymous user
        return {"user_id": "anonymous", "username": "anonymous"}

    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token")

        # In production, validate user exists in database
        return {"user_id": user_id, "username": payload.get("username", "user")}

    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """Rate limiting decorator."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user identifier (IP or user ID)
            user_id = "anonymous"  # In production, extract from request

            current_time = time.time()
            window_start = current_time - window_seconds

            # Clean old entries
            if user_id in rate_limit_storage:
                rate_limit_storage[user_id] = [
                    timestamp for timestamp in rate_limit_storage[user_id]
                    if timestamp > window_start
                ]
            else:
                rate_limit_storage[user_id] = []

            # Check rate limit
            if len(rate_limit_storage[user_id]) >= max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )

            # Add current request
            rate_limit_storage[user_id].append(current_time)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_document_access(document_id: int):
    """Dependency to check document access permissions."""

    def dependency(
            db: Session = Depends(get_db),
            current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        # In production, implement proper access control
        # For now, allow all authenticated users
        return True

    return dependency


def validate_file_upload():
    """Dependency to validate file upload parameters."""

    def dependency():
        # Add file validation logic here
        return True

    return dependency


class DatabaseManager:
    """Database dependency manager."""

    def __init__(self):
        self.db_session = None

    async def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        if not self.db_session:
            self.db_session = next(get_db())
        return self.db_session

    async def close_session(self):
        """Close database session."""
        if self.db_session:
            self.db_session.close()
            self.db_session = None


# Service dependencies
class ServiceDependencies:
    """Container for service dependencies."""

    def __init__(self):
        self._document_processor = None
        self._embeddings_service = None
        self._llm_service = None
        self._question_generator = None
        self._evaluation_service = None

    @property
    def document_processor(self):
        if not self._document_processor:
            from ..services.document_processor import AdvancedDocumentProcessor
            self._document_processor = AdvancedDocumentProcessor()
        return self._document_processor

    @property
    def embeddings_service(self):
        if not self._embeddings_service:
            from ..services.embeddings_service import AdvancedEmbeddingsService
            self._embeddings_service = AdvancedEmbeddingsService()
        return self._embeddings_service

    @property
    def llm_service(self):
        if not self._llm_service:
            from ..services.llm_service import OpenAIProvider, AnthropicProvider
            if settings.DEFAULT_LLM_PROVIDER == "openai":
                self._llm_service = OpenAIProvider()
            else:
                self._llm_service = AnthropicProvider()
        return self._llm_service

    @property
    def question_generator(self):
        if not self._question_generator:
            from ..services.question_generator import QuestionGenerator
            self._question_generator = QuestionGenerator()
        return self._question_generator

    @property
    def evaluation_service(self):
        if not self._evaluation_service:
            from ..services.evaluation_service import AnswerEvaluationService
            self._evaluation_service = AnswerEvaluationService()
        return self._evaluation_service


# Global service container
services = ServiceDependencies()


def get_document_processor():
    """Get document processor service."""
    return services.document_processor


def get_embeddings_service():
    """Get embeddings service."""
    return services.embeddings_service


def get_llm_service():
    """Get LLM service."""
    return services.llm_service


def get_question_generator():
    """Get question generator service."""
    return services.question_generator


def get_evaluation_service():
    """Get evaluation service."""
    return services.evaluation_service


# Validation dependencies
def validate_document_id(document_id: int):
    """Validate document ID format."""
    if document_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID"
        )
    return document_id


def validate_session_id(session_id: str):
    """Validate session ID format."""
    if not session_id or len(session_id) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID"
        )
    return session_id


def validate_pagination(page: int = 1, size: int = 10):
    """Validate pagination parameters."""
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be >= 1"
        )

    if size < 1 or size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Size must be between 1 and 100"
        )

    return {"page": page, "size": size}


# Background task dependencies
class BackgroundTaskManager:
    """Manager for background tasks."""

    def __init__(self):
        self.tasks = {}

    def add_task(self, task_id: str, task):
        """Add a background task."""
        self.tasks[task_id] = task

    def get_task_status(self, task_id: str):
        """Get task status."""
        return self.tasks.get(task_id, {}).get("status", "not_found")

    def remove_task(self, task_id: str):
        """Remove completed task."""
        if task_id in self.tasks:
            del self.tasks[task_id]


# Global task manager
task_manager = BackgroundTaskManager()


def get_task_manager():
    """Get background task manager."""
    return task_manager


# Cache dependencies
class CacheManager:
    """Cache manager for request caching."""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}

    def get(self, key: str):
        """Get cached value."""
        if key in self.cache:
            if time.time() < self.cache_ttl.get(key, 0):
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                if key in self.cache_ttl:
                    del self.cache_ttl[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set cached value."""
        self.cache[key] = value
        self.cache_ttl[key] = time.time() + ttl

    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.cache_ttl.clear()


# Global cache manager
cache_manager = CacheManager()


def get_cache_manager():
    """Get cache manager."""
    return cache_manager
