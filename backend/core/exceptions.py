from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseCustomException(Exception):
    """Base exception class for custom exceptions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(BaseCustomException):
    """Exception raised during document processing."""
    pass


class EmbeddingServiceError(BaseCustomException):
    """Exception raised by embedding service."""
    pass


class LLMServiceError(BaseCustomException):
    """Exception raised by LLM service."""
    pass


class VectorStoreError(BaseCustomException):
    """Exception raised by vector store operations."""
    pass


class ValidationError(BaseCustomException):
    """Exception raised during data validation."""
    pass


class AuthenticationError(BaseCustomException):
    """Exception raised during authentication."""
    pass


class RateLimitError(BaseCustomException):
    """Exception raised when rate limit is exceeded."""
    pass


class ConfigurationError(BaseCustomException):
    """Exception raised for configuration errors."""
    pass


# HTTP Exception handlers
def create_http_exception(
        status_code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create HTTP exception with details."""
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "details": details or {},
            "error_type": "application_error"
        }
    )


# Specific HTTP exceptions
class DocumentNotFoundError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )


class FileTooLargeError(HTTPException):
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size} bytes exceeds maximum allowed size {max_size} bytes"
        )


class UnsupportedFileTypeError(HTTPException):
    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_type}' not supported. Supported types: {', '.join(supported_types)}"
        )


class ProcessingInProgressError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document {document_id} is currently being processed"
        )


class InsufficientContextError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient context found in documents to answer the question"
        )


class LLMProviderError(HTTPException):
    def __init__(self, provider: str, error_message: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider '{provider}' error: {error_message}"
        )


# Exception mapping for automatic HTTP exception creation
EXCEPTION_MAPPING = {
    DocumentProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    EmbeddingServiceError: status.HTTP_503_SERVICE_UNAVAILABLE,
    LLMServiceError: status.HTTP_503_SERVICE_UNAVAILABLE,
    VectorStoreError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ValidationError: status.HTTP_400_BAD_REQUEST,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def map_exception_to_http(exception: BaseCustomException) -> HTTPException:
    """Map custom exception to HTTP exception."""
    status_code = EXCEPTION_MAPPING.get(type(exception), status.HTTP_500_INTERNAL_SERVER_ERROR)
    return create_http_exception(status_code, exception.message, exception.details)
