"""
FastAPI routes and API dependencies.
"""

from .routes import router
from .dependencies import (
    get_current_user,
    rate_limit,
    get_document_processor,
    get_embeddings_service,
    get_llm_service,
    get_question_generator,
    get_evaluation_service
)

__all__ = [
    'router',
    'get_current_user',
    'rate_limit',
    'get_document_processor',
    'get_embeddings_service',
    'get_llm_service',
    'get_question_generator',
    'get_evaluation_service'
]
