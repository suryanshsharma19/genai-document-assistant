from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class InteractionMode(str, Enum):
    QA = "qa"
    CHALLENGE = "challenge"
    SUMMARY = "summary"

class QuestionType(str, Enum):
    FACTUAL = "factual"
    INFERENTIAL = "inferential"
    ANALYTICAL = "analytical"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

# Base schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        use_enum_values = True

# Document schemas
class DocumentMetadata(BaseModel):
    pages: Optional[int] = None
    tables: Optional[List[Dict[str, Any]]] = []
    images: Optional[int] = 0
    fonts: Optional[List[str]] = []
    page_layouts: Optional[List[Dict[str, Any]]] = []
    processing_time: Optional[float] = None
    file_size: Optional[int] = None
    content_hash: Optional[str] = None

class DocumentChunkMetadata(BaseModel):
    chunk_id: str
    filename: str
    chunk_index: int
    total_chunks: int
    word_count: int
    char_count: int
    estimated_page: Optional[int] = None
    section_title: Optional[str] = None
    source_metadata: Optional[Dict[str, Any]] = {}

class DocumentChunk(BaseSchema):
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None

class ProcessedDocument(BaseSchema):
    filename: str
    content_hash: str
    total_chunks: int
    chunks: List[DocumentChunk]
    metadata: DocumentMetadata
    summary: str = Field(..., max_length=500)
    word_count: int
    processing_time: float
    key_topics: Optional[List[str]] = []

class DocumentCreate(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., gt=0)
    file_type: str = Field(..., min_length=1, max_length=10)

class DocumentResponse(BaseSchema):
    id: int
    filename: str
    original_filename: str
    content_hash: str
    file_size: int
    file_type: str
    total_chunks: int
    word_count: int
    page_count: int
    processing_status: ProcessingStatus
    summary: Optional[str] = None
    key_topics: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    is_active: bool
    is_indexed: bool

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool

# Search schemas
class SearchResult(BaseSchema):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    document_id: Optional[int] = None
    chunk_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[int]] = None
    k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_hybrid_search: bool = Field(default=True)
    filters: Optional[Dict[str, Any]] = {}

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float
    used_cache: bool = False

# LLM schemas
class LLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    context: List[SearchResult] = Field(default_factory=list)
    model: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    stream: bool = Field(default=False)

class LLMResponse(BaseSchema):
    response: str
    model_used: str
    tokens_used: int
    response_time: float
    confidence_score: Optional[float] = None
    context_used: List[str] = Field(default_factory=list)

# Question generation schemas
class GeneratedQuestion(BaseSchema):
    question_text: str = Field(..., min_length=10, max_length=500)
    question_type: QuestionType
    difficulty_level: DifficultyLevel
    correct_answer: str = Field(..., min_length=1, max_length=1000)
    answer_explanation: Optional[str] = None
    source_chunks: List[str] = Field(default_factory=list)

class QuestionGenerationRequest(BaseModel):
    document_id: int
    num_questions: int = Field(default=3, ge=1, le=10)
    difficulty_level: Optional[DifficultyLevel] = None
    question_types: Optional[List[QuestionType]] = None

class QuestionGenerationResponse(BaseModel):
    questions: List[GeneratedQuestion]
    document_id: int
    generation_time: float
    model_used: str

# Answer evaluation schemas
class AnswerEvaluationRequest(BaseModel):
    question_id: int
    user_answer: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)

class AnswerEvaluationResponse(BaseModel):
    is_correct: bool
    partial_credit: float = Field(..., ge=0.0, le=1.0)
    evaluation_score: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    correct_answer: str
    explanation: Optional[str] = None
    evaluation_time: float

# Conversation schemas
class ConversationRequest(BaseModel):
    document_id: int
    query: str = Field(..., min_length=1, max_length=2000)
    mode: InteractionMode = InteractionMode.QA
    session_id: str = Field(..., min_length=1, max_length=100)
    context_k: int = Field(default=5, ge=1, le=10)

class ConversationResponse(BaseModel):
    response: str
    mode: InteractionMode
    context_used: List[SearchResult]
    response_time: float
    confidence_score: Optional[float] = None
    session_id: str
    conversation_id: int

class ConversationHistory(BaseModel):
    conversations: List[Dict[str, Any]]
    session_id: str
    total_conversations: int

# System schemas
class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    version: str
    services: Dict[str, str] = Field(default_factory=dict)

class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_conversations: int
    total_questions: int
    avg_processing_time: float
    storage_used: int  # bytes
    cache_hit_rate: float
    uptime: float  # seconds

class ErrorResponse(BaseModel):
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    error_type: str = "application_error"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Batch processing schemas
class BatchProcessingRequest(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=100)
    operations: List[str] = Field(..., min_items=1)  # ["summarize", "generate_questions", "reindex"]
    priority: int = Field(default=1, ge=1, le=5)

class BatchProcessingResponse(BaseModel):
    batch_id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    estimated_completion: Optional[datetime] = None
    created_at: datetime

# Validation helpers
@validator('query', 'prompt', 'user_answer')
def validate_text_input(cls, v):
    if not v or not v.strip():
        raise ValueError('Text input cannot be empty')
    return v.strip()

@validator('file_size')
def validate_file_size(cls, v):
    if v <= 0:
        raise ValueError('File size must be positive')
    return v

@validator('similarity_score', 'confidence_score', 'evaluation_score')
def validate_score(cls, v):
    if v is not None and (v < 0.0 or v > 1.0):
        raise ValueError('Score must be between 0.0 and 1.0')
    return v
