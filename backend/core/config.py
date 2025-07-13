from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GenAI Document Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # Redis
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    CACHE_TTL: int = 3600

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.1

    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Vector Database
    VECTOR_DB_TYPE: str = "faiss"  # faiss, chroma, qdrant
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # Document Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS: List[str] = ["pdf", "txt", "docx", "md", "html"]
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"

    # Text Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_DOCUMENT: int = 1000

    # Search Configuration
    DEFAULT_SEARCH_K: int = 5
    MAX_SEARCH_K: int = 20
    SIMILARITY_THRESHOLD: float = 0.7
    HYBRID_SEARCH_ALPHA: float = 0.7

    # Question Generation
    DEFAULT_QUESTIONS_PER_DOCUMENT: int = 3
    MAX_QUESTIONS_PER_DOCUMENT: int = 10
    QUESTION_DIFFICULTIES: List[str] = ["easy", "medium", "hard"]

    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300
    RATE_LIMIT_PER_MINUTE: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "./logs/app.log"

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Feature Flags
    ENABLE_SEMANTIC_CACHE: bool = True
    ENABLE_QUESTION_GENERATION: bool = True
    ENABLE_DOCUMENT_SUMMARIZATION: bool = True
    ENABLE_HYBRID_SEARCH: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        for directory in [self.UPLOAD_DIR, self.PROCESSED_DIR,
                          Path(self.LOG_FILE).parent,
                          Path(self.DATABASE_URL.replace("sqlite:///", "")).parent]:
            Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
