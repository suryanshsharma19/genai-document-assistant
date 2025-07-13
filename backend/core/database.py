from sqlalchemy import create_engine, MetaData, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
from typing import Generator
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Database engine configuration
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DEBUG
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        echo=settings.DEBUG
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), unique=True, index=True)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(10), nullable=False)

    # Processing metadata
    total_chunks = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed

    # Content
    summary = Column(Text)
    key_topics = Column(JSON)
    metadata = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)

    # Flags
    is_active = Column(Boolean, default=True)
    is_indexed = Column(Boolean, default=False)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_id = Column(String(100), unique=True, index=True)

    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), index=True)
    word_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)

    # Metadata
    metadata = Column(JSON)
    estimated_page = Column(Integer)
    section_title = Column(String(255))

    # Vector data
    embedding_model = Column(String(100))
    embedding_dimension = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    document_id = Column(Integer, index=True)

    # Interaction data
    mode = Column(String(20), nullable=False)  # qa, challenge, summary
    user_input = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)

    # Context and metadata
    context_chunks = Column(JSON)  # List of chunk IDs used
    search_query = Column(Text)
    search_results_count = Column(Integer, default=0)
    response_time = Column(Float, default=0.0)

    # Quality metrics
    confidence_score = Column(Float)
    relevance_score = Column(Float)
    user_feedback = Column(Integer)  # 1-5 rating

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class GeneratedQuestion(Base):
    __tablename__ = "generated_questions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)

    # Question data
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)  # factual, inferential, analytical
    difficulty_level = Column(String(20), nullable=False)  # easy, medium, hard

    # Answer data
    correct_answer = Column(Text, nullable=False)
    answer_explanation = Column(Text)
    source_chunks = Column(JSON)  # Chunk IDs that contain the answer

    # Metadata
    generation_prompt = Column(Text)
    model_used = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class UserAnswer(Base):
    __tablename__ = "user_answers"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, index=True)
    session_id = Column(String(100), index=True)

    # Answer data
    user_response = Column(Text, nullable=False)
    is_correct = Column(Boolean)
    partial_credit = Column(Float, default=0.0)  # 0.0 to 1.0

    # Evaluation
    evaluation_score = Column(Float)
    evaluation_feedback = Column(Text)
    evaluation_model = Column(String(100))

    # Timing
    time_taken = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram

    # Metadata
    labels = Column(JSON)
    description = Column(Text)

    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)


# Database operations
def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise


def drop_tables():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {str(e)}")
        raise


# Database utilities
class DatabaseManager:
    """Database management utilities."""

    @staticmethod
    def get_document_by_hash(db: Session, content_hash: str) -> Document:
        """Get document by content hash."""
        return db.query(Document).filter(Document.content_hash == content_hash).first()

    @staticmethod
    def get_document_chunks(db: Session, document_id: int) -> list[DocumentChunk]:
        """Get all chunks for a document."""
        return db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()

    @staticmethod
    def get_conversation_history(db: Session, session_id: str, limit: int = 50) -> list[Conversation]:
        """Get conversation history for a session."""
        return (db.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
                .all())

    @staticmethod
    def record_metric(db: Session, name: str, value: float, metric_type: str = "gauge", labels: dict = None):
        """Record a system metric."""
        metric = SystemMetrics(
            metric_name=name,
            metric_value=value,
            metric_type=metric_type,
            labels=labels or {}
        )
        db.add(metric)
        db.commit()

    @staticmethod
    def cleanup_old_data(db: Session, days: int = 30):
        """Clean up old data."""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Clean old conversations
        db.query(Conversation).filter(Conversation.created_at < cutoff_date).delete()

        # Clean old metrics
        db.query(SystemMetrics).filter(SystemMetrics.recorded_at < cutoff_date).delete()

        db.commit()
        logger.info(f"Cleaned up data older than {days} days")


# Initialize database
if __name__ == "__main__":
    create_tables()
