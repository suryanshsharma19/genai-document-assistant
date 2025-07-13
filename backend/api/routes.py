from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import asyncio
import json
from pathlib import Path
import uuid
from datetime import datetime
import logging

from ..core.database import get_db, Document, DocumentChunk, Conversation, GeneratedQuestion, UserAnswer
from ..core.config import settings
from ..core.exceptions import *
from ..models.schemas import *
from ..services.document_processor import AdvancedDocumentProcessor
from ..services.embeddings_service import AdvancedEmbeddingsService
from ..services.llm_service import OpenAIProvider, AnthropicProvider
from ..services.question_generator import QuestionGenerator
from ..services.evaluation_service import AnswerEvaluationService
from ..api.dependencies import get_current_user, rate_limit

logger = logging.getLogger(__name__)

# Initialize services
document_processor = AdvancedDocumentProcessor()
embeddings_service = AdvancedEmbeddingsService()
question_generator = QuestionGenerator()
evaluation_service = AnswerEvaluationService()

# Create router
router = APIRouter()


# Document endpoints
@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Upload and process a document."""
    try:
        # Validate file
        if file.size > settings.MAX_FILE_SIZE:
            raise FileTooLargeError(file.size, settings.MAX_FILE_SIZE)

        file_extension = Path(file.filename).suffix.lower().lstrip('.')
        if file_extension not in settings.SUPPORTED_FORMATS:
            raise UnsupportedFileTypeError(file_extension, settings.SUPPORTED_FORMATS)

        # Save uploaded file
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Create database record
        document = Document(
            filename=f"{file_id}_{file.filename}",
            original_filename=file.filename,
            file_path=str(file_path),
            content_hash="",  # Will be updated after processing
            file_size=file.size,
            file_type=file_extension,
            processing_status="pending"
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # Process document in background
        background_tasks.add_task(process_document_background, document.id, file_path)

        return DocumentResponse.from_orm(document)

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(document_id: int, file_path: Path):
    """Background task to process uploaded document."""
    db = next(get_db())

    try:
        # Get document record
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return

        # Update status
        document.processing_status = "processing"
        db.commit()

        # Process document
        processed_doc = await document_processor.process_document(
            file_path, document.original_filename
        )

        # Update document record
        document.content_hash = processed_doc.content_hash
        document.total_chunks = processed_doc.total_chunks
        document.word_count = processed_doc.word_count
        document.page_count = processed_doc.metadata.pages or 0
        document.processing_time = processed_doc.processing_time
        document.summary = processed_doc.summary
        document.key_topics = processed_doc.key_topics
        document.metadata = processed_doc.metadata.dict()
        document.processing_status = "completed"
        document.processed_at = datetime.utcnow()

        # Save chunks to database
        for chunk in processed_doc.chunks:
            db_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.metadata.chunk_index,
                chunk_id=chunk.metadata.chunk_id,
                content=chunk.content,
                content_hash=chunk.metadata.chunk_id,
                word_count=chunk.metadata.word_count,
                char_count=chunk.metadata.char_count,
                metadata=chunk.metadata.dict(),
                estimated_page=chunk.metadata.estimated_page,
                section_title=chunk.metadata.section_title,
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION
            )
            db.add(db_chunk)

        db.commit()

        # Add to vector store
        await embeddings_service.add_documents(processed_doc.chunks)

        # Update indexing status
        document.is_indexed = True
        db.commit()

        logger.info(f"Successfully processed document {document_id}")

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")

        # Update status to failed
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = "failed"
            db.commit()

    finally:
        db.close()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
        page: int = 1,
        size: int = 10,
        status: Optional[ProcessingStatus] = None,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """List documents with pagination."""
    try:
        query = db.query(Document).filter(Document.is_active == True)

        if status:
            query = query.filter(Document.processing_status == status.value)

        total = query.count()
        documents = query.offset((page - 1) * size).limit(size).all()

        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            page=page,
            size=size,
            has_next=page * size < total,
            has_prev=page > 1
        )

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
        document_id: int,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Get document by ID."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise DocumentNotFoundError(str(document_id))

    return DocumentResponse.from_orm(document)


@router.delete("/documents/{document_id}")
async def delete_document(
        document_id: int,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Delete document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise DocumentNotFoundError(str(document_id))

    # Soft delete
    document.is_active = False
    db.commit()

    return {"message": "Document deleted successfully"}


# Search endpoints
@router.post("/search", response_model=SearchResponse)
async def search_documents(
        request: SearchRequest,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Search documents using semantic search."""
    try:
        start_time = datetime.utcnow()

        # Perform search
        if request.use_hybrid_search and settings.ENABLE_HYBRID_SEARCH:
            results = await embeddings_service.hybrid_search(
                request.query,
                request.k,
                settings.HYBRID_SEARCH_ALPHA
            )
        else:
            results = await embeddings_service.search(
                request.query,
                request.k,
                request.filters
            )

        search_time = (datetime.utcnow() - start_time).total_seconds()

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_time=search_time,
            used_cache=False  # TODO: Implement cache detection
        )

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation endpoints
@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
        request: ConversationRequest,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Create a new conversation."""
    try:
        start_time = datetime.utcnow()

        # Search for relevant context
        search_results = await embeddings_service.search(
            request.query,
            request.context_k,
            {"document_id": request.document_id} if request.document_id else None
        )

        # Generate response using LLM
        llm_provider = OpenAIProvider() if settings.DEFAULT_LLM_PROVIDER == "openai" else AnthropicProvider()

        response_text = ""
        async for chunk in llm_provider.generate_response(request.query, search_results):
            response_text += chunk

        response_time = (datetime.utcnow() - start_time).total_seconds()

        # Save conversation
        conversation = Conversation(
            session_id=request.session_id,
            document_id=request.document_id,
            mode=request.mode.value,
            user_input=request.query,
            assistant_response=response_text,
            context_chunks=[result.chunk_id for result in search_results if result.chunk_id],
            search_query=request.query,
            search_results_count=len(search_results),
            response_time=response_time
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return ConversationResponse(
            response=response_text,
            mode=request.mode,
            context_used=search_results,
            response_time=response_time,
            confidence_score=None,  # TODO: Implement confidence scoring
            session_id=request.session_id,
            conversation_id=conversation.id
        )

    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
        session_id: str,
        limit: int = 50,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Get conversation history for a session."""
    conversations = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .all()
    )

    return ConversationHistory(
        conversations=[
            {
                "id": conv.id,
                "mode": conv.mode,
                "user_input": conv.user_input,
                "assistant_response": conv.assistant_response,
                "created_at": conv.created_at.isoformat(),
                "response_time": conv.response_time
            }
            for conv in conversations
        ],
        session_id=session_id,
        total_conversations=len(conversations)
    )


# Question generation endpoints
@router.post("/questions/generate", response_model=QuestionGenerationResponse)
async def generate_questions(
        request: QuestionGenerationRequest,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Generate questions for a document."""
    try:
        start_time = datetime.utcnow()

        # Get document
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise DocumentNotFoundError(str(request.document_id))

        # Get document content
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == request.document_id).all()
        content = "\n".join([chunk.content for chunk in chunks[:10]])  # Limit content

        # Generate questions
        questions = await question_generator.generate_questions(
            content,
            request.num_questions,
            request.difficulty_level,
            request.question_types
        )

        # Save questions to database
        for question in questions:
            db_question = GeneratedQuestion(
                document_id=request.document_id,
                question_text=question.question_text,
                question_type=question.question_type.value,
                difficulty_level=question.difficulty_level.value,
                correct_answer=question.correct_answer,
                answer_explanation=question.answer_explanation,
                source_chunks=question.source_chunks,
                model_used=settings.DEFAULT_MODEL
            )
            db.add(db_question)

        db.commit()

        generation_time = (datetime.utcnow() - start_time).total_seconds()

        return QuestionGenerationResponse(
            questions=questions,
            document_id=request.document_id,
            generation_time=generation_time,
            model_used=settings.DEFAULT_MODEL
        )

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/questions/{document_id}", response_model=List[GeneratedQuestion])
async def get_questions(
        document_id: int,
        limit: int = 10,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Get generated questions for a document."""
    questions = (
        db.query(GeneratedQuestion)
        .filter(GeneratedQuestion.document_id == document_id)
        .limit(limit)
        .all()
    )

    return [
        GeneratedQuestion(
            question_text=q.question_text,
            question_type=QuestionType(q.question_type),
            difficulty_level=DifficultyLevel(q.difficulty_level),
            correct_answer=q.correct_answer,
            answer_explanation=q.answer_explanation,
            source_chunks=q.source_chunks or []
        )
        for q in questions
    ]


# Answer evaluation endpoints
@router.post("/answers/evaluate", response_model=AnswerEvaluationResponse)
async def evaluate_answer(
        request: AnswerEvaluationRequest,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Evaluate user answer."""
    try:
        # Get question
        question = db.query(GeneratedQuestion).filter(GeneratedQuestion.id == request.question_id).first()
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Convert to schema
        question_schema = GeneratedQuestion(
            question_text=question.question_text,
            question_type=QuestionType(question.question_type),
            difficulty_level=DifficultyLevel(question.difficulty_level),
            correct_answer=question.correct_answer,
            answer_explanation=question.answer_explanation,
            source_chunks=question.source_chunks or []
        )

        # Evaluate answer
        evaluation = await evaluation_service.evaluate_answer(
            question_schema,
            request.user_answer
        )

        # Save user answer
        user_answer = UserAnswer(
            question_id=request.question_id,
            session_id=request.session_id,
            user_response=request.user_answer,
            is_correct=evaluation.is_correct,
            partial_credit=evaluation.partial_credit,
            evaluation_score=evaluation.evaluation_score,
            evaluation_feedback=evaluation.feedback,
            evaluation_model=settings.DEFAULT_MODEL
        )

        db.add(user_answer)
        db.commit()

        return evaluation

    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# System endpoints
@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        services={
            "database": "healthy",
            "embeddings": "healthy",
            "llm": "healthy"
        }
    )


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_current_user)
):
    """Get system statistics."""
    try:
        # Database stats
        total_documents = db.query(Document).filter(Document.is_active == True).count()
        total_chunks = db.query(DocumentChunk).count()
        total_conversations = db.query(Conversation).count()
        total_questions = db.query(GeneratedQuestion).count()

        # Processing stats
        processing_times = [
            doc.processing_time for doc in
            db.query(Document).filter(Document.processing_time.isnot(None)).all()
        ]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        # Vector store stats
        vector_stats = await embeddings_service.get_document_statistics()

        return SystemStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_conversations=total_conversations,
            total_questions=total_questions,
            avg_processing_time=avg_processing_time,
            storage_used=vector_stats.get("index_size_mb", 0) * 1024 * 1024,  # Convert to bytes
            cache_hit_rate=0.0,  # TODO: Implement cache hit rate tracking
            uptime=0.0  # TODO: Implement uptime tracking
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
