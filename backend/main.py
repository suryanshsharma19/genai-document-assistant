from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from .core.config import settings
from .core.database import create_tables
from .api.routes import router
from .core.exceptions import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting GenAI Document Assistant Backend...")
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GenAI Document Assistant Backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Advanced document analysis and AI-powered Q&A system",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_V1_STR)


@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "document_processing_error"}
    )


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "embedding_service_error"}
    )


@app.exception_handler(LLMServiceError)
async def llm_service_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "llm_service_error"}
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "vector_store_error"}
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "validation_error"}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GenAI Document Assistant API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "services": {
            "database": "connected",
            "embeddings": "available",
            "llm": "available"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    ) 