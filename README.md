# GenAI Document Assistant

A sophisticated AI-powered document analysis and Q&A system built with FastAPI and Streamlit. Upload documents, ask questions, and test your knowledge through an interactive challenge mode.

## Features

### 🤖 AI-Powered Document Analysis
- **Multi-format Support**: PDF, DOCX, TXT, MD, HTML
- **Semantic Search**: Advanced vector-based document search
- **Context-Aware Q&A**: AI answers based on document content
- **Document Summarization**: Automatic key topic extraction

### 🎯 Interactive Challenge Mode
- **Question Generation**: AI creates questions from document content
- **Multiple Difficulty Levels**: Easy, Medium, Hard
- **Question Types**: Factual, Inferential, Analytical
- **Answer Evaluation**: AI evaluates responses with partial credit

### 🚀 Advanced Features
- **Hybrid Search**: Combines semantic and keyword search
- **Semantic Caching**: Reduces API calls for similar queries
- **Real-time Processing**: Background document processing
- **Comprehensive Analytics**: System metrics and usage statistics

## Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with async/await support
- **Database**: SQLAlchemy ORM with SQLite (configurable)
- **Vector Database**: FAISS for semantic search
- **Caching**: Redis for semantic caching
- **AI Services**: OpenAI and Anthropic integration

### Frontend (Streamlit)
- **Framework**: Streamlit for rapid web app development
- **Components**: Modular UI components
- **Styling**: Custom CSS with modern designs

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- OpenAI API key (or Anthropic API key)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd genai_document_assistant
```

### 2. Environment Setup
```bash
# Copy environment file
cp env.example .env

# Edit .env file with your API keys
# OPENAI_API_KEY=your-openai-api-key
# ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 3. Run with Docker (Recommended)
```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### 4. Run Locally (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend
streamlit run app.py --server.port 8501
```

## Usage

### 1. Upload Documents
- Navigate to the web interface
- Upload PDF, DOCX, TXT, or MD files
- Wait for processing to complete

### 2. Ask Questions
- Use "Ask Anything" mode to ask questions about your documents
- Get AI-powered answers based on document content
- View source references and confidence scores

### 3. Challenge Mode
- Generate questions from your documents
- Test your knowledge with interactive quizzes
- Get detailed feedback and explanations

## Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# LLM Configuration
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEFAULT_LLM_PROVIDER=openai

# Vector Database
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Document Processing
MAX_FILE_SIZE=104857600
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Feature Flags
Enable/disable features in `.env`:

```bash
ENABLE_SEMANTIC_CACHE=true
ENABLE_QUESTION_GENERATION=true
ENABLE_DOCUMENT_SUMMARIZATION=true
ENABLE_HYBRID_SEARCH=true
```

## API Documentation

### Endpoints

#### Documents
- `POST /api/v1/documents/upload` - Upload and process document
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document

#### Search & Q&A
- `POST /api/v1/search` - Search documents
- `POST /api/v1/conversations` - Ask questions

#### Challenge Mode
- `POST /api/v1/questions/generate` - Generate questions
- `POST /api/v1/answers/evaluate` - Evaluate answers

#### System
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - System statistics

### Interactive API Docs
Visit `http://localhost:8000/docs` for interactive API documentation.

## Development

### Project Structure
```
genai_document_assistant/
├── backend/
│   ├── api/           # FastAPI routes and dependencies
│   ├── core/          # Configuration, database, exceptions
│   ├── models/        # Pydantic schemas
│   ├── services/      # Business logic services
│   ├── utils/         # Utility functions
│   └── main.py        # FastAPI application entry point
├── frontend/
│   ├── components/    # Streamlit UI components
│   ├── utils/         # Frontend utilities
│   └── app.py         # Main Streamlit application
├── docker-compose.yml # Docker services configuration
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

### Adding New Features

#### Backend Services
1. Create service in `backend/services/`
2. Add schemas in `backend/models/schemas.py`
3. Create routes in `backend/api/routes.py`
4. Update dependencies in `backend/api/dependencies.py`

#### Frontend Components
1. Create component in `frontend/components/`
2. Import and use in `frontend/app.py`
3. Add utilities in `frontend/utils/` if needed

### Testing
```bash
# Run backend tests
cd backend
pytest

# Run frontend tests (if implemented)
cd frontend
pytest
```

## Deployment

### Production Setup
1. Set `DEBUG=false` in environment
2. Configure proper `SECRET_KEY`
3. Set up SSL certificates for nginx
4. Configure database (PostgreSQL recommended)
5. Set up monitoring and logging

### Docker Production
```bash
# Build production images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale backend=3 --scale frontend=2
```

## Troubleshooting

### Common Issues

#### 1. SpaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

#### 2. Redis Connection Error
- Ensure Redis is running: `docker-compose up redis`
- Check Redis URL in environment

#### 3. API Key Issues
- Verify API keys in `.env` file
- Check API provider status
- Ensure sufficient credits

#### 4. Memory Issues
- Reduce `MAX_CHUNKS_PER_DOCUMENT`
- Increase `EMBEDDING_BATCH_SIZE`
- Monitor system resources

### Logs
```bash
# View backend logs
docker-compose logs backend

# View frontend logs
docker-compose logs frontend

# View all logs
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API docs at `/docs`

---

**GenAI Document Assistant** - Advanced document analysis powered by AI 🤖
