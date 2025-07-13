import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import json
import hashlib
import time

from ..core.config import settings
from ..models.schemas import DocumentChunk, SearchResult
from ..utils.cache import SemanticCache
from ..core.exceptions import EmbeddingServiceError, VectorStoreError

logger = logging.getLogger(__name__)


class AdvancedEmbeddingsService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index = None
        self.document_store = {}
        self.metadata_store = {}
        self.semantic_cache = SemanticCache()
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index with advanced configuration."""
        try:
            # Use HNSW for better performance with larger datasets
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128

            # Load existing index if available
            index_path = Path(settings.FAISS_INDEX_PATH)
            if index_path.exists():
                self._load_index()

        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise EmbeddingServiceError(f"Failed to initialize vector store: {str(e)}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store with batch processing."""
        try:
            if not chunks:
                return

            # Generate embeddings in batches
            batch_size = settings.EMBEDDING_BATCH_SIZE
            embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch]

                # Generate embeddings
                batch_embeddings = await self._generate_embeddings_async(batch_texts)
                embeddings.extend(batch_embeddings)

            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')

            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings_array)

            # Store document metadata
            for i, chunk in enumerate(chunks):
                doc_id = start_idx + i
                self.document_store[doc_id] = chunk.content
                self.metadata_store[doc_id] = chunk.metadata.dict()
                chunk.embedding = embeddings[i]

            # Save index
            await self._save_index()

            logger.info(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise EmbeddingServiceError(f"Failed to add documents: {str(e)}")

    async def search(
            self,
            query: str,
            k: int = 5,
            filter_metadata: Optional[Dict[str, Any]] = None,
            use_cache: bool = True
    ) -> List[SearchResult]:
        """Advanced semantic search with filtering and caching."""
        try:
            # Check semantic cache first
            if use_cache and settings.ENABLE_SEMANTIC_CACHE:
                cached_result = await self.semantic_cache.get(query)
                if cached_result:
                    logger.info("Retrieved result from semantic cache")
                    return cached_result[:k]

            # Generate query embedding
            query_embedding = await self._generate_embeddings_async([query])
            query_vector = np.array(query_embedding).astype('float32')

            # Perform search with larger k for filtering
            search_k = min(k * 3, self.index.ntotal) if filter_metadata else k
            distances, indices = self.index.search(query_vector, search_k)

            # Convert results to SearchResult objects
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                # Apply metadata filtering
                if filter_metadata and not self._matches_filter(idx, filter_metadata):
                    continue

                # Calculate similarity score (convert distance to similarity)
                similarity = 1 / (1 + distance)

                result = SearchResult(
                    content=self.document_store[idx],
                    metadata=self.metadata_store[idx],
                    similarity_score=float(similarity),
                    rank=len(results) + 1,
                    chunk_id=self.metadata_store[idx].get('chunk_id')
                )
                results.append(result)

                if len(results) >= k:
                    break

            # Cache the results
            if use_cache and results and settings.ENABLE_SEMANTIC_CACHE:
                await self.semantic_cache.set(query, results)

            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")

    async def hybrid_search(
            self,
            query: str,
            k: int = 5,
            alpha: float = 0.7
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword matching."""
        try:
            # Semantic search
            semantic_results = await self.search(query, k * 2, use_cache=False)

            # Keyword search
            keyword_results = self._keyword_search(query, k * 2)

            # Combine and re-rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, alpha
            )

            return combined_results[:k]

        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            raise VectorStoreError(f"Hybrid search failed: {str(e)}")

    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """Simple keyword-based search using TF-IDF-like scoring."""
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc_id, content in self.document_store.items():
            content_terms = set(content.lower().split())

            # Calculate simple overlap score
            overlap = len(query_terms.intersection(content_terms))
            if overlap > 0:
                # Simple TF-IDF approximation
                tf_score = overlap / len(content_terms)
                idf_score = len(self.document_store) / (overlap + 1)
                score = tf_score * np.log(idf_score)

                result = SearchResult(
                    content=content,
                    metadata=self.metadata_store[doc_id],
                    similarity_score=score,
                    rank=0,
                    chunk_id=self.metadata_store[doc_id].get('chunk_id')
                )
                scored_docs.append(result)

        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x.similarity_score, reverse=True)
        for i, result in enumerate(scored_docs[:k]):
            result.rank = i + 1

        return scored_docs[:k]

    def _combine_search_results(
            self,
            semantic_results: List[SearchResult],
            keyword_results: List[SearchResult],
            alpha: float
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results using RRF."""
        doc_scores = {}

        # Add semantic scores
        for result in semantic_results:
            doc_key = self._get_doc_key(result)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + alpha * (1 / (result.rank + 60))

        # Add keyword scores
        for result in keyword_results:
            doc_key = self._get_doc_key(result)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + (1 - alpha) * (1 / (result.rank + 60))

        # Create combined results
        all_results = {self._get_doc_key(r): r for r in semantic_results + keyword_results}
        combined = []

        for doc_key, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_key in all_results:
                result = all_results[doc_key]
                result.similarity_score = score
                combined.append(result)

        # Update ranks
        for i, result in enumerate(combined):
            result.rank = i + 1

        return combined

    def _get_doc_key(self, result: SearchResult) -> str:
        """Generate a unique key for a document result."""
        return result.chunk_id or hashlib.md5(result.content.encode()).hexdigest()

    def _matches_filter(self, doc_idx: int, filter_metadata: Dict[str, Any]) -> bool:
        """Check if document matches metadata filters."""
        doc_metadata = self.metadata_store.get(doc_idx, {})

        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False

            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            else:
                if doc_metadata[key] != value:
                    return False

        return True

    async def _generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts
        )
        return embeddings.tolist()

    async def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            index_path = Path(settings.FAISS_INDEX_PATH)
            index_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(index_path / "index.faiss"))

            # Save document store
            with open(index_path / "documents.pkl", "wb") as f:
                pickle.dump(self.document_store, f)

            # Save metadata store
            with open(index_path / "metadata.json", "w") as f:
                json.dump(self.metadata_store, f, indent=2)

            logger.info("Successfully saved vector index and metadata")

        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            index_path = Path(settings.FAISS_INDEX_PATH)

            # Load FAISS index
            if (index_path / "index.faiss").exists():
                self.index = faiss.read_index(str(index_path / "index.faiss"))

            # Load document store
            if (index_path / "documents.pkl").exists():
                with open(index_path / "documents.pkl", "rb") as f:
                    self.document_store = pickle.load(f)

            # Load metadata store
            if (index_path / "metadata.json").exists():
                with open(index_path / "metadata.json", "r") as f:
                    self.metadata_store = json.load(f)
                    # Convert string keys back to integers
                    self.metadata_store = {int(k): v for k, v in self.metadata_store.items()}

            logger.info(f"Loaded vector index with {self.index.ntotal} documents")

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            self._initialize_index()

    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        total_docs = len(self.document_store)

        if total_docs == 0:
            return {"total_documents": 0}

        # Calculate statistics
        word_counts = [len(content.split()) for content in self.document_store.values()]

        stats = {
            "total_documents": total_docs,
            "total_words": sum(word_counts),
            "avg_words_per_doc": np.mean(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "index_size_mb": self.index.ntotal * self.dimension * 4 / (1024 * 1024)
        }

        # File type distribution
        file_types = {}
        for metadata in self.metadata_store.values():
            filename = metadata.get("filename", "unknown")
            ext = Path(filename).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

        stats["file_type_distribution"] = file_types

        return stats
