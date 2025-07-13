import asyncio
import json
import hashlib
import time
import logging
from typing import Any, Optional, List, Dict
from abc import ABC, abstractmethod
import pickle

from ..core.config import settings
from ..models.schemas import SearchResult

logger = logging.getLogger(__name__)


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend."""

    def __init__(self):
        self.cache = {}
        self.expiry = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Check expiry
            if key in self.expiry and time.time() > self.expiry[key]:
                await self.delete(key)
                return None
            return self.cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        try:
            self.cache[key] = value
            if ttl:
                self.expiry[key] = time.time() + ttl
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if key in self.cache:
                del self.cache[key]
            if key in self.expiry:
                del self.expiry[key]
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False

    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            self.cache.clear()
            self.expiry.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False


class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend."""

    def __init__(self):
        self.redis_client = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            if settings.REDIS_URL:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(settings.REDIS_URL)
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
        except Exception as e:
            logger.error(f"Error initializing Redis: {str(e)}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.redis_client:
            return None

        try:
            value = await self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in Redis."""
        if not self.redis_client:
            return False

        try:
            serialized_value = pickle.dumps(value)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.redis_client:
            return False

        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {str(e)}")
            return False

    async def clear(self) -> bool:
        """Clear all keys from Redis."""
        if not self.redis_client:
            return False

        try:
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis: {str(e)}")
            return False


class SemanticCache:
    """Semantic cache for search queries using embeddings."""

    def __init__(self, backend: Optional[BaseCacheBackend] = None):
        self.backend = backend or MemoryCacheBackend()
        self.similarity_threshold = settings.SEMANTIC_CACHE_THRESHOLD
        self.query_embeddings = {}
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize embedding model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            self.embedding_model = None

    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return f"semantic_cache:{hashlib.md5(query.encode()).hexdigest()}"

    def _generate_embedding_key(self, query: str) -> str:
        """Generate key for storing query embeddings."""
        return f"query_embedding:{hashlib.md5(query.encode()).hexdigest()}"

    async def get(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached results for semantically similar queries."""
        if not self.embedding_model:
            return None

        try:
            # Generate embedding for current query
            query_embedding = self.embedding_model.encode([query])[0]  # Fixed: get first element

            # Check for semantically similar cached queries
            for cached_query, cached_embedding in self.query_embeddings.items():
                similarity = self._calculate_similarity(query_embedding, cached_embedding)

                if similarity >= self.similarity_threshold:
                    cache_key = self._generate_cache_key(cached_query)
                    cached_results = await self.backend.get(cache_key)

                    if cached_results:
                        logger.info(f"Semantic cache hit for query: {query} (similarity: {similarity:.3f})")
                        return cached_results

            return None

        except Exception as e:
            logger.error(f"Error getting semantic cache: {str(e)}")
            return None

    async def set(self, query: str, results: List[SearchResult], ttl: int = None) -> bool:
        """Cache results for a query."""
        if not self.embedding_model:
            return False

        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])[0]  # Fixed: get first element

            # Store embedding for future similarity checks
            self.query_embeddings[query] = query_embedding

            # Cache the results
            cache_key = self._generate_cache_key(query)
            ttl = ttl or settings.CACHE_TTL

            return await self.backend.set(cache_key, results, ttl)

        except Exception as e:
            logger.error(f"Error setting semantic cache: {str(e)}")
            return False

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            # Fixed: Ensure proper shape and extract scalar value
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    async def clear(self) -> bool:
        """Clear all cached data."""
        self.query_embeddings.clear()
        return await self.backend.clear()


class QueryCache:
    """Simple query cache for exact matches."""

    def __init__(self, backend: Optional[BaseCacheBackend] = None):
        self.backend = backend or MemoryCacheBackend()

    def _generate_cache_key(self, query: str, filters: Dict = None) -> str:
        """Generate cache key for query and filters."""
        key_data = {"query": query, "filters": filters or {}}
        key_string = json.dumps(key_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def get(self, query: str, filters: Dict = None) -> Optional[List[SearchResult]]:
        """Get cached results for exact query match."""
        cache_key = self._generate_cache_key(query, filters)
        return await self.backend.get(cache_key)

    async def set(self, query: str, results: List[SearchResult], filters: Dict = None, ttl: int = None) -> bool:
        """Cache results for a query."""
        cache_key = self._generate_cache_key(query, filters)
        ttl = ttl or settings.CACHE_TTL
        return await self.backend.set(cache_key, results, ttl)

    async def clear(self) -> bool:
        """Clear all cached data."""
        return await self.backend.clear()


class CacheManager:
    """Main cache manager coordinating different cache types."""

    def __init__(self):
        # Initialize backends
        if settings.REDIS_URL:
            self.backend = RedisCacheBackend()
        else:
            self.backend = MemoryCacheBackend()

        # Initialize cache types
        self.semantic_cache = SemanticCache(self.backend)
        self.query_cache = QueryCache(self.backend)

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0
        }

    async def get_search_results(self, query: str, filters: Dict = None) -> Optional[List[SearchResult]]:
        """Get cached search results using multiple cache strategies."""
        try:
            # Try exact match first
            results = await self.query_cache.get(query, filters)
            if results:
                self.stats["hits"] += 1
                return results

            # Try semantic cache if enabled
            if settings.ENABLE_SEMANTIC_CACHE:
                results = await self.semantic_cache.get(query)
                if results:
                    self.stats["hits"] += 1
                    return results

            self.stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Error getting cached search results: {str(e)}")
            self.stats["errors"] += 1
            return None

    async def set_search_results(
            self,
            query: str,
            results: List[SearchResult],
            filters: Dict = None,
            ttl: int = None
    ) -> bool:
        """Cache search results using multiple strategies."""
        try:
            success = True

            # Cache exact match
            if not await self.query_cache.set(query, results, filters, ttl):
                success = False

            # Cache for semantic similarity if enabled
            if settings.ENABLE_SEMANTIC_CACHE:
                if not await self.semantic_cache.set(query, results, ttl):
                    success = False

            if success:
                self.stats["sets"] += 1
            else:
                self.stats["errors"] += 1

            return success

        except Exception as e:
            logger.error(f"Error caching search results: {str(e)}")
            self.stats["errors"] += 1
            return False

    async def clear_all(self) -> bool:
        """Clear all caches."""
        try:
            await self.query_cache.clear()
            await self.semantic_cache.clear()
            self.stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}
            return True
        except Exception as e:
            logger.error(f"Error clearing caches: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


# Global cache manager instance
cache_manager = CacheManager()
