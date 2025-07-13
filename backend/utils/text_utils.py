import re
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Advanced text processing utilities."""

    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?]+')
        self.whitespace_pattern = re.compile(r'\s+')

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved accuracy."""
        # Handle common abbreviations
        text = self._handle_abbreviations(text)

        # Split on sentence endings
        sentences = self.sentence_endings.split(text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _handle_abbreviations(self, text: str) -> str:
        """Handle common abbreviations to prevent incorrect sentence splitting."""
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.',
            'etc.', 'vs.', 'e.g.', 'i.e.', 'cf.', 'al.', 'Co.', 'Jr.', 'Sr.'
        ]

        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '<!DOT!>'))

        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)

        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using simple TF-IDF approach."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }

        filtered_words = [word for word in words if word not in stop_words]

        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores."""
        sentences = self.split_into_sentences(text)
        words = text.split()

        if not sentences or not words:
            return {"flesch_score": 0, "avg_sentence_length": 0, "avg_word_length": 0}

        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word) for word in words)

        # Average sentence length
        avg_sentence_length = num_words / num_sentences

        # Average word length
        avg_word_length = sum(len(word) for word in words) / num_words

        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (num_syllables / num_words))

        return {
            "flesch_score": max(0, min(100, flesch_score)),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "num_sentences": num_sentences,
            "num_words": num_words,
            "num_syllables": num_syllables
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified approach)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using simple pattern matching."""
        entities = {
            "PERSON": [],
            "ORGANIZATION": [],
            "LOCATION": [],
            "DATE": [],
            "MONEY": [],
            "EMAIL": [],
            "URL": []
        }

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["EMAIL"] = re.findall(email_pattern, text)

        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities["URL"] = re.findall(url_pattern, text)

        # Money pattern
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR)'
        entities["MONEY"] = re.findall(money_pattern, text, re.IGNORECASE)

        # Date pattern (simplified)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        entities["DATE"] = re.findall(date_pattern, text)

        # Capitalized words (potential names/organizations)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        capitalized_words = re.findall(capitalized_pattern, text)

        # Simple heuristics for person vs organization
        for word in capitalized_words:
            if any(title in word.lower() for title in ['inc', 'corp', 'ltd', 'llc', 'company']):
                entities["ORGANIZATION"].append(word)
            elif len(word.split()) <= 3:  # Assume short capitalized phrases are names
                entities["PERSON"].append(word)
            else:
                entities["LOCATION"].append(word)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities


class SemanticChunker:
    """Advanced semantic chunking using embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_chunk_size: int = 1000):
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = 0.3
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.model = None

    async def chunk_text(self, text: str) -> List[str]:
        """Perform semantic chunking of text."""
        if not self.model:
            # Fallback to simple chunking
            return self._simple_chunk(text)

        try:
            # Split into sentences
            text_processor = TextProcessor()
            sentences = text_processor.split_into_sentences(text)

            if len(sentences) <= 1:
                return [text]

            # Generate embeddings for sentences
            embeddings = await self._generate_embeddings_async(sentences)

            # Find semantic boundaries
            boundaries = self._find_semantic_boundaries(embeddings)

            # Create chunks based on boundaries
            chunks = self._create_chunks_from_boundaries(sentences, boundaries)

            # Ensure chunks don't exceed max size
            final_chunks = self._split_oversized_chunks(chunks)

            return final_chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}")
            return self._simple_chunk(text)

    async def _generate_embeddings_async(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences asynchronously."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.model.encode, sentences)
        return embeddings

    def _find_semantic_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Find semantic boundaries based on embedding similarities."""
        boundaries = [0]  # Always start with first sentence

        for i in range(1, len(embeddings)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

            # If similarity is below threshold, mark as boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i)

        boundaries.append(len(embeddings))  # Always end with last sentence
        return boundaries

    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """Create text chunks based on semantic boundaries."""
        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    def _split_oversized_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks that exceed maximum size."""
        final_chunks = []

        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split oversized chunk
                sub_chunks = self._simple_chunk(chunk)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """Simple chunking fallback method."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_chunk_statistics(self, chunks: List[str]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}

        chunk_lengths = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_words_per_chunk": np.mean(word_counts),
            "total_words": sum(word_counts),
            "total_characters": sum(chunk_lengths)
        }


class DocumentAnalyzer:
    """Advanced document analysis utilities."""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.semantic_chunker = SemanticChunker()

    async def analyze_document(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        analysis = {}

        # Basic text statistics
        analysis["basic_stats"] = self._get_basic_statistics(text)

        # Readability analysis
        analysis["readability"] = self.text_processor.calculate_readability_score(text)

        # Keyword extraction
        analysis["keywords"] = self.text_processor.extract_keywords(text, 20)

        # Named entity extraction
        analysis["entities"] = self.text_processor.extract_named_entities(text)

        # Semantic chunking analysis
        chunks = await self.semantic_chunker.chunk_text(text)
        analysis["chunking_stats"] = self.semantic_chunker.get_chunk_statistics(chunks)

        # Language detection (simplified)
        analysis["language_info"] = self._detect_language_features(text)

        return analysis

    def _get_basic_statistics(self, text: str) -> Dict[str, Any]:
        """Get basic text statistics."""
        sentences = self.text_processor.split_into_sentences(text)
        words = text.split()
        paragraphs = text.split('\n\n')

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": sum(len(word) for word in words) / len(words) if words else 0
        }

    def _detect_language_features(self, text: str) -> Dict[str, Any]:
        """Detect language features (simplified approach)."""
        # Count different types of characters
        uppercase_count = sum(1 for c in text if c.isupper())
        lowercase_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_count = sum(1 for c in text if c in '.,!?;:"()[]{}')

        total_chars = len(text)

        return {
            "uppercase_ratio": uppercase_count / total_chars if total_chars else 0,
            "lowercase_ratio": lowercase_count / total_chars if total_chars else 0,
            "digit_ratio": digit_count / total_chars if total_chars else 0,
            "punctuation_ratio": punctuation_count / total_chars if total_chars else 0,
            "estimated_language": "english"  # Simplified - always return English
        }

    def extract_document_structure(self, text: str) -> Dict[str, Any]:
        """Extract document structure information."""
        lines = text.split('\n')

        # Find potential headers (lines that are short and followed by longer content)
        headers = []
        for i, line in enumerate(lines):
            line = line.strip()
            if (len(line) < 100 and
                    line and
                    not line.endswith('.') and
                    i < len(lines) - 1 and
                    len(lines[i + 1].strip()) > 50):
                headers.append({
                    "text": line,
                    "line_number": i + 1,
                    "level": self._estimate_header_level(line)
                })

        # Find lists
        lists = []
        current_list = []
        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(r'^[\d\w][\.\)]\s+', line) or line.startswith('• ') or line.startswith('- '):
                current_list.append({"text": line, "line_number": i + 1})
            else:
                if current_list:
                    lists.append(current_list)
                    current_list = []

        if current_list:
            lists.append(current_list)

        return {
            "headers": headers,
            "lists": lists,
            "total_lines": len(lines),
            "empty_lines": sum(1 for line in lines if not line.strip())
        }

    def _estimate_header_level(self, text: str) -> int:
        """Estimate header level based on text characteristics."""
        if text.isupper():
            return 1
        elif text.istitle():
            return 2
        elif any(char.isupper() for char in text):
            return 3
        else:
            return 4
