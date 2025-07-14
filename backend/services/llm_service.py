import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import openai
import anthropic
import httpx
import json
import google.generativeai as genai

from ..core.config import settings
from ..models.schemas import SearchResult, LLMResponse
from ..core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_response(
            self,
            prompt: str,
            context: List[SearchResult],
            stream: bool = False,
            **kwargs
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def generate_questions(
            self,
            content: str,
            difficulty: str = "medium",
            num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        pass


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_response(
            self,
            prompt: str,
            context: List[SearchResult],
            stream: bool = False,
            **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI GPT models."""
        try:
            # Prepare context
            context_text = self._prepare_context(context)

            # Create system prompt
            system_prompt = """You are an intelligent document analysis assistant. Your task is to answer questions based ONLY on the provided context from the documents. 

Guidelines:
1. Always base your answers on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Include specific references to support your answers
4. Be comprehensive but concise
5. Highlight key insights and connections between different parts of the document
6. If asked about something not in the context, politely explain that you can only answer based on the provided documents"""

            # Create user prompt
            user_prompt = f"""Context from documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based on the context above. Include specific references to support your response."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            temperature = kwargs.get('temperature', settings.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', settings.MAX_TOKENS)

            if stream:
                response = await self.client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                response = await self.client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                yield response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")

    async def generate_questions(
            self,
            content: str,
            difficulty: str = "medium",
            num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate questions based on document content."""
        try:
            prompt = f"""Based on the following document content, generate {num_questions} {difficulty} difficulty questions that test comprehension and reasoning.

Document content:
{content[:3000]}...

For each question, provide:
1. The question text
2. The correct answer
3. The question type (factual, inferential, analytical)
4. A brief explanation of why this answer is correct

Format your response as a JSON array of question objects with the following structure:
[
  {{
    "question_text": "Your question here",
    "correct_answer": "The correct answer",
    "question_type": "factual/inferential/analytical",
    "explanation": "Brief explanation",
    "difficulty_level": "{difficulty}"
  }}
]"""

            response = await self.client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are an expert educator creating assessment questions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            # Parse JSON response
            try:
                questions = json.loads(response.choices[0].message.content)
                return questions
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON response, creating fallback questions")
                return self._create_fallback_questions(content, num_questions, difficulty)

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise LLMServiceError(f"Failed to generate questions: {str(e)}")

    def _prepare_context(self, context: List[SearchResult]) -> str:
        """Prepare context from search results."""
        context_parts = []
        for i, result in enumerate(context, 1):
            metadata = result.metadata
            source_info = f"Source: {metadata.get('filename', 'Unknown')}"
            if 'estimated_page' in metadata:
                source_info += f", Page: {metadata['estimated_page']}"

            context_parts.append(f"[Context {i}] {source_info}\n{result.content}\n")

        return "\n".join(context_parts)

    def _create_fallback_questions(self, content: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Create fallback questions when JSON parsing fails."""
        questions = []
        sentences = content.split('.')[:10]  # Use first 10 sentences

        for i in range(min(num_questions, len(sentences))):
            questions.append({
                "question_text": f"What does the document say about the topic mentioned in: '{sentences[i][:100]}...'?",
                "correct_answer": sentences[i].strip(),
                "question_type": "factual",
                "explanation": "This question tests basic comprehension of the document content.",
                "difficulty_level": difficulty
            })

        return questions


class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not provided")
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate_response(
            self,
            prompt: str,
            context: List[SearchResult],
            stream: bool = False,
            **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response using Anthropic Claude."""
        try:
            context_text = self._prepare_context(context)

            full_prompt = f"""You are an intelligent document analysis assistant. Answer the following question based ONLY on the provided context.

Context:
{context_text}

Question: {prompt}

Please provide a detailed answer with specific references to the source material. If the context doesn't contain enough information to answer the question, please state this clearly."""

            # Use the modern Messages API
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                stream=stream
            )

            if stream:
                async for chunk in response:
                    if chunk.type == "content_block_delta":
                        yield chunk.delta.text
            else:
                yield response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")

    async def generate_questions(
            self,
            content: str,
            difficulty: str = "medium",
            num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate questions based on document content."""
        try:
            prompt = f"""Based on the following document content, generate {num_questions} {difficulty} difficulty questions that test comprehension and reasoning.

Document content:
{content[:3000]}...

For each question, provide:
1. The question text
2. The correct answer
3. The question type (factual, inferential, analytical)
4. A brief explanation of why this answer is correct

Format your response as a JSON array of question objects."""

            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse JSON response
            try:
                questions = json.loads(response.content[0].text)
                return questions
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, creating fallback questions")
                return self._create_fallback_questions(content, num_questions, difficulty)

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise LLMServiceError(f"Failed to generate questions: {str(e)}")

    def _prepare_context(self, context: List[SearchResult]) -> str:
        """Prepare context from search results."""
        context_parts = []
        for i, result in enumerate(context, 1):
            metadata = result.metadata
            source_info = f"Source: {metadata.get('filename', 'Unknown')}"
            if 'estimated_page' in metadata:
                source_info += f", Page: {metadata['estimated_page']}"

            context_parts.append(f"[Context {i}] {source_info}\n{result.content}\n")

        return "\n".join(context_parts)

    def _create_fallback_questions(self, content: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Create fallback questions when JSON parsing fails."""
        questions = []
        sentences = content.split('.')[:10]  # Use first 10 sentences

        for i in range(min(num_questions, len(sentences))):
            questions.append({
                "question_text": f"What does the document say about the topic mentioned in: '{sentences[i][:100]}...'?",
                "correct_answer": sentences[i].strip(),
                "question_type": "factual",
                "explanation": "This question tests basic comprehension of the document content.",
                "difficulty_level": difficulty
            })

        return questions


class GeminiProvider(OpenAIProvider):
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not provided")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate_response(
            self,
            prompt: str,
            context: List[SearchResult],
            stream: bool = False,
            **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            # Prepare context
            context_text = "\n".join([f"[Context {i+1}] {r.metadata.get('filename', 'Unknown')}: {r.content}" for i, r in enumerate(context)])
            user_prompt = f"Context from documents:\n{context_text}\n\nQuestion: {prompt}\n\nPlease provide a detailed answer based on the context above. Include specific references to support your response."
            # Gemini API is synchronous, so run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(user_prompt)
            )
            yield response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")

    async def generate_questions(
            self,
            content: str,
            difficulty: str = "medium",
            num_questions: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            prompt = f"""Based on the following document content, generate {num_questions} {difficulty} difficulty questions that test comprehension and reasoning.\n\nDocument content:\n{content[:3000]}...\n\nFor each question, provide:\n1. The question text\n2. The correct answer\n3. The question type (factual, inferential, analytical)\n4. A brief explanation of why this answer is correct\n\nFormat your response as a JSON array of question objects with the following structure:\n[\n  {{\n    \"question_text\": \"Your question here\",\n    \"correct_answer\": \"The correct answer\",\n    \"question_type\": \"factual/inferential/analytical\",\n    \"explanation\": \"Brief explanation\",\n    \"difficulty_level\": \"{difficulty}\"\n  }}\n]"""
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            try:
                questions = json.loads(response.text)
                return questions
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, creating fallback questions")
                return self._create_fallback_questions(content, num_questions, difficulty)
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise LLMServiceError(f"Failed to generate questions: {str(e)}")
