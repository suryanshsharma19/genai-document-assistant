import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime

from ..core.config import settings
from ..models.schemas import GeneratedQuestion, QuestionType, DifficultyLevel
from ..services.llm_service import OpenAIProvider, AnthropicProvider
from ..core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Advanced question generation service."""

    def __init__(self):
        self.llm_provider = self._get_llm_provider()
        self.question_templates = self._load_question_templates()

    def _get_llm_provider(self):
        """Get the configured LLM provider."""
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            return OpenAIProvider()
        elif settings.DEFAULT_LLM_PROVIDER == "anthropic":
            return AnthropicProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.DEFAULT_LLM_PROVIDER}")

    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different types and difficulties."""
        return {
            "factual_easy": [
                "What is mentioned about {topic} in the document?",
                "According to the document, what is {topic}?",
                "The document states that {topic} is what?",
                "What does the document say about {topic}?"
            ],
            "factual_medium": [
                "How does the document describe the relationship between {topic1} and {topic2}?",
                "What are the key characteristics of {topic} mentioned in the document?",
                "According to the document, what are the main components of {topic}?"
            ],
            "factual_hard": [
                "What specific evidence does the document provide to support the claim about {topic}?",
                "How does the document's description of {topic} compare to {topic2}?"
            ],
            "inferential_easy": [
                "Based on the document, what can you infer about {topic}?",
                "What does the document suggest about {topic}?"
            ],
            "inferential_medium": [
                "What conclusions can be drawn from the document's discussion of {topic}?",
                "Based on the information provided, what might be the implications of {topic}?"
            ],
            "inferential_hard": [
                "How might the concepts discussed in the document apply to {scenario}?",
                "What underlying assumptions about {topic} can be inferred from the document?"
            ],
            "analytical_easy": [
                "Why is {topic} important according to the document?",
                "What is the purpose of {topic} as described in the document?"
            ],
            "analytical_medium": [
                "How does the document's argument about {topic} support its main thesis?",
                "What is the significance of {topic} in the context of the document?"
            ],
            "analytical_hard": [
                "Analyze the document's approach to {topic}. What are its strengths and limitations?",
                "How does the document's treatment of {topic} reflect broader themes or patterns?"
            ]
        }

    async def generate_questions(
            self,
            content: str,
            num_questions: int = 3,
            difficulty_level: Optional[DifficultyLevel] = None,
            question_types: Optional[List[QuestionType]] = None
    ) -> List[GeneratedQuestion]:
        """Generate questions based on document content."""
        try:
            # Set defaults
            if difficulty_level is None:
                difficulty_level = DifficultyLevel.MEDIUM

            if question_types is None:
                question_types = [QuestionType.FACTUAL, QuestionType.INFERENTIAL, QuestionType.ANALYTICAL]

            # Extract key topics from content
            key_topics = await self._extract_key_topics(content)

            # Generate questions using LLM
            llm_questions = await self._generate_with_llm(
                content, num_questions, difficulty_level.value, question_types
            )

            # Generate template-based questions as backup
            template_questions = await self._generate_with_templates(
                content, key_topics, num_questions, difficulty_level, question_types
            )

            # Combine and select best questions
            all_questions = llm_questions + template_questions
            selected_questions = self._select_best_questions(all_questions, num_questions)

            return selected_questions

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise LLMServiceError(f"Failed to generate questions: {str(e)}")

    async def _generate_with_llm(
            self,
            content: str,
            num_questions: int,
            difficulty: str,
            question_types: List[QuestionType]
    ) -> List[GeneratedQuestion]:
        """Generate questions using LLM."""
        try:
            # Prepare content (limit length for API)
            content_sample = content[:4000] if len(content) > 4000 else content

            # Create prompt
            types_str = ", ".join([t.value for t in question_types])
            prompt = f"""Based on the following document content, generate {num_questions} questions with {difficulty} difficulty level.

Question types to include: {types_str}

Document content:
{content_sample}

Requirements:
1. Questions should test different aspects of comprehension and reasoning
2. Include a mix of question types: {types_str}
3. Difficulty level: {difficulty}
4. Each question should have a clear, accurate answer based on the document
5. Provide explanations for why the answers are correct

Format your response as a JSON array with this structure:
[
  {{
    "question_text": "Your question here",
    "correct_answer": "The correct answer based on the document",
    "question_type": "factual/inferential/analytical",
    "difficulty_level": "{difficulty}",
    "explanation": "Brief explanation of why this answer is correct",
    "source_reference": "Brief reference to the relevant part of the document"
  }}
]"""

            # Generate questions using LLM
            questions_data = await self.llm_provider.generate_questions(
                content_sample, difficulty, num_questions
            )

            # Convert to GeneratedQuestion objects
            questions = []
            for q_data in questions_data:
                try:
                    question = GeneratedQuestion(
                        question_text=q_data.get("question_text", ""),
                        question_type=QuestionType(q_data.get("question_type", "factual")),
                        difficulty_level=DifficultyLevel(q_data.get("difficulty_level", difficulty)),
                        correct_answer=q_data.get("correct_answer", ""),
                        answer_explanation=q_data.get("explanation", ""),
                        source_chunks=[]  # Will be populated later
                    )
                    questions.append(question)
                except Exception as e:
                    logger.warning(f"Error parsing question data: {str(e)}")
                    continue

            return questions

        except Exception as e:
            logger.warning(f"LLM question generation failed: {str(e)}")
            return []

    async def _generate_with_templates(
            self,
            content: str,
            key_topics: List[str],
            num_questions: int,
            difficulty_level: DifficultyLevel,
            question_types: List[QuestionType]
    ) -> List[GeneratedQuestion]:
        """Generate questions using templates as backup."""
        questions = []

        try:
            for q_type in question_types:
                for difficulty in [difficulty_level.value]:
                    template_key = f"{q_type.value}_{difficulty}"
                    templates = self.question_templates.get(template_key, [])

                    if templates and key_topics:
                        template = random.choice(templates)
                        topic = random.choice(key_topics)

                        # Simple template filling
                        question_text = template.replace("{topic}", topic)
                        if "{topic1}" in template and len(key_topics) > 1:
                            question_text = question_text.replace("{topic1}", key_topics[0])
                            question_text = question_text.replace("{topic2}", key_topics[1])

                        # Generate simple answer
                        answer = await self._generate_template_answer(content, question_text, topic)

                        question = GeneratedQuestion(
                            question_text=question_text,
                            question_type=q_type,
                            difficulty_level=difficulty_level,
                            correct_answer=answer,
                            answer_explanation=f"This question tests {q_type.value} understanding of {topic}.",
                            source_chunks=[]
                        )
                        questions.append(question)

                        if len(questions) >= num_questions:
                            break

                if len(questions) >= num_questions:
                    break

            return questions[:num_questions]

        except Exception as e:
            logger.warning(f"Template question generation failed: {str(e)}")
            return []

    async def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content."""
        # Simple keyword extraction
        words = content.lower().split()

        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    async def _generate_template_answer(self, content: str, question: str, topic: str) -> str:
        """Generate a simple answer for template-based questions."""
        # Find sentences containing the topic
        sentences = content.split('.')
        relevant_sentences = [s.strip() for s in sentences if topic.lower() in s.lower()]

        if relevant_sentences:
            return relevant_sentences[0][:200] + "..." if len(relevant_sentences[0]) > 200 else relevant_sentences[0]
        else:
            return f"The document discusses {topic} in the context of the main subject matter."

    def _select_best_questions(self, questions: List[GeneratedQuestion], num_questions: int) -> List[GeneratedQuestion]:
        """Select the best questions from generated options."""
        if len(questions) <= num_questions:
            return questions

        # Score questions based on various criteria
        scored_questions = []
        for question in questions:
            score = 0

            # Length score (prefer medium-length questions)
            q_len = len(question.question_text.split())
            if 8 <= q_len <= 20:
                score += 2
            elif 5 <= q_len <= 25:
                score += 1

            # Answer length score
            a_len = len(question.correct_answer.split())
            if 5 <= a_len <= 50:
                score += 1

            # Question type diversity bonus
            if question.question_type == QuestionType.ANALYTICAL:
                score += 2
            elif question.question_type == QuestionType.INFERENTIAL:
                score += 1

            # Has explanation bonus
            if question.answer_explanation:
                score += 1

            scored_questions.append((score, question))

        # Sort by score and return top questions
        scored_questions.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in scored_questions[:num_questions]]

    async def evaluate_question_quality(self, question: GeneratedQuestion, content: str) -> Dict[str, Any]:
        """Evaluate the quality of a generated question."""
        quality_score = 0
        feedback = []

        # Check if question is answerable from content
        if question.correct_answer.lower() in content.lower():
            quality_score += 3
            feedback.append("Question is answerable from the document")
        else:
            feedback.append("Question may not be fully answerable from the document")

        # Check question clarity
        if len(question.question_text.split()) >= 5:
            quality_score += 2
            feedback.append("Question has appropriate length")
        else:
            feedback.append("Question might be too short")

        # Check answer quality
        if len(question.correct_answer.split()) >= 3:
            quality_score += 2
            feedback.append("Answer has sufficient detail")
        else:
            feedback.append("Answer might need more detail")

        # Check for explanation
        if question.answer_explanation:
            quality_score += 1
            feedback.append("Includes explanation")
        else:
            feedback.append("Missing explanation")

        return {
            "quality_score": quality_score,
            "max_score": 8,
            "feedback": feedback,
            "percentage": (quality_score / 8) * 100
        }
