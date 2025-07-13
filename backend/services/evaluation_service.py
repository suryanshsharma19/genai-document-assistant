import asyncio
import logging
from typing import Dict, Any, List, Optional
import re
from datetime import datetime
import json

from ..core.config import settings
from ..models.schemas import GeneratedQuestion, AnswerEvaluationResponse
from ..services.llm_service import OpenAIProvider, AnthropicProvider
from ..core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class AnswerEvaluationService:
    """Service for evaluating user answers against correct answers."""

    def __init__(self):
        self.llm_provider = self._get_llm_provider()
        self.evaluation_criteria = self._load_evaluation_criteria()

    def _get_llm_provider(self):
        """Get the configured LLM provider."""
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            return OpenAIProvider()
        elif settings.DEFAULT_LLM_PROVIDER == "anthropic":
            return AnthropicProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.DEFAULT_LLM_PROVIDER}")

    def _load_evaluation_criteria(self) -> Dict[str, Any]:
        """Load evaluation criteria for different question types."""
        return {
            "factual": {
                "weight_accuracy": 0.7,
                "weight_completeness": 0.2,
                "weight_clarity": 0.1,
                "keywords_importance": 0.8
            },
            "inferential": {
                "weight_accuracy": 0.5,
                "weight_reasoning": 0.3,
                "weight_completeness": 0.2,
                "keywords_importance": 0.6
            },
            "analytical": {
                "weight_accuracy": 0.4,
                "weight_reasoning": 0.4,
                "weight_depth": 0.2,
                "keywords_importance": 0.5
            }
        }

    async def evaluate_answer(
            self,
            question: GeneratedQuestion,
            user_answer: str,
            document_content: str = ""
    ) -> AnswerEvaluationResponse:
        """Evaluate user answer against the correct answer."""
        try:
            start_time = datetime.utcnow()

            # Preprocess answers
            user_answer_clean = self._preprocess_answer(user_answer)
            correct_answer_clean = self._preprocess_answer(question.correct_answer)

            # Perform multiple evaluation methods
            similarity_score = await self._calculate_semantic_similarity(user_answer_clean, correct_answer_clean)
            keyword_score = self._calculate_keyword_overlap(user_answer_clean, correct_answer_clean)
            llm_evaluation = await self._evaluate_with_llm(question, user_answer_clean, document_content)

            # Combine scores based on question type
            final_score = self._combine_scores(
                similarity_score, keyword_score, llm_evaluation, question.question_type.value
            )

            # Determine if answer is correct
            is_correct = final_score >= 0.7
            partial_credit = min(final_score, 1.0)

            # Generate feedback
            feedback = await self._generate_feedback(
                question, user_answer_clean, correct_answer_clean, final_score, llm_evaluation
            )

            evaluation_time = (datetime.utcnow() - start_time).total_seconds()

            return AnswerEvaluationResponse(
                is_correct=is_correct,
                partial_credit=partial_credit,
                evaluation_score=final_score,
                feedback=feedback,
                correct_answer=question.correct_answer,
                explanation=question.answer_explanation,
                evaluation_time=evaluation_time
            )

        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise LLMServiceError(f"Failed to evaluate answer: {str(e)}")

    def _preprocess_answer(self, answer: str) -> str:
        """Preprocess answer text for evaluation."""
        # Convert to lowercase
        answer = answer.lower().strip()

        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)

        # Remove common filler words for comparison
        filler_words = ['um', 'uh', 'like', 'you know', 'i think', 'i believe', 'maybe', 'perhaps']
        for filler in filler_words:
            answer = answer.replace(filler, '')

        # Clean up punctuation
        answer = re.sub(r'[^\w\s]', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()

        return answer

    async def _calculate_semantic_similarity(self, user_answer: str, correct_answer: str) -> float:
        """Calculate semantic similarity between answers using embeddings."""
        try:
            # Use the same embedding model as the main service
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.EMBEDDING_MODEL)

            # Generate embeddings
            embeddings = model.encode([user_answer, correct_answer])

            # Calculate cosine similarity - FIXED: Compare the two embeddings correctly
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            # Extract individual embeddings and calculate similarity
            user_embedding = embeddings[0].reshape(1, -1)
            correct_embedding = embeddings[1].reshape(1, -1)

            similarity = cosine_similarity(user_embedding, correct_embedding)[0][0]
            return float(similarity)

        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_keyword_overlap(self, user_answer: str, correct_answer: str) -> float:
        """Calculate keyword overlap between answers."""
        try:
            # Extract keywords (words longer than 3 characters)
            user_keywords = set(word for word in user_answer.split() if len(word) > 3)
            correct_keywords = set(word for word in correct_answer.split() if len(word) > 3)

            if not correct_keywords:
                return 0.0

            # Calculate Jaccard similarity
            intersection = len(user_keywords.intersection(correct_keywords))
            union = len(user_keywords.union(correct_keywords))

            if union == 0:
                return 0.0

            return intersection / union

        except Exception as e:
            logger.warning(f"Keyword overlap calculation failed: {str(e)}")
            return 0.0

    async def _evaluate_with_llm(
            self,
            question: GeneratedQuestion,
            user_answer: str,
            document_content: str
    ) -> Dict[str, Any]:
        """Evaluate answer using LLM for nuanced assessment."""
        try:
            prompt = f"""Evaluate the following user answer against the correct answer for the given question.

Question: {question.question_text}
Question Type: {question.question_type.value}
Difficulty: {question.difficulty_level.value}

Correct Answer: {question.correct_answer}
User Answer: {user_answer}

Document Context (if relevant):
{document_content[:1000] if document_content else "No additional context provided"}

Please evaluate the user's answer on the following criteria:
1. Accuracy: How factually correct is the answer?
2. Completeness: Does it address all parts of the question?
3. Reasoning: Is the logic sound (for inferential/analytical questions)?
4. Clarity: Is the answer well-expressed?

Provide your evaluation as a JSON object with the following structure:
{{
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "reasoning_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "suggestions": ["list of improvement suggestions"]
}}"""

            # Generate evaluation using LLM
            response_generator = self.llm_provider.generate_response(
                prompt, [], stream=False, temperature=0.1
            )

            response_text = ""
            async for chunk in response_generator:
                response_text += chunk

            # Parse JSON response
            try:
                evaluation = json.loads(response_text)
                return evaluation
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM evaluation JSON")
                return self._create_fallback_evaluation(user_answer, question.correct_answer)

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {str(e)}")
            return self._create_fallback_evaluation(user_answer, question.correct_answer)

    def _create_fallback_evaluation(self, user_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Create fallback evaluation when LLM evaluation fails."""
        # Simple heuristic evaluation
        user_words = set(user_answer.lower().split())
        correct_words = set(correct_answer.lower().split())

        overlap = len(user_words.intersection(correct_words))
        total = len(correct_words)

        score = overlap / total if total > 0 else 0.0

        return {
            "accuracy_score": score,
            "completeness_score": score,
            "reasoning_score": score * 0.8,
            "clarity_score": 0.7 if len(user_answer.split()) > 3 else 0.3,
            "overall_score": score,
            "strengths": ["Answer provided"] if user_answer.strip() else [],
            "weaknesses": ["Could be more detailed"] if score < 0.7 else [],
            "suggestions": ["Try to include more specific details from the document"]
        }

    def _combine_scores(
            self,
            similarity_score: float,
            keyword_score: float,
            llm_evaluation: Dict[str, Any],
            question_type: str
    ) -> float:
        """Combine different evaluation scores based on question type."""
        criteria = self.evaluation_criteria.get(question_type, self.evaluation_criteria["factual"])

        # Get LLM scores
        llm_accuracy = llm_evaluation.get("accuracy_score", 0.0)
        llm_completeness = llm_evaluation.get("completeness_score", 0.0)
        llm_reasoning = llm_evaluation.get("reasoning_score", 0.0)
        llm_overall = llm_evaluation.get("overall_score", 0.0)

        # Combine scores based on question type
        if question_type == "factual":
            final_score = (
                    criteria["weight_accuracy"] * max(llm_accuracy, keyword_score) +
                    criteria["weight_completeness"] * llm_completeness +
                    criteria["weight_clarity"] * similarity_score
            )
        elif question_type == "inferential":
            final_score = (
                    criteria["weight_accuracy"] * llm_accuracy +
                    criteria["weight_reasoning"] * llm_reasoning +
                    criteria["weight_completeness"] * llm_completeness
            )
        else:  # analytical
            final_score = (
                    criteria["weight_accuracy"] * llm_accuracy +
                    criteria["weight_reasoning"] * llm_reasoning +
                    criteria["weight_depth"] * llm_overall
            )

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))

    async def _generate_feedback(
            self,
            question: GeneratedQuestion,
            user_answer: str,
            correct_answer: str,
            final_score: float,
            llm_evaluation: Dict[str, Any]
    ) -> str:
        """Generate comprehensive feedback for the user."""
        feedback_parts = []

        # Overall assessment
        if final_score >= 0.9:
            feedback_parts.append("Excellent answer! You demonstrated a thorough understanding of the topic.")
        elif final_score >= 0.7:
            feedback_parts.append("Good answer! You covered the main points well.")
        elif final_score >= 0.5:
            feedback_parts.append("Partial answer. You got some key points but missed others.")
        else:
            feedback_parts.append("Your answer needs improvement. Please review the document content.")

        # Specific strengths
        strengths = llm_evaluation.get("strengths", [])
        if strengths:
            feedback_parts.append(f"Strengths: {', '.join(strengths)}")

        # Areas for improvement
        weaknesses = llm_evaluation.get("weaknesses", [])
        if weaknesses:
            feedback_parts.append(f"Areas for improvement: {', '.join(weaknesses)}")

        # Suggestions
        suggestions = llm_evaluation.get("suggestions", [])
        if suggestions:
            feedback_parts.append(f"Suggestions: {', '.join(suggestions)}")

        # Question-specific feedback
        if question.question_type.value == "factual":
            if final_score < 0.7:
                feedback_parts.append(
                    "For factual questions, focus on providing specific information directly from the document.")
        elif question.question_type.value == "inferential":
            if final_score < 0.7:
                feedback_parts.append(
                    "For inferential questions, explain your reasoning and connect different pieces of information.")
        elif question.question_type.value == "analytical":
            if final_score < 0.7:
                feedback_parts.append(
                    "For analytical questions, provide deeper analysis and consider multiple perspectives.")

        return " ".join(feedback_parts)

    async def batch_evaluate_answers(
            self,
            evaluations: List[Dict[str, Any]]
    ) -> List[AnswerEvaluationResponse]:
        """Evaluate multiple answers in batch."""
        results = []

        for eval_data in evaluations:
            try:
                question = eval_data["question"]
                user_answer = eval_data["user_answer"]
                document_content = eval_data.get("document_content", "")

                result = await self.evaluate_answer(question, user_answer, document_content)
                results.append(result)

            except Exception as e:
                logger.error(f"Error in batch evaluation: {str(e)}")
                # Add error result
                results.append(AnswerEvaluationResponse(
                    is_correct=False,
                    partial_credit=0.0,
                    evaluation_score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    correct_answer="",
                    explanation="",
                    evaluation_time=0.0
                ))

        return results

    def get_evaluation_statistics(self, evaluations: List[AnswerEvaluationResponse]) -> Dict[str, Any]:
        """Get statistics from a set of evaluations."""
        if not evaluations:
            return {}

        scores = [eval.evaluation_score for eval in evaluations]
        correct_count = sum(1 for eval in evaluations if eval.is_correct)

        return {
            "total_evaluations": len(evaluations),
            "correct_answers": correct_count,
            "accuracy_rate": correct_count / len(evaluations),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "average_evaluation_time": sum(eval.evaluation_time for eval in evaluations) / len(evaluations)
        }
