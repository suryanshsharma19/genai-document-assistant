import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class ChallengeMode:
    """Challenge mode component for testing user knowledge."""

    def __init__(self, api_base: str):
        self.api_base = api_base

    def render(self, document_id: int, session_id: str):
        """Render the challenge mode interface."""
        # Initialize session state for challenge mode
        if f'challenge_state_{document_id}' not in st.session_state:
            st.session_state[f'challenge_state_{document_id}'] = {
                'questions': [],
                'current_question': 0,
                'user_answers': {},
                'evaluations': {},
                'quiz_started': False,
                'quiz_completed': False,
                'start_time': None,
                'end_time': None
            }

        challenge_state = st.session_state[f'challenge_state_{document_id}']

        # Render based on current state
        if not challenge_state['quiz_started']:
            self._render_quiz_setup(document_id, session_id, challenge_state)
        elif challenge_state['quiz_completed']:
            self._render_quiz_results(document_id, challenge_state)
        else:
            self._render_quiz_interface(document_id, session_id, challenge_state)

    def _render_quiz_setup(self, document_id: int, session_id: str, challenge_state: Dict[str, Any]):
        """Render quiz setup interface."""
        st.subheader("🎯 Challenge Setup")
        st.write("Configure your knowledge test and generate questions based on your document.")

        # Quiz configuration
        col1, col2 = st.columns(2)

        with col1:
            num_questions = st.slider(
                "Number of Questions",
                min_value=1,
                max_value=10,
                value=5,
                help="How many questions to generate"
            )

        with col2:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["easy", "medium", "hard"],
                index=1,
                help="Question difficulty level"
            )

        # Question types
        st.subheader("📝 Question Types")

        col1, col2, col3 = st.columns(3)

        with col1:
            include_factual = st.checkbox("📊 Factual", value=True, help="Questions about facts in the document")

        with col2:
            include_inferential = st.checkbox("🧠 Inferential", value=True, help="Questions requiring inference")

        with col3:
            include_analytical = st.checkbox("🔍 Analytical", value=True, help="Questions requiring analysis")

        # Generate questions button
        if st.button("🚀 Generate Questions & Start Quiz", use_container_width=True):
            if not any([include_factual, include_inferential, include_analytical]):
                st.error("Please select at least one question type.")
                return

            question_types = []
            if include_factual:
                question_types.append("factual")
            if include_inferential:
                question_types.append("inferential")
            if include_analytical:
                question_types.append("analytical")

            with st.spinner("🤖 Generating questions..."):
                questions = self._generate_questions(
                    document_id, num_questions, difficulty, question_types
                )

                if questions:
                    challenge_state['questions'] = questions
                    challenge_state['quiz_started'] = True
                    challenge_state['start_time'] = time.time()
                    st.success(f"✅ Generated {len(questions)} questions!")
                    st.rerun()
                else:
                    st.error("❌ Failed to generate questions. Please try again.")

    def _render_quiz_interface(self, document_id: int, session_id: str, challenge_state: Dict[str, Any]):
        """Render the main quiz interface."""
        questions = challenge_state['questions']
        current_idx = challenge_state['current_question']

        if current_idx >= len(questions):
            # Quiz completed
            challenge_state['quiz_completed'] = True
            challenge_state['end_time'] = time.time()
            st.rerun()
            return

        current_question = questions[current_idx]

        # Progress indicator
        progress = (current_idx + 1) / len(questions)
        st.progress(progress)
        st.write(f"Question {current_idx + 1} of {len(questions)}")

        # Question display
        self._render_question(current_question, current_idx, document_id, session_id, challenge_state)

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_idx > 0:
                if st.button("⬅️ Previous"):
                    challenge_state['current_question'] = current_idx - 1
                    st.rerun()

        with col3:
            if current_idx < len(questions) - 1:
                if st.button("Next ➡️"):
                    challenge_state['current_question'] = current_idx + 1
                    st.rerun()
            else:
                if st.button("🏁 Finish Quiz"):
                    challenge_state['quiz_completed'] = True
                    challenge_state['end_time'] = time.time()
                    st.rerun()

        # Quiz controls
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Restart Quiz"):
                self._restart_quiz(document_id, challenge_state)
                st.rerun()

        with col2:
            if st.button("❌ End Quiz"):
                challenge_state['quiz_completed'] = True
                challenge_state['end_time'] = time.time()
                st.rerun()

    def _render_question(
            self,
            question: Dict[str, Any],
            question_idx: int,
            document_id: int,
            session_id: str,
            challenge_state: Dict[str, Any]
    ):
        """Render a single question."""
        # Question header
        st.markdown(f"""
        <div class="question-card">
            <h4>❓ Question {question_idx + 1}</h4>
            <p><strong>Type:</strong> {question.get('question_type', 'Unknown').title()}</p>
            <p><strong>Difficulty:</strong> {question.get('difficulty_level', 'Unknown').title()}</p>
        </div>
        """, unsafe_allow_html=True)

        # Question text
        st.write("### " + question.get('question_text', ''))

        # Answer input
        answer_key = f"answer_{question_idx}"
        current_answer = challenge_state['user_answers'].get(question_idx, "")

        user_answer = st.text_area(
            "Your Answer:",
            value=current_answer,
            height=150,
            key=answer_key,
            placeholder="Type your answer here..."
        )

        # Save answer
        if user_answer != current_answer:
            challenge_state['user_answers'][question_idx] = user_answer

        # Submit answer button
        if st.button("✅ Submit Answer", key=f"submit_{question_idx}"):
            if user_answer.strip():
                self._evaluate_answer(question, user_answer, question_idx, session_id, challenge_state)
            else:
                st.warning("Please provide an answer before submitting.")

        # Show evaluation if available
        if question_idx in challenge_state['evaluations']:
            self._display_evaluation(challenge_state['evaluations'][question_idx])

    def _render_quiz_results(self, document_id: int, challenge_state: Dict[str, Any]):
        """Render quiz results and statistics."""
        st.subheader("🏆 Quiz Results")

        questions = challenge_state['questions']
        evaluations = challenge_state['evaluations']

        # Calculate statistics
        total_questions = len(questions)
        answered_questions = len([q for q in challenge_state['user_answers'].values() if q.strip()])
        evaluated_questions = len(evaluations)

        if evaluated_questions > 0:
            correct_answers = sum(1 for eval in evaluations.values() if eval.get('is_correct', False))
            average_score = sum(eval.get('evaluation_score', 0) for eval in evaluations.values()) / evaluated_questions
            total_time = challenge_state['end_time'] - challenge_state['start_time'] if challenge_state['end_time'] and \
                                                                                        challenge_state[
                                                                                            'start_time'] else 0
        else:
            correct_answers = 0
            average_score = 0
            total_time = 0

        # Results summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Questions", total_questions)

        with col2:
            st.metric("Correct Answers", correct_answers)

        with col3:
            st.metric("Average Score", f"{average_score:.1%}")

        with col4:
            st.metric("Total Time", f"{total_time:.0f}s")

        # Performance chart
        if evaluated_questions > 0:
            st.subheader("📊 Performance Breakdown")

            # Create performance data
            performance_data = []
            for i, question in enumerate(questions):
                if i in evaluations:
                    eval_data = evaluations[i]
                    performance_data.append({
                        'Question': f"Q{i + 1}",
                        'Score': eval_data.get('evaluation_score', 0),
                        'Correct': eval_data.get('is_correct', False)
                    })

            if performance_data:
                # Simple bar chart using Streamlit
                scores = [item['Score'] for item in performance_data]
                st.bar_chart(scores)

        # Detailed results
        st.subheader("📝 Detailed Results")

        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('question_text', '')[:50]}...", expanded=False):
                # Question details
                st.write(f"**Type:** {question.get('question_type', 'Unknown').title()}")
                st.write(f"**Difficulty:** {question.get('difficulty_level', 'Unknown').title()}")
                st.write(f"**Question:** {question.get('question_text', '')}")

                # User answer
                user_answer = challenge_state['user_answers'].get(i, "Not answered")
                st.write(f"**Your Answer:** {user_answer}")

                # Correct answer
                st.write(f"**Correct Answer:** {question.get('correct_answer', 'Not available')}")

                # Evaluation
                if i in evaluations:
                    eval_data = evaluations[i]
                    self._display_evaluation(eval_data)
                else:
                    st.info("Answer not evaluated")

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Retake Quiz"):
                self._restart_quiz(document_id, challenge_state)
                st.rerun()

        with col2:
            if st.button("📊 New Quiz"):
                challenge_state['quiz_started'] = False
                challenge_state['quiz_completed'] = False
                st.rerun()

        with col3:
            if st.button("📥 Export Results"):
                self._export_results(challenge_state)

    def _display_evaluation(self, evaluation: Dict[str, Any]):
        """Display evaluation results."""
        is_correct = evaluation.get('is_correct', False)
        score = evaluation.get('evaluation_score', 0)
        feedback = evaluation.get('feedback', '')

        # Score display
        if is_correct:
            st.success(f"✅ Correct! Score: {score:.1%}")
        else:
            st.error(f"❌ Incorrect. Score: {score:.1%}")

        # Feedback
        if feedback:
            st.info(f"💡 **Feedback:** {feedback}")

        # Partial credit
        partial_credit = evaluation.get('partial_credit', 0)
        if partial_credit > 0 and not is_correct:
            st.warning(f"⚡ Partial Credit: {partial_credit:.1%}")

    def _generate_questions(
            self,
            document_id: int,
            num_questions: int,
            difficulty: str,
            question_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate questions using the API."""
        try:
            payload = {
                "document_id": document_id,
                "num_questions": num_questions,
                "difficulty_level": difficulty,
                "question_types": question_types
            }

            response = requests.post(
                f"{self.api_base}/questions/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('questions', [])
            else:
                logger.error(f"Question generation failed: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _evaluate_answer(
            self,
            question: Dict[str, Any],
            user_answer: str,
            question_idx: int,
            session_id: str,
            challenge_state: Dict[str, Any]
    ):
        """Evaluate user answer using the API."""
        try:
            # Note: This assumes the question has an ID from the database
            # In a real implementation, you'd need to store question IDs
            question_id = question.get('id', question_idx)  # Fallback to index

            payload = {
                "question_id": question_id,
                "user_answer": user_answer,
                "session_id": session_id
            }

            with st.spinner("🤖 Evaluating your answer..."):
                response = requests.post(
                    f"{self.api_base}/answers/evaluate",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    evaluation = response.json()
                    challenge_state['evaluations'][question_idx] = evaluation
                    st.success("✅ Answer evaluated!")
                    st.rerun()
                else:
                    st.error(f"Evaluation failed: {response.text}")

        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            st.error(f"Evaluation error: {str(e)}")

    def _restart_quiz(self, document_id: int, challenge_state: Dict[str, Any]):
        """Restart the quiz with the same questions."""
        challenge_state['current_question'] = 0
        challenge_state['user_answers'] = {}
        challenge_state['evaluations'] = {}
        challenge_state['quiz_completed'] = False
        challenge_state['start_time'] = time.time()
        challenge_state['end_time'] = None

    def _export_results(self, challenge_state: Dict[str, Any]):
        """Export quiz results as JSON."""
        try:
            export_data = {
                "quiz_timestamp": time.time(),
                "total_questions": len(challenge_state['questions']),
                "questions": challenge_state['questions'],
                "user_answers": challenge_state['user_answers'],
                "evaluations": challenge_state['evaluations'],
                "start_time": challenge_state['start_time'],
                "end_time": challenge_state['end_time'],
                "total_time": challenge_state['end_time'] - challenge_state['start_time'] if challenge_state[
                                                                                                 'end_time'] and
                                                                                             challenge_state[
                                                                                                 'start_time'] else 0
            }

            json_data = json.dumps(export_data, indent=2)

            st.download_button(
                label="📥 Download Quiz Results",
                data=json_data,
                file_name=f"quiz_results_{int(time.time())}.json",
                mime="application/json"
            )

        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            st.error(f"Failed to export results: {str(e)}")
