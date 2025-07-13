import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ChatInterface:
    """Chat interface component for Ask Anything mode."""

    def __init__(self, api_base: str):
        self.api_base = api_base

    def render(self, document_id: int, session_id: str):
        """Render the chat interface."""
        # Initialize chat history in session state
        if f'chat_history_{document_id}' not in st.session_state:
            st.session_state[f'chat_history_{document_id}'] = []

        chat_history = st.session_state[f'chat_history_{document_id}']

        # Display chat history
        self._render_chat_history(chat_history)

        # Chat input
        self._render_chat_input(document_id, session_id, chat_history)

        # Chat controls
        self._render_chat_controls(document_id)

    def _render_chat_history(self, chat_history: List[Dict[str, Any]]):
        """Render chat message history."""
        if not chat_history:
            st.info("💬 Start a conversation by asking a question about your document!")
            return

        # Create a container for chat messages
        chat_container = st.container()

        with chat_container:
            for message in chat_history:
                if message['role'] == 'user':
                    self._render_user_message(message)
                else:
                    self._render_assistant_message(message)

    def _render_user_message(self, message: Dict[str, Any]):
        """Render user message."""
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

    def _render_assistant_message(self, message: Dict[str, Any]):
        """Render assistant message."""
        content = message['content']
        context = message.get('context', [])
        response_time = message.get('response_time', 0)

        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

        # Show response metadata
        if context or response_time:
            with st.expander("📊 Response Details", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Response Time", f"{response_time:.2f}s")

                with col2:
                    st.metric("Context Sources", len(context))

                if context:
                    st.subheader("📚 Sources Used")
                    for i, ctx in enumerate(context[:3], 1):  # Show top 3 sources
                        with st.container():
                            st.write(f"**Source {i}** (Similarity: {ctx.get('similarity_score', 0):.2f})")
                            st.write(f"📄 {ctx.get('metadata', {}).get('filename', 'Unknown')}")
                            if 'estimated_page' in ctx.get('metadata', {}):
                                st.write(f"📖 Page {ctx['metadata']['estimated_page']}")

                            # Show snippet
                            content_snippet = ctx.get('content', '')[:200] + "..."
                            st.text_area(
                                f"Content Preview {i}",
                                content_snippet,
                                height=80,
                                disabled=True,
                                key=f"preview_{i}_{hash(content_snippet)}"
                            )

    def _render_chat_input(self, document_id: int, session_id: str, chat_history: List[Dict[str, Any]]):
        """Render chat input area."""
        with st.form(key=f"chat_form_{document_id}", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                user_input = st.text_area(
                    "Ask a question about your document:",
                    placeholder="What is the main topic of this document?",
                    height=100,
                    key=f"chat_input_{document_id}"
                )

            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)

            # Advanced options
            with st.expander("⚙️ Advanced Options", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    context_k = st.slider(
                        "Context Sources",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Number of document sections to use as context"
                    )

                with col2:
                    use_hybrid_search = st.checkbox(
                        "Hybrid Search",
                        value=True,
                        help="Combine semantic and keyword search"
                    )

            if submit_button and user_input.strip():
                self._process_user_input(
                    user_input.strip(),
                    document_id,
                    session_id,
                    chat_history,
                    context_k,
                    use_hybrid_search
                )

    def _render_chat_controls(self, document_id: int):
        """Render chat control buttons."""
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Chat", key=f"clear_chat_{document_id}"):
                st.session_state[f'chat_history_{document_id}'] = []
                st.rerun()

        with col2:
            if st.button("📥 Export Chat", key=f"export_chat_{document_id}"):
                self._export_chat_history(st.session_state[f'chat_history_{document_id}'])

        with col3:
            if st.button("🔄 Refresh", key=f"refresh_chat_{document_id}"):
                st.rerun()

    def _process_user_input(
            self,
            user_input: str,
            document_id: int,
            session_id: str,
            chat_history: List[Dict[str, Any]],
            context_k: int,
            use_hybrid_search: bool
    ):
        """Process user input and get AI response."""
        # Add user message to history
        chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': st.session_state.get('current_time', '')
        })

        # Show loading spinner
        with st.spinner("🤖 Thinking..."):
            try:
                # Make API request
                response = self._get_ai_response(
                    user_input,
                    document_id,
                    session_id,
                    context_k,
                    use_hybrid_search
                )

                if response:
                    # Add assistant response to history
                    chat_history.append({
                        'role': 'assistant',
                        'content': response['response'],
                        'context': response.get('context_used', []),
                        'response_time': response.get('response_time', 0),
                        'timestamp': st.session_state.get('current_time', '')
                    })

                    st.success("✅ Response generated!")
                else:
                    st.error("❌ Failed to get response. Please try again.")

            except Exception as e:
                logger.error(f"Error processing user input: {str(e)}")
                st.error(f"❌ Error: {str(e)}")

        # Update session state and rerun
        st.session_state[f'chat_history_{document_id}'] = chat_history
        st.rerun()

    def _get_ai_response(
            self,
            query: str,
            document_id: int,
            session_id: str,
            context_k: int,
            use_hybrid_search: bool
    ) -> Optional[Dict[str, Any]]:
        """Get AI response from backend."""
        try:
            payload = {
                "document_id": document_id,
                "query": query,
                "mode": "qa",
                "session_id": session_id,
                "context_k": context_k
            }

            response = requests.post(
                f"{self.api_base}/conversations",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return None
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return None

    def _export_chat_history(self, chat_history: List[Dict[str, Any]]):
        """Export chat history as JSON."""
        if not chat_history:
            st.warning("No chat history to export.")
            return

        try:
            # Prepare export data
            export_data = {
                "export_timestamp": st.session_state.get('current_time', ''),
                "total_messages": len(chat_history),
                "chat_history": chat_history
            }

            # Convert to JSON
            json_data = json.dumps(export_data, indent=2)

            # Provide download
            st.download_button(
                label="📥 Download Chat History",
                data=json_data,
                file_name=f"chat_history_{st.session_state.get('session_id', 'unknown')}.json",
                mime="application/json"
            )

        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            st.error(f"Failed to export chat history: {str(e)}")

    def get_conversation_stats(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not chat_history:
            return {}

        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in chat_history if msg['role'] == 'assistant']

        total_response_time = sum(
            msg.get('response_time', 0) for msg in assistant_messages
        )

        avg_response_time = (
            total_response_time / len(assistant_messages)
            if assistant_messages else 0
        )

        return {
            "total_messages": len(chat_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_response_time": avg_response_time,
            "total_response_time": total_response_time
        }
