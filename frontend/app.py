import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
from components.chat_interface import ChatInterface
from components.document_viewer import DocumentViewer
from components.challenge_mode import ChallengeMode
from utils.ui_helpers import UIHelpers

# Configuration
BACKEND_URL = "http://localhost:8000"
API_BASE = f"{BACKEND_URL}/api/v1"


class GenAIDocumentAssistant:
    """Main Streamlit application for GenAI Document Assistant."""

    def __init__(self):
        self.ui_helpers = UIHelpers()
        self.chat_interface = ChatInterface(API_BASE)
        self.document_viewer = DocumentViewer(API_BASE)
        self.challenge_mode = ChallengeMode(API_BASE)

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        if 'current_document' not in st.session_state:
            st.session_state.current_document = None

        if 'documents' not in st.session_state:
            st.session_state.documents = []

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "Ask Anything"

        if 'generated_questions' not in st.session_state:
            st.session_state.generated_questions = []

        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = {}

    def run(self):
        """Run the main application."""
        # Page configuration
        st.set_page_config(
            page_title="GenAI Document Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        self._apply_custom_css()

        # Header
        self._render_header()

        # Sidebar
        self._render_sidebar()

        # Main content
        self._render_main_content()

        # Footer
        self._render_footer()

    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }

        .main-header h1 {
            color: white;
            text-align: center;
            margin: 0;
        }

        .document-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }

        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
        }

        .user-message {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }

        .assistant-message {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }

        .question-card {
            background: #fff3e0;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin-bottom: 1rem;
        }

        .success-message {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }

        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }

        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }

        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_header(self):
        """Render application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 GenAI Document Assistant</h1>
            <p style="color: white; text-align: center; margin: 0;">
                Upload documents, ask questions, and test your knowledge with AI-powered assistance
            </p>
        </div>
        """, unsafe_allow_html=True)

    def _render_sidebar(self):
        """Render sidebar with document management and mode selection."""
        with st.sidebar:
            st.header("📁 Document Management")

            # Document upload
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'txt', 'docx', 'md'],
                help="Upload a document to analyze"
            )

            if uploaded_file is not None:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        success = self._upload_document(uploaded_file)
                        if success:
                            st.success("Document processed successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to process document")

            # Document list
            st.subheader("📄 Your Documents")
            self._render_document_list()

            # Mode selection
            st.header("🎯 Interaction Mode")
            mode = st.selectbox(
                "Choose Mode",
                ["Ask Anything", "Challenge Me", "Auto Summary"],
                index=0 if st.session_state.current_mode == "Ask Anything" else
                1 if st.session_state.current_mode == "Challenge Me" else 2
            )

            if mode != st.session_state.current_mode:
                st.session_state.current_mode = mode
                st.rerun()

            # Session info
            st.header("ℹ️ Session Info")
            st.info(f"Session ID: {st.session_state.session_id[:8]}...")

            if st.button("Clear Session"):
                self._clear_session()
                st.rerun()

    def _render_main_content(self):
        """Render main content area based on selected mode."""
        if not st.session_state.current_document:
            st.warning("Please upload and select a document to get started.")
            return

        # Document info
        self._render_document_info()

        # Mode-specific content
        if st.session_state.current_mode == "Ask Anything":
            self._render_ask_anything_mode()
        elif st.session_state.current_mode == "Challenge Me":
            self._render_challenge_mode()
        elif st.session_state.current_mode == "Auto Summary":
            self._render_summary_mode()

    def _render_document_info(self):
        """Render current document information."""
        doc = st.session_state.current_document

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Document", doc.get('original_filename', 'Unknown'))

        with col2:
            st.metric("Pages", doc.get('page_count', 0))

        with col3:
            st.metric("Words", f"{doc.get('word_count', 0):,}")

        with col4:
            st.metric("Chunks", doc.get('total_chunks', 0))

        # Document summary
        if doc.get('summary'):
            with st.expander("📝 Document Summary", expanded=False):
                st.write(doc['summary'])

        # Key topics
        if doc.get('key_topics'):
            st.subheader("🏷️ Key Topics")
            topics = doc['key_topics'][:10]  # Show first 10 topics
            cols = st.columns(min(len(topics), 5))
            for i, topic in enumerate(topics):
                with cols[i % 5]:
                    st.tag(topic)

    def _render_ask_anything_mode(self):
        """Render Ask Anything mode interface."""
        st.header("💬 Ask Anything")
        st.write("Ask any question about your document and get detailed answers with references.")

        # Chat interface
        self.chat_interface.render(
            st.session_state.current_document['id'],
            st.session_state.session_id
        )

    def _render_challenge_mode(self):
        """Render Challenge Me mode interface."""
        st.header("🎯 Challenge Me")
        st.write("Test your understanding with AI-generated questions.")

        # Challenge interface
        self.challenge_mode.render(
            st.session_state.current_document['id'],
            st.session_state.session_id
        )

    def _render_summary_mode(self):
        """Render Auto Summary mode interface."""
        st.header("📋 Auto Summary")
        st.write("Get an automatically generated summary of your document.")

        doc = st.session_state.current_document

        if doc.get('summary'):
            st.markdown(f"""
            <div class="document-card">
                <h4>Document Summary</h4>
                <p>{doc['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Summary not available for this document.")

        # Additional insights
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Document Statistics")
            stats = {
                "Total Words": f"{doc.get('word_count', 0):,}",
                "Total Pages": doc.get('page_count', 0),
                "Processing Time": f"{doc.get('processing_time', 0):.2f}s",
                "File Size": f"{doc.get('file_size', 0) / 1024:.1f} KB"
            }

            for key, value in stats.items():
                st.metric(key, value)

        with col2:
            st.subheader("🏷️ Key Topics")
            topics = doc.get('key_topics', [])
            if topics:
                for topic in topics[:10]:
                    st.write(f"• {topic}")
            else:
                st.info("No key topics extracted.")

    def _render_document_list(self):
        """Render list of uploaded documents."""
        documents = self._get_documents()

        if not documents:
            st.info("No documents uploaded yet.")
            return

        for doc in documents:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"📄 {doc['original_filename']}")
                    st.caption(f"Status: {doc['processing_status']} | {doc['word_count']} words")

                with col2:
                    if st.button("Select", key=f"select_{doc['id']}"):
                        st.session_state.current_document = doc
                        st.rerun()

                # Highlight current document
                if (st.session_state.current_document and
                        st.session_state.current_document['id'] == doc['id']):
                    st.success("✓ Currently selected")

    def _render_footer(self):
        """Render application footer."""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("🤖 Powered by GenAI")

        with col2:
            st.caption("📊 Built with Streamlit")

        with col3:
            st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    def _upload_document(self, uploaded_file) -> bool:
        """Upload document to backend."""
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_BASE}/documents/upload", files=files)

            if response.status_code == 200:
                # Refresh document list
                st.session_state.documents = self._get_documents()
                return True
            else:
                logger.error(f"Upload failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return False

    def _get_documents(self) -> List[Dict[str, Any]]:
        """Get list of documents from backend."""
        try:
            response = requests.get(f"{API_BASE}/documents")
            if response.status_code == 200:
                data = response.json()
                return data.get('documents', [])
            else:
                logger.error(f"Failed to get documents: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []

    def _clear_session(self):
        """Clear session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_session_state()


def main():
    """Main application entry point."""
    app = GenAIDocumentAssistant()
    app.run()


if __name__ == "__main__":
    main()
