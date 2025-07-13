import streamlit as st
import requests
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentViewer:
    """Document viewer component for displaying document content and metadata."""

    def __init__(self, api_base: str):
        self.api_base = api_base

    def render(self, document_id: int):
        """Render the document viewer."""
        # Get document details
        document = self._get_document_details(document_id)

        if not document:
            st.error("Failed to load document details.")
            return

        # Document header
        self._render_document_header(document)

        # Document tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Content", "📊 Statistics", "🏷️ Metadata", "🔍 Search"])

        with tab1:
            self._render_content_tab(document_id)

        with tab2:
            self._render_statistics_tab(document)

        with tab3:
            self._render_metadata_tab(document)

        with tab4:
            self._render_search_tab(document_id)

    def _render_document_header(self, document: Dict[str, Any]):
        """Render document header with basic info."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Document",
                document.get('original_filename', 'Unknown'),
                help="Original filename"
            )

        with col2:
            st.metric(
                "📊 Status",
                document.get('processing_status', 'Unknown').title(),
                help="Processing status"
            )

        with col3:
            st.metric(
                "📖 Pages",
                document.get('page_count', 0),
                help="Number of pages"
            )

        with col4:
            st.metric(
                "📝 Words",
                f"{document.get('word_count', 0):,}",
                help="Total word count"
            )

        # Processing info
        if document.get('processed_at'):
            st.info(f"✅ Processed on {document['processed_at']}")

        # Document summary
        if document.get('summary'):
            with st.expander("📋 Document Summary", expanded=True):
                st.write(document['summary'])

    def _render_content_tab(self, document_id: int):
        """Render document content tab."""
        st.subheader("📄 Document Content")

        # Get document chunks
        chunks = self._get_document_chunks(document_id)

        if not chunks:
            st.info("No content available or document is still processing.")
            return

        # Pagination controls
        chunks_per_page = 5
        total_pages = (len(chunks) + chunks_per_page - 1) // chunks_per_page

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            current_page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                key=f"content_page_{document_id}"
            )

        # Display chunks for current page
        start_idx = (current_page - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, len(chunks))

        for i in range(start_idx, end_idx):
            chunk = chunks[i]
            self._render_chunk(chunk, i + 1)

    def _render_chunk(self, chunk: Dict[str, Any], chunk_number: int):
        """Render a single document chunk."""
        metadata = chunk.get('metadata', {})

        with st.expander(f"📄 Chunk {chunk_number} - Page {metadata.get('estimated_page', '?')}", expanded=False):
            # Chunk metadata
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Words", chunk.get('word_count', 0))

            with col2:
                st.metric("Characters", chunk.get('char_count', 0))

            with col3:
                if metadata.get('section_title'):
                    st.write(f"**Section:** {metadata['section_title']}")

            # Chunk content
            st.text_area(
                "Content",
                chunk.get('content', ''),
                height=200,
                disabled=True,
                key=f"chunk_content_{chunk_number}"
            )

    def _render_statistics_tab(self, document: Dict[str, Any]):
        """Render document statistics tab."""
        st.subheader("📊 Document Statistics")

        # Basic statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Content Metrics")

            metrics = {
                "Total Words": f"{document.get('word_count', 0):,}",
                "Total Pages": document.get('page_count', 0),
                "Total Chunks": document.get('total_chunks', 0),
                "File Size": f"{document.get('file_size', 0) / 1024:.1f} KB",
                "Processing Time": f"{document.get('processing_time', 0):.2f}s"
            }

            for metric, value in metrics.items():
                st.metric(metric, value)

        with col2:
            st.subheader("🏷️ Content Analysis")

            # Key topics
            key_topics = document.get('key_topics', [])
            if key_topics:
                st.write("**Top Topics:**")
                for i, topic in enumerate(key_topics[:10], 1):
                    st.write(f"{i}. {topic}")
            else:
                st.info("No key topics extracted.")

        # Metadata analysis
        metadata = document.get('metadata', {})
        if metadata:
            st.subheader("🔍 Document Metadata")

            # Display metadata in expandable sections
            if 'pages' in metadata:
                with st.expander("📖 Page Information"):
                    st.json(metadata.get('page_layouts', []))

            if 'tables' in metadata:
                with st.expander("📊 Tables"):
                    st.write(f"Found {len(metadata.get('tables', []))} tables")
                    if metadata['tables']:
                        st.json(metadata['tables'])

            if 'images' in metadata:
                with st.expander("🖼️ Images"):
                    st.write(f"Found {metadata.get('images', 0)} images")

            if 'fonts' in metadata:
                with st.expander("🔤 Fonts"):
                    fonts = metadata.get('fonts', [])
                    if fonts:
                        st.write(", ".join(fonts))

    def _render_metadata_tab(self, document: Dict[str, Any]):
        """Render document metadata tab."""
        st.subheader("🏷️ Document Metadata")

        # Basic metadata
        basic_metadata = {
            "ID": document.get('id'),
            "Filename": document.get('filename'),
            "Original Filename": document.get('original_filename'),
            "File Type": document.get('file_type'),
            "Content Hash": document.get('content_hash'),
            "Created At": document.get('created_at'),
            "Updated At": document.get('updated_at'),
            "Processed At": document.get('processed_at'),
            "Is Active": document.get('is_active'),
            "Is Indexed": document.get('is_indexed')
        }

        # Display basic metadata
        for key, value in basic_metadata.items():
            if value is not None:
                st.write(f"**{key}:** {value}")

        # Advanced metadata
        metadata = document.get('metadata', {})
        if metadata:
            st.subheader("🔬 Advanced Metadata")

            # Use tabs for different metadata categories
            if any(key in metadata for key in ['pages', 'tables', 'images', 'fonts']):
                meta_tab1, meta_tab2, meta_tab3 = st.tabs(["📄 Structure", "📊 Content", "🔧 Technical"])

                with meta_tab1:
                    # Document structure
                    if 'pages' in metadata:
                        st.metric("Total Pages", metadata['pages'])

                    if 'page_layouts' in metadata:
                        st.write("**Page Layouts:**")
                        for layout in metadata['page_layouts'][:5]:  # Show first 5
                            st.write(
                                f"Page {layout.get('page', '?')}: {layout.get('width', 0):.0f}x{layout.get('height', 0):.0f}")

                with meta_tab2:
                    # Content analysis
                    if 'tables' in metadata:
                        st.metric("Tables Found", len(metadata['tables']))

                    if 'images' in metadata:
                        st.metric("Images Found", metadata['images'])

                    if 'text_blocks' in metadata:
                        st.metric("Text Blocks", metadata['text_blocks'])

                with meta_tab3:
                    # Technical details
                    if 'fonts' in metadata:
                        st.write("**Fonts Used:**")
                        fonts = metadata['fonts']
                        if fonts:
                            st.write(", ".join(fonts[:10]))  # Show first 10 fonts

                    if 'processing_time' in metadata:
                        st.metric("Processing Time", f"{metadata['processing_time']:.2f}s")

            # Raw metadata
            with st.expander("🔍 Raw Metadata (JSON)", expanded=False):
                st.json(metadata)

    def _render_search_tab(self, document_id: int):
        """Render document search tab."""
        st.subheader("🔍 Search Document")

        # Search input
        search_query = st.text_input(
            "Search within document:",
            placeholder="Enter search terms...",
            key=f"doc_search_{document_id}"
        )

        col1, col2 = st.columns(2)

        with col1:
            search_k = st.slider(
                "Number of results",
                min_value=1,
                max_value=10,
                value=5,
                key=f"search_k_{document_id}"
            )

        with col2:
            use_hybrid = st.checkbox(
                "Hybrid search",
                value=True,
                key=f"hybrid_search_{document_id}"
            )

        if st.button("🔍 Search", key=f"search_btn_{document_id}"):
            if search_query.strip():
                self._perform_document_search(search_query, document_id, search_k, use_hybrid)
            else:
                st.warning("Please enter a search query.")

    def _perform_document_search(self, query: str, document_id: int, k: int, use_hybrid: bool):
        """Perform search within document."""
        try:
            payload = {
                "query": query,
                "document_ids": [document_id],
                "k": k,
                "use_hybrid_search": use_hybrid
            }

            response = requests.post(
                f"{self.api_base}/search",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                self._display_search_results(results)
            else:
                st.error(f"Search failed: {response.text}")

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            st.error(f"Search error: {str(e)}")

    def _display_search_results(self, results: Dict[str, Any]):
        """Display search results."""
        search_results = results.get('results', [])

        if not search_results:
            st.info("No results found.")
            return

        st.success(f"Found {len(search_results)} results in {results.get('search_time', 0):.2f}s")

        for i, result in enumerate(search_results, 1):
            with st.expander(f"Result {i} (Similarity: {result.get('similarity_score', 0):.3f})", expanded=i == 1):
                # Result metadata
                metadata = result.get('metadata', {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Page:** {metadata.get('estimated_page', '?')}")

                with col2:
                    st.write(f"**Chunk:** {metadata.get('chunk_index', '?')}")

                with col3:
                    st.write(f"**Words:** {metadata.get('word_count', 0)}")

                # Result content
                st.text_area(
                    "Content",
                    result.get('content', ''),
                    height=150,
                    disabled=True,
                    key=f"search_result_{i}"
                )

    def _get_document_details(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document details from API."""
        try:
            response = requests.get(f"{self.api_base}/documents/{document_id}")

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get document details: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error getting document details: {str(e)}")
            return None

    def _get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get document chunks from API."""
        try:
            # Note: This endpoint would need to be implemented in the backend
            response = requests.get(f"{self.api_base}/documents/{document_id}/chunks")

            if response.status_code == 200:
                return response.json().get('chunks', [])
            else:
                logger.warning(f"Failed to get document chunks: {response.text}")
                return []

        except Exception as e:
            logger.warning(f"Error getting document chunks: {str(e)}")
            return []
