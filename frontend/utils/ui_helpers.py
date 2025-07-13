import streamlit as st
from typing import Any, List, Dict, Optional, Union
import json
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class UIHelpers:
    """Advanced UI helper utilities for Streamlit components."""

    @staticmethod
    def render_loading_spinner(message: str = "Loading..."):
        """Context manager for showing a loading spinner."""
        return st.spinner(message)

    @staticmethod
    def render_error_message(message: str, details: Optional[str] = None):
        """Render an enhanced error message box."""
        st.error(f"❌ {message}")
        if details:
            with st.expander("Error Details", expanded=False):
                st.code(details)

    @staticmethod
    def render_success_message(message: str, details: Optional[str] = None):
        """Render an enhanced success message box."""
        st.success(f"✅ {message}")
        if details:
            st.info(details)

    @staticmethod
    def render_info_message(message: str, icon: str = "ℹ️"):
        """Render an info message box with custom icon."""
        st.info(f"{icon} {message}")

    @staticmethod
    def render_warning_message(message: str, details: Optional[str] = None):
        """Render a warning message box."""
        st.warning(f"⚠️ {message}")
        if details:
            st.caption(details)

    @staticmethod
    def render_metric(label: str, value: Any, delta: Any = None, help_text: str = None):
        """Render an enhanced metric component."""
        st.metric(label=label, value=value, delta=delta, help=help_text)

    @staticmethod
    def render_tag(tag_text: str, color: str = "#e0e0e0", text_color: str = "#000000"):
        """Render a customizable tag-like label."""
        st.markdown(
            f"<span style='background-color:{color}; color:{text_color}; border-radius:5px; "
            f"padding:2px 8px; margin:2px; display:inline-block; font-size:0.8em;'>{tag_text}</span>",
            unsafe_allow_html=True
        )

    @staticmethod
    def render_tags(tags: List[str], max_tags: int = 10):
        """Render multiple tags in a row."""
        if not tags:
            return

        displayed_tags = tags[:max_tags]
        cols = st.columns(min(len(displayed_tags), 5))

        for i, tag in enumerate(displayed_tags):
            with cols[i % 5]:
                UIHelpers.render_tag(tag)

        if len(tags) > max_tags:
            st.caption(f"... and {len(tags) - max_tags} more")

    @staticmethod
    def render_horizontal_rule():
        """Render a horizontal rule."""
        st.markdown("---")

    @staticmethod
    def render_expander(label: str, expanded: bool = False):
        """Context manager for an expander."""
        return st.expander(label, expanded=expanded)

    @staticmethod
    def render_columns(num_columns: int, gap: str = "small"):
        """Create columns with gap control."""
        return st.columns(num_columns, gap=gap)

    @staticmethod
    def render_button(
            label: str,
            key: str = None,
            help_text: str = None,
            disabled: bool = False,
            use_container_width: bool = False
    ):
        """Render an enhanced button."""
        return st.button(
            label,
            key=key,
            help=help_text,
            disabled=disabled,
            use_container_width=use_container_width
        )

    @staticmethod
    def render_text_area(
            label: str,
            value: str = "",
            height: int = 100,
            key: str = None,
            disabled: bool = False,
            placeholder: str = None
    ):
        """Render an enhanced text area."""
        return st.text_area(
            label,
            value=value,
            height=height,
            key=key,
            disabled=disabled,
            placeholder=placeholder
        )

    @staticmethod
    def render_selectbox(
            label: str,
            options: List[Any],
            index: int = 0,
            key: str = None,
            help_text: str = None,
            format_func: Optional[callable] = None
    ):
        """Render an enhanced selectbox."""
        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
            help=help_text,
            format_func=format_func
        )

    @staticmethod
    def render_multiselect(
            label: str,
            options: List[Any],
            default: List[Any] = None,
            key: str = None,
            help_text: str = None
    ):
        """Render a multiselect widget."""
        return st.multiselect(
            label,
            options,
            default=default or [],
            key=key,
            help=help_text
        )

    @staticmethod
    def render_slider(
            label: str,
            min_value: Union[int, float],
            max_value: Union[int, float],
            value: Union[int, float] = None,
            step: Union[int, float] = None,
            key: str = None,
            help_text: str = None
    ):
        """Render an enhanced slider."""
        return st.slider(
            label,
            min_value,
            max_value,
            value=value,
            step=step,
            key=key,
            help=help_text
        )

    @staticmethod
    def render_checkbox(
            label: str,
            value: bool = False,
            key: str = None,
            help_text: str = None
    ):
        """Render an enhanced checkbox."""
        return st.checkbox(label, value=value, key=key, help=help_text)

    @staticmethod
    def render_radio(
            label: str,
            options: List[Any],
            index: int = 0,
            key: str = None,
            help_text: str = None,
            horizontal: bool = False
    ):
        """Render a radio button group."""
        return st.radio(
            label,
            options,
            index=index,
            key=key,
            help=help_text,
            horizontal=horizontal
        )

    @staticmethod
    def render_form(key: str, clear_on_submit: bool = False):
        """Context manager for a form."""
        return st.form(key=key, clear_on_submit=clear_on_submit)

    @staticmethod
    def render_download_button(
            label: str,
            data: Union[str, bytes],
            file_name: str,
            mime: str = "application/octet-stream",
            key: str = None
    ):
        """Render an enhanced download button."""
        return st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key
        )

    @staticmethod
    def render_file_uploader(
            label: str,
            type: List[str] = None,
            accept_multiple_files: bool = False,
            key: str = None,
            help_text: str = None
    ):
        """Render a file uploader."""
        return st.file_uploader(
            label,
            type=type,
            accept_multiple_files=accept_multiple_files,
            key=key,
            help=help_text
        )

    @staticmethod
    def render_progress_bar(value: float, text: str = None):
        """Render a progress bar."""
        progress_bar = st.progress(value)
        if text:
            st.caption(text)
        return progress_bar

    @staticmethod
    def render_json_viewer(data: Dict[str, Any], expanded: bool = False):
        """Render JSON data in an expandable viewer."""
        with st.expander("🔍 JSON Data", expanded=expanded):
            st.json(data)

    @staticmethod
    def render_code_block(code: str, language: str = "python"):
        """Render a code block with syntax highlighting."""
        st.code(code, language=language)

    @staticmethod
    def render_data_table(data: List[Dict[str, Any]], key: str = None):
        """Render a data table."""
        if data:
            st.dataframe(data, key=key)
        else:
            st.info("No data to display")

    @staticmethod
    def render_chart(data: List[Union[int, float]], chart_type: str = "line"):
        """Render a simple chart."""
        if chart_type == "line":
            st.line_chart(data)
        elif chart_type == "bar":
            st.bar_chart(data)
        elif chart_type == "area":
            st.area_chart(data)
        else:
            st.line_chart(data)

    @staticmethod
    def render_status_indicator(status: str, message: str = ""):
        """Render a status indicator."""
        status_colors = {
            "success": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "info": "🔵",
            "processing": "🟠"
        }

        icon = status_colors.get(status.lower(), "⚪")
        st.write(f"{icon} {message}")

    @staticmethod
    def render_timeline_item(timestamp: str, title: str, description: str = ""):
        """Render a timeline item."""
        st.markdown(f"""
        <div style="border-left: 2px solid #667eea; padding-left: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: bold; color: #667eea;">{title}</div>
            <div style="font-size: 0.8em; color: #666;">{timestamp}</div>
            {f"<div style='margin-top: 0.5rem;'>{description}</div>" if description else ""}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_card(title: str, content: str, footer: str = "", color: str = "#f8f9fa"):
        """Render a card component."""
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{title}</h4>
            <div style="margin-bottom: 0.5rem;">{content}</div>
            {f"<div style='font-size: 0.8em; color: #666;'>{footer}</div>" if footer else ""}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_notification(message: str, type: str = "info", duration: int = 3):
        """Render a temporary notification."""
        notification_types = {
            "info": ("ℹ️", "#e3f2fd"),
            "success": ("✅", "#e8f5e8"),
            "warning": ("⚠️", "#fff3e0"),
            "error": ("❌", "#ffebee")
        }

        icon, bg_color = notification_types.get(type, ("ℹ️", "#e3f2fd"))

        placeholder = st.empty()
        placeholder.markdown(f"""
        <div style="background-color: {bg_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            {icon} {message}
        </div>
        """, unsafe_allow_html=True)

        # Auto-hide after duration (simplified - in real app you'd use JavaScript)
        return placeholder

    @staticmethod
    def render_sidebar_metric(label: str, value: Any, delta: Any = None):
        """Render a metric in the sidebar."""
        with st.sidebar:
            st.metric(label, value, delta)

    @staticmethod
    def render_tabs(tab_names: List[str]):
        """Render tabs and return tab objects."""
        return st.tabs(tab_names)

    @staticmethod
    def format_timestamp(timestamp: Union[str, datetime]) -> str:
        """Format timestamp for display."""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return timestamp
        else:
            dt = timestamp

        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def create_download_link(data: Union[str, bytes], filename: str, text: str = "Download"):
        """Create a download link for data."""
        if isinstance(data, str):
            data = data.encode()

        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
        st.markdown(href, unsafe_allow_html=True)

    @staticmethod
    def render_empty_state(message: str, icon: str = "📭", action_text: str = None, action_callback: callable = None):
        """Render an empty state with optional action."""
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
            <div style="font-size: 1.2rem; margin-bottom: 1rem;">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        if action_text and action_callback:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(action_text, use_container_width=True):
                    action_callback()

    @staticmethod
    def render_loading_skeleton(lines: int = 3):
        """Render a loading skeleton placeholder."""
        for i in range(lines):
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); 
                        height: 1rem; margin: 0.5rem 0; border-radius: 4px; 
                        animation: shimmer 1.5s infinite;">
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def get_session_state_summary() -> Dict[str, Any]:
        """Get a summary of current session state."""
        return {
            "total_keys": len(st.session_state.keys()),
            "keys": list(st.session_state.keys()),
            "memory_usage": sum(len(str(v)) for v in st.session_state.values())
        }

    @staticmethod
    def clear_session_state(exclude_keys: List[str] = None):
        """Clear session state except for specified keys."""
        exclude_keys = exclude_keys or []
        keys_to_delete = [key for key in st.session_state.keys() if key not in exclude_keys]

        for key in keys_to_delete:
            del st.session_state[key]

    @staticmethod
    def debug_session_state():
        """Debug helper to display session state."""
        with st.expander("🐛 Debug: Session State", expanded=False):
            st.json(dict(st.session_state))
