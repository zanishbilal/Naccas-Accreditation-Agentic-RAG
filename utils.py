"""
Utility functions for the NACCAS Policy Assistant.
"""
import uuid
import streamlit as st


def new_chat():
    """Create a new chat session with a unique ID."""
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chat_sessions[new_id] = []


def save_state():
    """Save the current state (placeholder for future implementation)."""
    pass


def update_chat_name(chat_id, message):
    """Update the chat name based on the first message."""
    st.session_state.chat_names[chat_id] = message["content"][:30]


def process_uploaded_file(file):
    """Process uploaded files and return appropriate message."""
    if file.type == "application/pdf":
        return f"[PDF] File uploaded: {file.name}"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return f"[DOCX] File uploaded: {file.name}"
    else:
        return None