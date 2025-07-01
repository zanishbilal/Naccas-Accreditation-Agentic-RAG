# ✅ utils/chat.py
import uuid
import streamlit as st

def new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chat_sessions[new_id] = []

def save_state():
    pass

def update_chat_name(chat_id, message):
    st.session_state.chat_names[chat_id] = message["content"][:30]

def process_uploaded_file(file):
    if file.type == "application/pdf":
        return f"[PDF] File uploaded: {file.name}"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return f"[DOCX] File uploaded: {file.name}"
    return None