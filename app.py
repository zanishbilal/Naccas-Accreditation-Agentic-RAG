# ✅ app.py
import streamlit as st
from core.workflow import graph
from utils.chat import new_chat, save_state, update_chat_name, process_uploaded_file
from langchain_core.messages import BaseMessage
from pyngrok import ngrok
from config import NGROK_AUTH_TOKEN
import os

st.markdown("""
<style>
.sidebar .sidebar-content {
    width: 200px !important;
}
.css-1vq4p4l.e1f1d6gn2 {
    padding-bottom: 90px !important;
}
div[data-testid="stForm"] input[type="text"] {
    width: 100% !important;
    max-width: 500px !important;
}
input[type="text"] {
    font-size: 0.85rem !important;
    height: 2.2rem !important;
    padding: 0.3rem 0.5rem !important;
    width: 100% !important;
}
button[kind="primary"] {
    font-size: 0.85rem !important;
    padding: 0.3rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "chat_names" not in st.session_state:
    st.session_state.chat_names = {}
if "current_chat_id" not in st.session_state:
    new_chat()

st.sidebar.title("Chat Sessions")
st.sidebar.button("New Chat", on_click=new_chat)

for chat_id in list(st.session_state.chat_sessions.keys())[:5]:
    chat_name = st.session_state.chat_names.get(chat_id, f"Chat {chat_id[:8]}")
    if st.sidebar.button(chat_name, key=chat_id):
        st.session_state.current_chat_id = chat_id
        save_state()

st.title("🧾 NACCAS Policy Assistant")
if st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_chat_id] = []
    save_state()

st.subheader("Chat History")
for message in st.session_state.chat_sessions[st.session_state.current_chat_id]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    if message["role"] == "user" and st.session_state.current_chat_id not in st.session_state.chat_names:
        update_chat_name(st.session_state.current_chat_id, message)

with st.form(key="input_form", clear_on_submit=True):
    upload_col, input_col, button_col = st.columns([1, 3, 1])
    with upload_col:
        uploaded_file = st.file_uploader("", type=['pdf', 'docx'], label_visibility="collapsed")
    with input_col:
        user_input = st.text_input("", placeholder="Ask about NACCAS policies...", key="user_input")
    with button_col:
        submit_button = st.form_submit_button("Send")

if submit_button and (user_input or uploaded_file):
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]

    if uploaded_file:
        file_content = process_uploaded_file(uploaded_file)
        if file_content:
            current_chat.append({"role": "user", "content": f"Uploaded file content:\n{file_content}"})
            update_chat_name(st.session_state.current_chat_id, {"role": "user", "content": file_content})

    if user_input:
        current_chat.append({"role": "user", "content": user_input})
        update_chat_name(st.session_state.current_chat_id, {"role": "user", "content": user_input})

    streamed_messages = []
    for chunk in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for node, update in chunk.items():
            if node != "generate_answer":
                continue
            if "messages" in update and update["messages"]:
                last_msg = update["messages"][-1]
                content = last_msg.content if isinstance(last_msg, BaseMessage) else last_msg.get("content")
                if content:
                    streamed_messages.append({"role": "assistant", "content": content})

    for msg in streamed_messages:
        current_chat.append(msg)

    save_state()
    st.rerun()

if os.getenv("USE_NGROK", "false").lower() == "true":
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8501)
    st.sidebar.write("🌍 Public URL:")
    st.sidebar.write(public_url)
