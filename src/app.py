import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

from config import envConfig
from huggingface_hub import login
login(envConfig.HUGGINGFACE_TOKEN)

import streamlit as st
import ulid
from langchain_core.runnables.config import RunnableConfig
from events.chat import handle_bot_chat
from agents.graph import build_graph
from agents.states import RuntimeState
from rag.chain import chain
from llms.chat_models import main_chat_model


def setup_page() -> None:
    st.set_page_config(
        page_title="RAG App",
        page_icon="😂",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "graph" not in st.session_state:
        st.session_state.graph = chain
    if "chat_config" not in st.session_state:
        st.session_state.chat_config = RunnableConfig(configurable={"thread_id": ulid.ULID(), "user_id": ulid.ULID()})
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = RuntimeState(messages=[])
        print("Agent initializes successfully")
    if "messages" not in st.session_state:
        st.session_state.messages = []


def setup_chat_interface() -> None:
    st.chat_message("assistant").markdown(
        "Hello! I am your RAG assistant. How can I help you today?"
    )
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_message = st.chat_input(placeholder="Enter your command...")
    if user_message:
        with st.chat_message("user"):
            st.write(user_message)

        st.session_state.messages.append({"role": "user", "content": user_message})

        handle_bot_chat(user_message)
        

def main():
    # Setup page
    setup_page()
    
    # Setup chat interface
    setup_chat_interface()


if __name__ == "__main__":
    main()