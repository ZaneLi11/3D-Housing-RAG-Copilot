import json
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import APP_TITLE, CHAT_HISTORY_FILE, DATA_DIR
from core.memory_manager import HierarchicalMemoryManager
from core.rag_chain import RAGChainManager
from core.vector_store import KnowledgeBaseManager

# --- 1. Page Configuration ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("Personal Literature Research Assistant")
st.markdown("---")


# --- 2. Helper Functions for History ---
def save_chat_to_disk(messages):
    """Saves the current session to the chat_history folder."""
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def load_chat_from_disk():
    """Loads the session from the chat_history folder."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# --- 3. Initialize Backend Managers ---
@st.cache_resource
def init_managers():
    kb_manager = KnowledgeBaseManager()
    retriever = kb_manager.get_retriever()
    rag_manager = RAGChainManager(retriever)
    memory_manager = HierarchicalMemoryManager()
    return kb_manager, rag_manager, memory_manager


kb_manager, rag_manager, memory_manager = init_managers()

# --- 4. Sidebar: Management ---
with st.sidebar:
    st.header("Workspace")

    # Reset Button
    if st.button("Reset Session"):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        memory_manager.reset_session_memory()
        st.rerun()

    st.markdown("---")
    st.subheader("Document Library")
    uploaded_file = st.file_uploader("Upload paper/report (PDF)", type="pdf")

    if uploaded_file:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Index Document"):
            with st.spinner("Indexing document..."):
                result = kb_manager.add_document(uploaded_file.name)
                if result["status"] == "success":
                    st.success(result["message"])
                elif result["status"] == "warning":
                    st.warning(result["message"])
                else:
                    st.error(result["message"])

# --- 5. Main Chat Interface ---
# Load messages from disk on first run
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_from_disk()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your papers, methods, findings, or comparisons..."):
    # 1. Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_to_disk(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare History for LLM
    history_langchain = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history_langchain.append(HumanMessage(content=m["content"]))
        else:
            history_langchain.append(AIMessage(content=m["content"]))

    # 3. Stream Assistant Response
    memory_context = memory_manager.get_memory_context(prompt)
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            rag_manager.ask_stream(prompt, history_langchain, memory_context=memory_context)
        )

    # 4. Finalize storage
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_to_disk(st.session_state.messages)
    memory_manager.maybe_update_memories(st.session_state.messages)

