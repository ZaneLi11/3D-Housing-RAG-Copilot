import streamlit as st
import os
import json
from core.vector_store import KnowledgeBaseManager
from core.rag_chain import RAGChainManager
from config import DATA_DIR, CHAT_HISTORY_FILE
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Configuration ---
st.set_page_config(page_title="3D Housing RAG Copilot", layout="wide")
st.title("🏠 3D-Printed Housing AI Copilot")
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
    return kb_manager, rag_manager

kb_manager, rag_manager = init_managers()

# --- 4. Sidebar: Management ---
with st.sidebar:
    st.header("Control Panel")
    
    # Reset Button
    if st.button("Reset Chat Session"):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.rerun()
        
    st.markdown("---")
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload PDF report", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process Document"):
            with st.spinner("Processing..."):
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
if prompt := st.chat_input("Ask me about 3D construction technology..."):
    # 1. Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_to_disk(st.session_state.messages) # Save immediately
    
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
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            rag_manager.ask_stream(prompt, history_langchain)
        )
    
    # 4. Finalize storage
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_to_disk(st.session_state.messages) # Save AI response