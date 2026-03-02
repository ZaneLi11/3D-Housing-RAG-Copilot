# 🏠 3D-Housing RAG Copilot

A professional RAG (Retrieval-Augmented Generation) assistant designed to analyze 3D-printed housing feasibility studies and technical reports. This project uses **Ollama** for local LLM inference and **ChromaDB** for high-performance vector storage.

---

## 🌟 Key Features

* **Intelligent Retrieval**: Uses `RecursiveCharacterTextSplitter` to maintain semantic integrity during document chunking.
* **Smart Deduplication**: Implements **MD5 hashing** to prevent redundant document indexing, ensuring database efficiency.
* **Persistent Memory**: Conversations are automatically saved to the `chat_history/` folder in JSON format, allowing users to resume sessions even after a restart.
* **Streaming UI**: A modern, responsive chat interface built with Streamlit, featuring real-time "typewriter" style responses.
* **Local & Private**: Runs 100% locally via Ollama—no data leaves your machine.

---

## 🛠️ Tech Stack

* **LLM Framework**: LangChain (LCEL)
* **Inference Engine**: Ollama (Llama 3 / DeepSeek)
* **Vector Database**: ChromaDB
* **Frontend**: Streamlit
* **Language**: Python 3.10+

---

## 📂 Project Structure

```text
.
├── app.py                # Main Streamlit UI & Session Management
├── config.py             # Global configurations, paths, and LLM settings
├── core/
│   ├── document_processor.py  # PDF loading and semantic text splitting
│   ├── vector_store.py       # ChromaDB management & MD5 deduplication logic
│   └── rag_chain.py          # LangChain LCEL & streaming response logic
├── data/                 # Raw PDF storage (Local Knowledge Base)
├── chroma_db/            # Persistent Vector Database files
└── chat_history/         # Persistent JSON chat logs
```

---

## 🚀 Getting Started
### 1. Prerequisites
Ensure you have Ollama installed and your model downloaded:

```bash
ollama run deepseek-r1:8b  # or your preferred model
```
If you are using a different model, please update the `LLM_MODEL` variable in `config.py`.

### 2. Installation
Clone the repository and install the required Python packages:

``` bash
git clone https://github.com/ZaneLi11/3D-Housing-RAG-Copilot
cd 3D-Housing-RAG-Copilot
pip install -r requirements.txt
```

### 3. Running the App
Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 🔧 Core Logic Highlights
### Smart Document Loading
The system ensures that each PDF is only indexed once by calculating its MD5 hash. If you attempt to upload the same file twice, the system will identify the duplicate and skip the embedding process to save resources.

### Context-Aware Streaming
By utilizing MessagesPlaceholder and st.write_stream, the assistant provides a fluid, ChatGPT-like experience while maintaining the full context of the ongoing conversation.

---

## 📝 License
Distributed under the MIT License.