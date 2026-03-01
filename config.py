import os

# 1. Absolute Path Configuration
# Ensures the script finds the correct directories regardless of where it is executed from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")

# 2. Model Configuration
# Easily switch to a more powerful LLM in the future by changing these variables.
LLM_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# 3. Document Splitting Configuration
# Core parameters for RAG text extraction and embedding tuning.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# 4. Retrieval Configuration
# Number of most relevant document chunks to retrieve from the database per query.
RETRIEVER_K = 3

# 5. System Prompt Configuration
# Persona tuned specifically for the commercial competition.
SYSTEM_PROMPT = """
You are an expert AI Copilot specializing in 3D-printed housing, construction feasibility, and industry analysis.
Your goal is to help users analyze reports and extract precise data.

Please answer the user's questions strictly based on the provided context below.
If the context does not contain the answer, explicitly state "I cannot find the answer in the provided documents."
Do NOT make up information.

Context information:
{context}
"""