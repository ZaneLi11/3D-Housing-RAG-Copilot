import os

# 1. Absolute Path Configuration
# Ensures the script finds the correct directories regardless of where it is executed from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")

CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "research_chat.json")

# Ensure required directories exist on startup.
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# 2. Model Configuration
# Easily switch to a more powerful LLM in the future by changing these variables.
LLM_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# 2.1 App Identity
APP_TITLE = "Personal Literature Research Assistant"
VECTOR_COLLECTION_NAME = "personal_literature_collection"

# 3. Document Splitting Configuration
# Core parameters for RAG text extraction and embedding tuning.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# 4. Retrieval Configuration
# Number of most relevant document chunks to retrieve from the database per query.
RETRIEVER_K = 3

# 5. Hierarchical Memory Configuration
# L1 summary refresh interval (message count) and L2 long-term memory retrieval size.
MEMORY_SUMMARY_INTERVAL_MESSAGES = 6
MEMORY_RETRIEVER_K = 3
MEMORY_MAX_FACTS_PER_UPDATE = 5

# 6. System Prompt Configuration
# Persona tuned for personal literature analysis and evidence-backed responses.
SYSTEM_PROMPT = """
You are a personal literature research assistant.
Your goal is to help users read, compare, and synthesize academic papers and technical documents.

Please answer the user's questions strictly based on the provided context below.
If the context does not contain the answer, explicitly state "I cannot find the answer in the provided documents."
Do NOT make up information.
When possible, quote short supporting snippets from context and clearly separate facts from your inference.

Memory context from previous conversation:
{memory_context}

Context information:
{context}
"""
