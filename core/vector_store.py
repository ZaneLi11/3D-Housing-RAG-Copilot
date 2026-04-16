import os
import sys
import hashlib
import json

# Ensure the root directory is in the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from core.document_processor import process_single_pdf
from config import (
    DATA_DIR,
    CHROMA_DB_DIR,
    LLM_BASE_URL,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    VECTOR_COLLECTION_NAME,
)


class KnowledgeBaseManager:
    def __init__(self):
        # 1. Initialize the Embedding Model
        self.embeddings = OllamaEmbeddings(
            base_url=LLM_BASE_URL,
            model=EMBEDDING_MODEL
        )
        
        # 2. Connect to the Chroma database
        self.vector_store = Chroma(
            collection_name=VECTOR_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        
        # 3. Set up the MD5 registry file path
        self.registry_path = os.path.join(DATA_DIR, "md5_registry.json")
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Creates an empty JSON registry array if the file does not exist."""
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump([], f)
    
    def _calculate_md5(self, file_path: str) -> str:
        """
        Calculates the MD5 hash of a file to detect duplicates.
        Reads the file in chunks to prevent memory overload with large PDFs.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_duplicate(self, file_hash: str) -> bool:
        """
        Checks if the file hash already exists in the JSON registry.
        """
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
        return file_hash in registry

    def _register_hash(self, file_hash: str):
        """
        Saves a newly processed file's hash into the JSON registry.
        """
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
            
        registry.append(file_hash)
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
            
    def add_document(self, file_name: str) -> dict:
        """
        Processes a PDF, checks for duplicates via MD5, and adds it to the vector store.
        
        Args:
            file_name (str): The name of the file currently resting in the DATA_DIR.
            
        Returns:
            dict: A status dictionary containing success state and messages.
        """
        file_path = os.path.join(DATA_DIR, file_name)
        
        # Step 1: Check if file actually exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": "File not found."}
            
        # Step 2: Calculate fingerprint and check for duplicates
        file_hash = self._calculate_md5(file_path)
        
        if self._is_duplicate(file_hash):
            return {"status": "warning", "message": f"Document '{file_name}' already exists in the database."}
            
        try:
            # Step 3: Extract and split the document using our processor
            # Note: This calls the function we will write in document_processor.py
            chunks = process_single_pdf(file_name)
            
            # Step 4: Generate unique IDs for each chunk to allow future updates/deletes
            chunk_ids = [f"{file_hash}_chunk_{i}" for i in range(len(chunks))]
            
            # Step 5: Add to Chroma DB
            self.vector_store.add_documents(documents=chunks, ids=chunk_ids)
            
            # Step 6: Mark as processed in our JSON registry
            self._register_hash(file_hash)
            
            return {"status": "success", "message": f"Successfully embedded {len(chunks)} chunks from '{file_name}'."}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_retriever(self):
        """Returns the retriever interface for the RAG chain."""
        return self.vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})


if __name__ == "__main__":
    # Local test block to verify initialization
    manager = KnowledgeBaseManager()
    print("Knowledge Base Manager initialized successfully.")
