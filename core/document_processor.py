import os
import sys

# Standard boilerplate to ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def process_single_pdf(file_name: str):
    """
    Loads a PDF file from the data directory and splits it into text chunks.
    
    Args:
        file_name (str): Name of the PDF file located in the data/ folder.
        
    Returns:
        list: A list of Document objects with page content and metadata.
    """
    # Combine the data directory path with the file name
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Check if the file exists locally before processing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")
        
    # 1. Loading the document
    print(f"Loading document: {file_name}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Initializing the text splitter
    # RecursiveCharacterTextSplitter tries to split at paragraphs, then sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 3. Splitting documents into chunks
    print(f"Splitting document into chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully processed {file_name}. Total chunks: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    # Quick Test: Replace with a file name that exists in your data/ folder
    try:
        test_chunks = process_single_pdf("3D_Printed_Home_Feasibility_Study_FINAL_2021_AHFC_Branded.pdf")
        print(f"Sample Content: {test_chunks[0].page_content[:150]}...")
    except Exception as e:
        print(f"Test failed: {e}")