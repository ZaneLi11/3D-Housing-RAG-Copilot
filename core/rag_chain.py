import os
import sys

# Standard path management
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, LLM_BASE_URL, SYSTEM_PROMPT


class RAGChainManager:
    def __init__(self, retriever):
        """
        Initializes the RAG pipeline by connecting the LLM, Prompt, and Retriever.
        
        Args:
            retriever: The Chroma database retriever instance.
        """
        self.retriever = retriever
        
        # 1. Initialize the LLM (The Brain)
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL
        )
        
        # 2. Build the Prompt (The Instructions)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{input}")
        ])
        
        # 3. Assemble the Chain using LCEL (LangChain Expression Language)
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        """Formats the retrieved document chunks into a clean text block for the LLM."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Constructs the executable RAG pipeline."""
        # This is the "Magic" pipeline
        return (
            {
                "context": self.retriever | self._format_docs, 
                "input": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Executes the chain and returns the final answer."""
        return self.chain.invoke(question)
    
if __name__ == "__main__":
    from core.vector_store import KnowledgeBaseManager
    
    # Initialize the database manager first
    kb_manager = KnowledgeBaseManager()
    
    # Get the retriever (The interface we defined earlier!)
    retriever = kb_manager.get_retriever()
    
    # Create the RAG chain
    rag_manager = RAGChainManager(retriever)
    
    # Test a question
    print("Testing RAG Chain...")
    response = rag_manager.ask("How does weather impact 3D printing housing building?")
    print(f"\nAI Response:\n{response}")