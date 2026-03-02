import os
import sys

# Standard path management
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from config import LLM_MODEL, LLM_BASE_URL, SYSTEM_PROMPT

class RAGChainManager:
    def __init__(self, retriever):
        self.retriever = retriever
        
        # 1. Initialize the LLM
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL
        )
        
        # 2. Updated Prompt: Added MessagesPlaceholder for conversation memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"), # This is where old chats go
            ("user", "{input}")
        ])
        
        # 3. Build the Chain
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Constructs the pipeline and maps variables."""
        return (
                    {
                        # 这里的 itemgetter("input") 是关键！它确保检索器只拿到问题字符串
                        "context": itemgetter("input") | self.retriever | self._format_docs, 
                        "input": itemgetter("input"),
                        "history": itemgetter("history")
                    }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_stream(self, question: str, history_messages: list):
        """
        Generates a stream of response tokens.
        history_messages: A list of HumanMessage/AIMessage objects.
        """
        # We pass a dictionary that matches the keys in _build_chain
        return self.chain.stream({
            "input": question,
            "history": history_messages
        })

    def ask(self, question: str, history_messages: list = None):
        """Standard synchronous invoke (for testing)."""
        return self.chain.invoke({
            "input": question,
            "history": history_messages or []
        })