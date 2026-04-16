import os
import sys
from operator import itemgetter

# Standard path management
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from config import LLM_MODEL, LLM_BASE_URL, SYSTEM_PROMPT


class RAGChainManager:
    def __init__(self, retriever):
        self.retriever = retriever

        # 1. Initialize the LLM
        self.llm = ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL)

        # 2. Prompt with history support
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{input}"),
            ]
        )

        # 3. Build the chain
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Constructs the pipeline and maps variables."""
        return (
            {
                "context": itemgetter("input") | self.retriever | self._format_docs,
                "input": itemgetter("input"),
                "history": itemgetter("history"),
                "memory_context": itemgetter("memory_context"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_stream(self, question: str, history_messages: list, memory_context: str = ""):
        """
        Generates a stream of response tokens.
        history_messages: A list of HumanMessage/AIMessage objects.
        """
        return self.chain.stream(
            {
                "input": question,
                "history": history_messages,
                "memory_context": memory_context,
            }
        )

    def ask(self, question: str, history_messages: list = None, memory_context: str = ""):
        """Standard synchronous invoke (for testing)."""
        return self.chain.invoke(
            {
                "input": question,
                "history": history_messages or [],
                "memory_context": memory_context,
            }
        )
