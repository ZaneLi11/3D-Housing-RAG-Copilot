import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Ensure the root directory is in the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHROMA_DB_DIR,
    CHAT_HISTORY_DIR,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    MEMORY_MAX_FACTS_PER_UPDATE,
    MEMORY_RETRIEVER_K,
    MEMORY_SUMMARY_INTERVAL_MESSAGES,
)


class HierarchicalMemoryManager:
    """
    L1: session summary memory (rolling summary file)
    L2: long-term fact memory (vector store + dedup registry)
    """

    def __init__(self):
        self.summary_state_path = os.path.join(CHAT_HISTORY_DIR, "memory_state.json")
        self.fact_registry_path = os.path.join(CHAT_HISTORY_DIR, "memory_fact_registry.json")
        self._ensure_files()

        self.llm = ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0)
        self.embeddings = OllamaEmbeddings(base_url=LLM_BASE_URL, model=EMBEDDING_MODEL)
        self.memory_store = Chroma(
            collection_name="chat_long_term_memory",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

    def _ensure_files(self):
        if not os.path.exists(self.summary_state_path):
            self._save_json(
                self.summary_state_path,
                {"summary": "", "last_message_index": 0, "updated_at": None},
            )
        if not os.path.exists(self.fact_registry_path):
            self._save_json(self.fact_registry_path, [])

    def _load_json(self, path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _to_text_block(self, messages: List[Dict]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _normalize_llm_output(self, content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def _update_session_summary(self, old_summary: str, new_messages: List[Dict]) -> str:
        transcript = self._to_text_block(new_messages)
        if not transcript:
            return old_summary

        prompt = f"""
You are a memory summarizer for a technical assistant.
Update the rolling session summary using the previous summary and the new transcript.

Rules:
- Keep only durable and useful context.
- Keep decisions, constraints, and open tasks.
- Remove small talk and duplicates.
- Maximum 180 words.

Previous summary:
{old_summary or "(empty)"}

New transcript:
{transcript}

Return only the updated summary.
""".strip()

        response = self.llm.invoke(prompt)
        updated = self._normalize_llm_output(response.content)
        return updated or old_summary

    def _extract_candidate_facts(self, new_messages: List[Dict]) -> List[Dict]:
        transcript = self._to_text_block(new_messages)
        if not transcript:
            return []

        prompt = f"""
Extract reusable long-term facts from the transcript for future assistant turns.
Only keep stable facts such as user preference, project goal, hard constraint, accepted decision.
Ignore temporary details.

Output format:
One fact per line:
type|confidence|fact

type must be one of: preference, goal, constraint, decision
confidence must be between 0 and 1
Return at most {MEMORY_MAX_FACTS_PER_UPDATE} facts.

Transcript:
{transcript}
""".strip()

        response = self.llm.invoke(prompt)
        raw = self._normalize_llm_output(response.content)
        facts: List[Dict] = []

        for line in raw.splitlines():
            item = line.strip().lstrip("-").strip()
            if not item or "|" not in item:
                continue
            parts = [x.strip() for x in item.split("|", 2)]
            if len(parts) != 3:
                continue
            fact_type, confidence_text, fact_text = parts
            if fact_type not in {"preference", "goal", "constraint", "decision"}:
                continue
            try:
                confidence = float(confidence_text)
            except ValueError:
                continue
            if not fact_text:
                continue
            facts.append(
                {
                    "type": fact_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "fact": fact_text,
                }
            )

        return facts[:MEMORY_MAX_FACTS_PER_UPDATE]

    def _upsert_facts(self, facts: List[Dict]):
        if not facts:
            return

        registry = set(self._load_json(self.fact_registry_path, []))
        docs: List[Document] = []
        ids: List[str] = []

        for item in facts:
            if item["confidence"] < 0.65:
                continue

            normalized = item["fact"].strip().lower()
            fact_id = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            if fact_id in registry:
                continue

            docs.append(
                Document(
                    page_content=item["fact"].strip(),
                    metadata={
                        "type": item["type"],
                        "confidence": item["confidence"],
                        "created_at": self._utc_now(),
                    },
                )
            )
            ids.append(fact_id)
            registry.add(fact_id)

        if docs:
            self.memory_store.add_documents(documents=docs, ids=ids)
            self._save_json(self.fact_registry_path, sorted(registry))

    def maybe_update_memories(self, messages: List[Dict]):
        state = self._load_json(
            self.summary_state_path,
            {"summary": "", "last_message_index": 0, "updated_at": None},
        )
        last_index = int(state.get("last_message_index", 0))
        new_count = len(messages) - last_index

        if new_count < MEMORY_SUMMARY_INTERVAL_MESSAGES:
            return

        new_messages = messages[last_index:]
        updated_summary = self._update_session_summary(state.get("summary", ""), new_messages)
        state["summary"] = updated_summary
        state["last_message_index"] = len(messages)
        state["updated_at"] = self._utc_now()
        self._save_json(self.summary_state_path, state)

        facts = self._extract_candidate_facts(new_messages)
        self._upsert_facts(facts)

    def _retrieve_relevant_facts(self, query: str, k: int = MEMORY_RETRIEVER_K) -> List[Document]:
        if not query.strip():
            return []

        try:
            docs = self.memory_store.similarity_search(query, k=k)
        except Exception:
            docs = []
        return docs

    def get_memory_context(self, query: str) -> str:
        state = self._load_json(
            self.summary_state_path,
            {"summary": "", "last_message_index": 0, "updated_at": None},
        )
        summary = state.get("summary", "").strip()
        facts = self._retrieve_relevant_facts(query, k=MEMORY_RETRIEVER_K)

        lines = []
        if summary:
            lines.append("Session summary:")
            lines.append(summary)

        if facts:
            lines.append("Long-term facts:")
            for i, doc in enumerate(facts, start=1):
                fact_type = doc.metadata.get("type", "fact")
                lines.append(f"{i}. [{fact_type}] {doc.page_content}")

        return "\n".join(lines).strip()

    def reset_session_memory(self):
        self._save_json(
            self.summary_state_path,
            {"summary": "", "last_message_index": 0, "updated_at": None},
        )
