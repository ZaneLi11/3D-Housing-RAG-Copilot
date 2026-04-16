# Personal Literature Research Assistant

一个本地优先（local-first）的个人文献研究检索助手。  
支持上传 PDF、建立向量索引、基于证据对话问答，并逐步构建长短期记忆。

## 已更新功能
- PDF 文档上传与语义切分（`PyMuPDF + RecursiveCharacterTextSplitter`）
- Chroma 向量检索（本地持久化）
- 文档去重（MD5 指纹）
- Streamlit 对话界面与会话持久化
- 分级记忆（长短期记忆）
  - 短期记忆（L0）：当前对话历史
  - 会话摘要记忆（L1）：按消息阈值滚动更新
  - 长期事实记忆（L2）：提取稳定事实并向量检索回注入

## 本次更新内容（长短期记忆）
- 新增 `core/memory_manager.py` 统一管理记忆读写逻辑
- 在问答链路中接入 `memory_context`，回答时同时参考：
  - 对话历史（L0）
  - 会话摘要（L1）
  - 长期事实（L2）
  - 文档检索上下文（RAG）
- 新增记忆参数（`config.py`）：
  - `MEMORY_SUMMARY_INTERVAL_MESSAGES`
  - `MEMORY_RETRIEVER_K`
  - `MEMORY_MAX_FACTS_PER_UPDATE`
- 新增会话重置时的记忆状态清理

## 即将更新功能
- RAG 重排序（Two-stage retrieval + reranker）
- 文献来源追踪（文件名/页码/引用片段展示）
- Tool 协议与 Skill 协议（可扩展任务编排）
- 多会话管理（研究主题分组、会话切换）
- 记忆质量控制（手动确认写入、记忆可视化与编辑）
- 基础评测闭环（检索命中率、忠实度、相关性）

## 技术栈
- Python 3.10+
- Streamlit
- LangChain
- Ollama（`deepseek-r1:8b`, `nomic-embed-text:v1.5`）
- ChromaDB

## 项目结构
```text
.
|-- app.py
|-- config.py
|-- requirements.txt
|-- core/
|   |-- document_processor.py
|   |-- memory_manager.py
|   |-- rag_chain.py
|   `-- vector_store.py
|-- data/
|-- chroma_db/
`-- chat_history/
```

## 快速开始
```bash
conda create -n research-rag python=3.11 -y
conda activate research-rag
pip install -r requirements.txt
```

拉取 Ollama 模型：
```bash
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text:v1.5
```

启动项目：
```bash
streamlit run app.py
```

## 备注
- 若大幅修改提示词或领域后检索效果下降，可清空 `chroma_db/` 后重新索引。
- 当前会话文件默认路径：`chat_history/research_chat.json`。
