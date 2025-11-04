from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict

from .integrations import ElasticsearchDocumentIndex, MinerUDocumentParser, build_metadata_store
from .retriever import CourseKnowledgeBase


class QAState(TypedDict, total=False):
    question: str
    filters: dict[str, str | None]
    top_k: int
    hits: list[Any]
    citations: list[dict[str, Any]]
    context: str
    prompt: str
    answer: str
    followups: list[str]
    trace: list[dict[str, Any]]


class ChainLike(Protocol):
    def invoke(self, state: QAState) -> QAState:
        ...


class CourseQAWorkflow:
    """LangChain-style RAG chain for course-material QA.

    The project prefers LangChain LCEL (`RunnableLambda | RunnableLambda`).
    If LangChain is not installed in the interview environment, it falls back
    to a tiny local chain executor with the same step boundaries.
    """

    def __init__(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "sample_documents.json"
        self.kb = CourseKnowledgeBase(data_path)
        self.document_parser = MinerUDocumentParser()
        self.search_index = ElasticsearchDocumentIndex()
        self.search_index_available = self.search_index.is_available()
        if self.search_index_available:
            self.search_index.upsert_documents(self.kb.indexable_documents())
        else:
            self.search_index.mark_local_fallback(len(self.kb.indexable_documents()))
        self.metadata_store = build_metadata_store()
        self.metadata_store.upsert_documents(self.kb.indexable_documents())
        self.chain = self._build_chain()

    def add_temporary_document(self, title: str, content: str, user_id: str, course_id: str, ttl_seconds: int) -> str:
        parsed_content = self.document_parser.parse_text(title, content)
        return self.kb.add_temporary_document(
            title=title,
            content=parsed_content,
            metadata={"user_id": user_id, "course_id": course_id},
            ttl_seconds=ttl_seconds,
        )

    def ask(self, question: str, user_id: str, course_id: str | None, file_id: str | None, top_k: int) -> QAState:
        return self.chain.invoke(
            {
                "question": question,
                "filters": {"user_id": user_id, "course_id": course_id, "file_id": file_id},
                "top_k": top_k,
                "trace": [],
            }
        )

    def _build_chain(self) -> ChainLike:
        try:
            from langchain_core.runnables import RunnableLambda

            return (
                RunnableLambda(self._retrieve_documents)
                | RunnableLambda(self._format_context)
                | RunnableLambda(self._build_prompt)
                | RunnableLambda(self._generate_answer)
                | RunnableLambda(self._review_citations)
            )
        except ModuleNotFoundError:
            return LocalRAGChain(
                [
                    self._retrieve_documents,
                    self._format_context,
                    self._build_prompt,
                    self._generate_answer,
                    self._review_citations,
                ]
            )

    def _retrieve_documents(self, state: QAState) -> QAState:
        es_ids = (
            self.search_index.search_ids(state["question"], filters=state["filters"], top_k=state["top_k"])
            if self.search_index_available
            else []
        )
        state["hits"] = self.kb.search(
            state["question"],
            top_k=state["top_k"],
            filters=state["filters"],
            preferred_ids=es_ids,
        )
        state["citations"] = [hit.citation(i + 1) for i, hit in enumerate(state["hits"])]
        state["trace"].append(
            {
                "node": "langchain_retriever",
                "message": f"召回 Top{len(state['hits'])} 课程片段",
                "retrieval_backend": "elasticsearch" if es_ids else "local_bm25",
                "temporary_store": self.kb.temporary_store.provider,
                "elasticsearch": self.search_index.last_status,
                "elasticsearch_indexed_count": self.search_index.indexed_count,
                "postgres": self.metadata_store.last_status,
                "postgres_document_count": self.metadata_store.document_count(),
                "parser": self.document_parser.provider,
            }
        )
        return state

    def _format_context(self, state: QAState) -> QAState:
        state["context"] = "\n".join(f"[{i + 1}] {hit.snippet}" for i, hit in enumerate(state["hits"]))
        state["trace"].append({"node": "context_formatter", "message": "格式化引用上下文"})
        return state

    def _build_prompt(self, state: QAState) -> QAState:
        state["prompt"] = (
            "你是课程资料问答助手。请仅依据资料片段回答，并保留引用编号。\n\n"
            f"问题：{state['question']}\n\n资料片段：\n{state['context']}"
        )
        state["trace"].append({"node": "prompt_template", "message": "构造带引用约束的 Prompt"})
        return state

    def _generate_answer(self, state: QAState) -> QAState:
        if not state["hits"]:
            state["answer"] = "当前课程资料范围内没有找到直接依据，请检查 course_id / file_id 或上传临时文档。"
        else:
            lines = ["回答："]
            for i, hit in enumerate(state["hits"], 1):
                lines.append(f"[{i}] {hit.snippet}")
            state["answer"] = "\n".join(lines)
        state["followups"] = ["这部分通常怎么考？", "能给一个例子吗？", "和上一章有什么联系？"]
        state["trace"].append({"node": "answer_chain", "message": "生成答案和推荐追问"})
        return state

    def _review_citations(self, state: QAState) -> QAState:
        supported = bool(state.get("citations"))
        state["trace"].append({"node": "citation_checker", "message": f"引用检查：{'通过' if supported else '未找到引用'}"})
        return state


class LocalRAGChain:
    def __init__(self, steps):
        self.steps = steps

    def invoke(self, state: QAState) -> QAState:
        for step in self.steps:
            state = step(state)
        return state
