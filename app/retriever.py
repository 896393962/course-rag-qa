from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .integrations import TemporaryDocumentRecord, TemporaryDocumentStore, build_temporary_store


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class CourseDocument:
    id: str
    title: str
    source: str
    content: str
    metadata: dict[str, Any]
    expires_at: float | None = None


@dataclass(frozen=True)
class RetrievalHit:
    document: CourseDocument
    score: float
    snippet: str

    def citation(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "title": self.document.title,
            "source": self.document.source,
            "score": round(self.score, 4),
            "metadata": self.document.metadata,
            "snippet": self.snippet,
        }


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class CourseKnowledgeBase:
    """独立课程知识库，支持 user/course/file 隔离和临时文档 TTL."""

    def __init__(self, data_path: str | Path, temporary_store: TemporaryDocumentStore | None = None):
        raw_items = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.documents = [CourseDocument(**item) for item in raw_items]
        self.temporary_store = temporary_store or build_temporary_store()

    def add_temporary_document(self, title: str, content: str, metadata: dict[str, Any], ttl_seconds: int) -> str:
        return self.temporary_store.save(title=title, content=content, metadata=metadata, ttl_seconds=ttl_seconds)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | None],
        preferred_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        self._drop_expired()
        preferred_ids = preferred_ids or []
        preferred_rank = {doc_id: index for index, doc_id in enumerate(preferred_ids)}
        docs = [doc for doc in self._all_documents() if self._match_filters(doc, filters)]
        query_counts = Counter(tokenize(query))
        doc_freq = Counter()
        doc_counts = {doc.id: Counter(tokenize(doc.title + " " + doc.content)) for doc in docs}
        for counts in doc_counts.values():
            doc_freq.update(counts.keys())

        hits: list[RetrievalHit] = []
        total = max(len(docs), 1)
        for doc in docs:
            score = 0.0
            counts = doc_counts[doc.id]
            for term, q_count in query_counts.items():
                if counts[term] == 0:
                    continue
                idf = math.log((1 + total) / (1 + doc_freq[term])) + 1
                score += (1 + math.log(counts[term])) * idf * q_count
            if doc.id in preferred_rank:
                score += 10.0 - preferred_rank[doc.id] * 0.1
            if score > 0:
                hits.append(RetrievalHit(document=doc, score=score, snippet=doc.content[:220]))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def _match_filters(self, doc: CourseDocument, filters: dict[str, str | None]) -> bool:
        return all(not value or str(doc.metadata.get(key)) == value for key, value in filters.items())

    def _drop_expired(self) -> None:
        now = time.time()
        self.documents = [doc for doc in self.documents if doc.expires_at is None or doc.expires_at > now]

    def _all_documents(self) -> list[CourseDocument]:
        temporary_docs = [self._from_temporary(record) for record in self.temporary_store.load_active()]
        return [*self.documents, *temporary_docs]

    def indexable_documents(self) -> list[dict[str, Any]]:
        rows = []
        for doc in self.documents:
            rows.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "content": doc.content,
                    **doc.metadata,
                }
            )
        return rows

    def _from_temporary(self, record: TemporaryDocumentRecord) -> CourseDocument:
        return CourseDocument(
            id=record.id,
            title=record.title,
            source=record.source,
            content=record.content,
            metadata=record.metadata,
            expires_at=record.expires_at,
        )
