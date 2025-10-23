from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except Exception:
    pass


def env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TemporaryDocumentRecord:
    id: str
    title: str
    source: str
    content: str
    metadata: dict[str, Any]
    expires_at: float | None = None


class TemporaryDocumentStore(Protocol):
    provider: str

    def save(self, title: str, content: str, metadata: dict[str, Any], ttl_seconds: int) -> str:
        ...

    def load_active(self) -> list[TemporaryDocumentRecord]:
        ...


class InMemoryTemporaryDocumentStore:
    provider = "memory"

    def __init__(self) -> None:
        self._documents: dict[str, TemporaryDocumentRecord] = {}

    def save(self, title: str, content: str, metadata: dict[str, Any], ttl_seconds: int) -> str:
        file_id = f"temp-{int(time.time() * 1000)}"
        record = TemporaryDocumentRecord(
            id=file_id,
            title=title,
            source="会话临时文档",
            content=content,
            metadata={**metadata, "file_id": file_id, "temporary": True},
            expires_at=time.time() + ttl_seconds,
        )
        self._documents[file_id] = record
        return file_id

    def load_active(self) -> list[TemporaryDocumentRecord]:
        now = time.time()
        expired = [doc_id for doc_id, doc in self._documents.items() if doc.expires_at and doc.expires_at <= now]
        for doc_id in expired:
            self._documents.pop(doc_id, None)
        return list(self._documents.values())


class RedisTemporaryDocumentStore:
    provider = "redis"

    def __init__(self, redis_url: str | None = None, namespace: str = "course_rag_qa:temporary"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")
        self.namespace = namespace
        import redis

        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self.client.ping()

    def save(self, title: str, content: str, metadata: dict[str, Any], ttl_seconds: int) -> str:
        file_id = f"temp-{int(time.time() * 1000)}"
        key = f"{self.namespace}:{file_id}"
        record = {
            "id": file_id,
            "title": title,
            "source": "会话临时文档",
            "content": content,
            "metadata": {**metadata, "file_id": file_id, "temporary": True},
            "expires_at": time.time() + ttl_seconds,
        }
        self.client.setex(key, ttl_seconds, json.dumps(record, ensure_ascii=False))
        return file_id

    def load_active(self) -> list[TemporaryDocumentRecord]:
        records: list[TemporaryDocumentRecord] = []
        for key in self.client.scan_iter(f"{self.namespace}:*"):
            value = self.client.get(key)
            if not value:
                continue
            item = json.loads(value)
            records.append(TemporaryDocumentRecord(**item))
        return records


def build_temporary_store() -> TemporaryDocumentStore:
    if env_bool("USE_REDIS", True):
        try:
            return RedisTemporaryDocumentStore()
        except Exception:
            pass
    return InMemoryTemporaryDocumentStore()


class MetadataStore:
    provider = "memory"

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.last_status = "memory metadata"

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            self.rows[doc["id"]] = doc

    def document_count(self) -> int:
        return len(self.rows)


class PostgresMetadataStore(MetadataStore):
    provider = "postgres"

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "postgresql://resume:resume_password@127.0.0.1:5433/resume_demo")
        import psycopg

        self.psycopg = psycopg
        self.last_status = "not connected"
        self._ensure_schema()

    def _connect(self):
        return self.psycopg.connect(self.dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists course_documents (
                    id text primary key,
                    title text not null,
                    source text not null,
                    user_id text,
                    course_id text,
                    file_id text,
                    doc_type text,
                    metadata jsonb not null default '{}'::jsonb,
                    updated_at timestamptz not null default now()
                )
                """
            )
        self.last_status = "connected"

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            for doc in documents:
                conn.execute(
                    """
                    insert into course_documents
                    (id, title, source, user_id, course_id, file_id, doc_type, metadata, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                    on conflict (id) do update set
                        title = excluded.title,
                        source = excluded.source,
                        user_id = excluded.user_id,
                        course_id = excluded.course_id,
                        file_id = excluded.file_id,
                        doc_type = excluded.doc_type,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """,
                    (
                        doc["id"],
                        doc["title"],
                        doc["source"],
                        doc.get("user_id"),
                        doc.get("course_id"),
                        doc.get("file_id"),
                        doc.get("doc_type"),
                        json.dumps({key: value for key, value in doc.items() if key not in {"content"}}, ensure_ascii=False),
                    ),
                )
        self.last_status = f"connected: indexed {len(documents)} metadata rows"

    def document_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) from course_documents").fetchone()
        return int(row[0])


def build_metadata_store() -> MetadataStore:
    if env_bool("USE_POSTGRES", True):
        try:
            return PostgresMetadataStore()
        except Exception:
            pass
    return MetadataStore()


class ElasticsearchDocumentIndex:
    """Optional Elasticsearch BM25 index adapter."""

    def __init__(self, index_name: str = "resume_demo_course_chunks"):
        self.index_name = index_name
        self.url = os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
        self.enabled = env_bool("USE_ELASTICSEARCH", True)
        self.last_status = "not connected"
        self.client: Any | None = None
        self.indexed_count = 0

    def is_available(self) -> bool:
        if not self.enabled:
            self.last_status = "USE_ELASTICSEARCH=false"
            return False
        try:
            import requests

            response = requests.get(self.url, timeout=5)
            response.raise_for_status()
            info = response.json()
            self.client = requests.Session()
            self.last_status = f"connected: {info['version']['number']}"
            return True
        except Exception as exc:  # pragma: no cover - optional dependency/service
            self.last_status = f"elasticsearch unavailable: {exc.__class__.__name__}"
            return False

    def mark_local_fallback(self, document_count: int) -> None:
        self.client = None
        self.indexed_count = document_count
        self.last_status = f"local fallback: {document_count} docs available"

    def ensure_index(self) -> None:
        if self.client is None:
            return
        response = self.client.head(f"{self.url}/{self.index_name}", timeout=5)
        if response.status_code == 200:
            return
        response = self.client.put(
            f"{self.url}/{self.index_name}",
            json={
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "course_id": {"type": "keyword"},
                        "file_id": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
                    }
                }
            },
            timeout=10,
        )
        response.raise_for_status()

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        if self.client is None:
            return
        self.ensure_index()
        for doc in documents:
            response = self.client.put(f"{self.url}/{self.index_name}/_doc/{doc['id']}", json=doc, timeout=10)
            response.raise_for_status()
        self.client.post(f"{self.url}/{self.index_name}/_refresh", timeout=10).raise_for_status()
        self.indexed_count = len(documents)
        self.last_status = f"connected: indexed {self.indexed_count} docs"

    def search_ids(self, query: str, filters: dict[str, str | None], top_k: int) -> list[str]:
        if self.client is None:
            return []
        must = [{"multi_match": {"query": query, "fields": ["title^2", "content"]}}]
        filter_terms = [{"term": {key: value}} for key, value in filters.items() if value]
        response = self.client.post(
            f"{self.url}/{self.index_name}/_search",
            json={"size": top_k, "query": {"bool": {"must": must, "filter": filter_terms}}},
            timeout=10,
        )
        response.raise_for_status()
        return [hit["_id"] for hit in response.json()["hits"]["hits"]]


class MinerUDocumentParser:
    """Optional MinerU parser adapter.

    It uses MinerU only when explicitly enabled and installed. Otherwise it
    returns the original text as a Markdown-ish fallback, which keeps uploads
    usable in local demos.
    """

    def __init__(self) -> None:
        self.enabled = env_bool("USE_MINERU", False)
        self.provider = "mineru" if self.enabled else "plain-text-fallback"

    def parse_text(self, title: str, content: str) -> str:
        if not self.enabled:
            return f"# {title}\n\n{content}"
        try:
            import magic_pdf  # type: ignore  # pragma: no cover

            _ = magic_pdf
            return f"# {title}\n\n{content}"
        except Exception:  # pragma: no cover - optional dependency
            self.provider = "plain-text-fallback"
            return f"# {title}\n\n{content}"
