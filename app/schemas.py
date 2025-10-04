from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    user_id: str = "demo"
    course_id: str | None = None
    file_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class TemporaryDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=5)
    user_id: str = "demo"
    course_id: str = "temp"
    ttl_seconds: int = Field(default=7200, ge=60)


class Citation(BaseModel):
    index: int
    title: str
    source: str
    score: float
    metadata: dict[str, Any]
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    followups: list[str]
    trace: list[dict[str, Any]]


class TemporaryDocumentResponse(BaseModel):
    file_id: str
    ttl_seconds: int
