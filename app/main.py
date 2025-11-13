from __future__ import annotations

from fastapi import FastAPI

from .evaluation import evaluate
from .schemas import AskRequest, AskResponse, TemporaryDocumentRequest, TemporaryDocumentResponse
from .workflow import CourseQAWorkflow


app = FastAPI(title="Course RAG QA", version="1.0.0")
workflow = CourseQAWorkflow()


@app.get("/health")
def health():
    return {"status": "ok", "project": "course_rag_qa"}


@app.get("/integrations/status")
def integrations_status():
    return {
        "temporary_store": {"provider": workflow.kb.temporary_store.provider},
        "elasticsearch": {
            "available": workflow.search_index_available,
            "status": workflow.search_index.last_status,
            "indexed_count": workflow.search_index.indexed_count,
        },
        "postgres": {
            "provider": workflow.metadata_store.provider,
            "status": workflow.metadata_store.last_status,
            "document_count": workflow.metadata_store.document_count(),
        },
        "parser": {"provider": workflow.document_parser.provider},
    }


@app.post("/documents/temporary", response_model=TemporaryDocumentResponse)
def add_temporary_document(request: TemporaryDocumentRequest):
    file_id = workflow.add_temporary_document(
        title=request.title,
        content=request.content,
        user_id=request.user_id,
        course_id=request.course_id,
        ttl_seconds=request.ttl_seconds,
    )
    return {"file_id": file_id, "ttl_seconds": request.ttl_seconds}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    state = workflow.ask(request.question, request.user_id, request.course_id, request.file_id, request.top_k)
    return {
        "answer": state["answer"],
        "citations": state["citations"],
        "followups": state["followups"],
        "trace": state["trace"],
    }


@app.get("/metrics/retrieval")
def retrieval_metrics(top_k: int = 5):
    return evaluate(top_k=top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8102)
