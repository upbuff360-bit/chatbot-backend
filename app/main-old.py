from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.admin_store import AdminStore
from app.crawl_job_store import CrawlJobStore
from app.manual_knowledge_service import ManualKnowledgeService
from app.rag_pipeline import IngestionStats, RAGPipeline
from app.vector_store import VectorStore
from app.website_service import WebsiteService

load_dotenv()


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool
    documents_loaded: int
    chunks_indexed: int
    startup_error: Optional[str] = None


class ActivityResponse(BaseModel):
    id: str
    type: str
    agent_id: str
    description: str
    timestamp: str


class DashboardSummaryResponse(BaseModel):
    total_agents: int
    total_documents: int
    total_conversations: int
    recent_activity: list[ActivityResponse]


class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AgentUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    id: str
    name: str
    created_at: str
    document_count: int
    conversation_count: int


class KnowledgeDocumentResponse(BaseModel):
    id: str
    file_name: str
    uploaded_at: str
    status: str
    source_type: str = "pdf"
    source_url: str | None = None
    content: str | None = None
    question: str | None = None
    answer: str | None = None
    page_count: int | None = None
    page_urls: list[str] | None = None


class KnowledgeDocumentUpdateRequest(BaseModel):
    file_name: str | None = Field(default=None, min_length=1)
    content: str | None = None
    answer: str | None = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class AgentSettingsResponse(BaseModel):
    system_prompt: str
    temperature: float
    welcome_message: str


class AgentSettingsUpdateRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=0.0, le=1.0)
    welcome_message: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


class WebsiteCrawlRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


class TextSnippetCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class QACreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class CrawlJobResponse(BaseModel):
    id: str
    agent_id: str
    source_url: str
    status: str
    stage: str
    discovered_pages: int
    indexed_pages: int
    current_url: str | None = None
    message: str | None = None
    error: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    source_type: str = "website"


admin_store: AdminStore | None = None
startup_error: str | None = None
startup_stats = IngestionStats(documents_loaded=0, chunks_indexed=0)
crawl_job_store = CrawlJobStore()


def agent_collection_name(agent_id: str) -> str:
    return f"knowledge_base_{agent_id.replace('-', '_')}"


def require_store() -> AdminStore:
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service has not finished starting.")
    return admin_store


def build_pipeline(agent_id: str) -> RAGPipeline:
    store = require_store()
    pdf_directory = store.get_agent_pdf_dir(agent_id)
    website_directory = store.get_agent_website_dir(agent_id)
    snippets_directory = store.get_agent_snippet_dir(agent_id)
    qa_directory = store.get_agent_qa_dir(agent_id)
    return RAGPipeline(
        pdf_directory=pdf_directory,
        website_directory=website_directory,
        snippets_directory=snippets_directory,
        qa_directory=qa_directory,
        collection_name=agent_collection_name(agent_id),
    )


def normalize_website_url(url: str) -> str:
    normalized = url.strip()
    if normalized and not urlparse(normalized).scheme:
        normalized = f"https://{normalized}"
    return normalized


def serialize_crawl_job(job_id: str) -> CrawlJobResponse:
    job = crawl_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Crawl job '{job_id}' was not found.")
    return CrawlJobResponse(**job.to_dict())


def refresh_agent_index(agent_id: str) -> IngestionStats:
    store = require_store()
    pipeline = build_pipeline(agent_id)
    stats = pipeline.ingest_documents(recreate=True)
    store.sync_documents(
        agent_id,
        [path.name for path in store.get_agent_pdf_dir(agent_id).glob("*.pdf")],
    )
    return stats


# FIX: New helper that resolves the stable doc_id for a document record.
# This is the value stored in each chunk's payload during ingest, so Qdrant
# can filter and delete exactly the right set of vectors.
#
#   PDF         → file name   (PDFService sets doc_id = pdf_path.name)
#   Website     → source_url  (WebsiteService sets doc_id = root source URL)
#   Text snippet→ document id (ManualKnowledgeService sets doc_id = record.id)
#   Q&A         → document id (same)
#
# Falls back to an empty string (no-op delete) if the field is missing, which
# is safe — it just means old chunks without doc_id stay in the collection
# until the next full refresh.
def resolve_doc_id(document: dict) -> str:
    source_type = document.get("source_type", "pdf")
    if source_type == "pdf":
        return str(document.get("file_name") or "")
    if source_type == "website":
        return str(document.get("source_url") or "")
    # text_snippet and qa both use the admin-store UUID
    return str(document.get("id") or "")


def serialize_document(agent_id: str, document: dict) -> KnowledgeDocumentResponse:
    store = require_store()
    payload = dict(document)
    source_type = payload.get("source_type")
    if source_type == "website":
        source = WebsiteService(store.get_agent_website_dir(agent_id)).list_sources().get(str(payload.get("source_url") or "").strip())
        if source is not None:
            payload["page_count"] = source.page_count
            payload["page_urls"] = source.page_urls
    elif source_type in {"text_snippet", "qa"}:
        manual_service = ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id),
            store.get_agent_qa_dir(agent_id),
        )
        if source_type == "text_snippet":
            record = manual_service.list_text_snippets().get(str(payload.get("id")))
            if record is not None:
                payload["content"] = record.content
        else:
            record = manual_service.list_qa().get(str(payload.get("id")))
            if record is not None:
                payload["question"] = record.question
                payload["answer"] = record.answer
    return KnowledgeDocumentResponse(**payload)


def run_website_crawl_job(job_id: str) -> None:
    job = crawl_job_store.get(job_id)
    if job is None:
        return

    store = require_store()
    website_service = WebsiteService(store.get_agent_website_dir(job.agent_id))
    crawl_job_store.update(
        job_id,
        status="running",
        stage="crawling",
        message="Starting website crawl.",
    )

    def on_progress(update: dict[str, int | str | None]) -> None:
        crawl_job_store.update(
            job_id,
            stage=str(update.get("stage") or "crawling"),
            discovered_pages=int(update.get("discovered_pages") or 0),
            indexed_pages=int(update.get("indexed_pages") or 0),
            current_url=str(update.get("current_url")) if update.get("current_url") else None,
            message=str(update.get("message")) if update.get("message") else None,
        )

    try:
        crawl = website_service.crawl(job.source_url, progress_callback=on_progress)
        crawl_job_store.update(
            job_id,
            stage="indexing",
            message=f"Building embeddings for {len(crawl.pages)} website pages.",
            discovered_pages=len(crawl.pages),
            indexed_pages=len(crawl.pages),
            current_url=None,
        )
        website_service.save_crawl(crawl)
        pipeline = build_pipeline(job.agent_id)
        pipeline.ingest_documents(recreate=True)
        document = store.upsert_website_source(
            job.agent_id,
            display_name=crawl.display_name,
            source_url=crawl.source_url,
            status="indexed",
        )
        crawl_job_store.update(
            job_id,
            status="completed",
            stage="completed",
            message=f"Indexed {len(crawl.pages)} pages successfully.",
            discovered_pages=len(crawl.pages),
            indexed_pages=len(crawl.pages),
            document_id=document["id"],
            document_name=document["file_name"],
        )
    except ValueError as exc:
        store.upsert_website_source(
            job.agent_id,
            display_name=job.source_url,
            source_url=job.source_url,
            status="failed",
        )
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            message="Website crawl failed.",
        )
    except Exception as exc:
        store.upsert_website_source(
            job.agent_id,
            display_name=job.source_url,
            source_url=job.source_url,
            status="failed",
        )
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            message="Website crawl failed.",
        )


def summarize_agent(agent: dict) -> AgentResponse:
    return AgentResponse(
        id=agent["id"],
        name=agent["name"],
        created_at=agent["created_at"],
        document_count=agent["document_count"],
        conversation_count=agent["conversation_count"],
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global admin_store
    global startup_error
    global startup_stats

    admin_store = AdminStore()
    startup_error = None
    startup_stats = IngestionStats(documents_loaded=0, chunks_indexed=0)

    try:
        for agent in admin_store.list_agents():
            pipeline = build_pipeline(agent["id"])
            stats = pipeline.ingest_documents(recreate=True)
            admin_store.sync_documents(
                agent["id"],
                [path.name for path in admin_store.get_agent_pdf_dir(agent["id"]).glob("*.pdf")],
            )
            startup_stats.documents_loaded += stats.documents_loaded
            startup_stats.chunks_indexed += stats.chunks_indexed
    except Exception as exc:
        startup_error = f"Startup sync failed: {exc}"

    yield
    VectorStore.close_shared_client()


app = FastAPI(
    title="PDF RAG API",
    description="Retrieval-Augmented Generation API over local PDF documents.",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in (
        os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        .split(",")
    )
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    store = require_store()
    summary = store.dashboard_summary()
    return HealthResponse(
        status="ok" if startup_error is None else "degraded",
        openai_configured=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        documents_loaded=summary["total_documents"],
        chunks_indexed=startup_stats.chunks_indexed,
        startup_error=startup_error,
    )


@app.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary() -> DashboardSummaryResponse:
    store = require_store()
    return DashboardSummaryResponse(**store.dashboard_summary())


@app.get("/agents", response_model=list[AgentResponse])
def list_agents() -> list[AgentResponse]:
    store = require_store()
    return [summarize_agent(agent) for agent in store.list_agents()]


@app.post("/agents", response_model=AgentResponse, status_code=201)
def create_agent(request: AgentCreateRequest) -> AgentResponse:
    store = require_store()
    agent = store.create_agent(request.name)
    return summarize_agent(agent)


@app.put("/agents/{agent_id}", response_model=AgentResponse)
def update_agent(agent_id: str, request: AgentUpdateRequest) -> AgentResponse:
    store = require_store()
    try:
        agent = store.update_agent(agent_id, request.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summarize_agent(agent)


@app.get("/agents/{agent_id}", response_model=AgentResponse)
def get_agent(agent_id: str) -> AgentResponse:
    store = require_store()
    try:
        agent = store.require_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summarize_agent(
        {
            "id": agent["id"],
            "name": agent["name"],
            "created_at": agent["created_at"],
            "document_count": len(agent["documents"]),
            "conversation_count": len(agent["conversations"]),
        }
    )


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> None:
    store = require_store()
    try:
        store.require_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    pdf_dir = store.get_agent_pdf_dir(agent_id).parent
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir, ignore_errors=True)

    vector_store = VectorStore(collection_name=agent_collection_name(agent_id))
    vector_store.delete_collection()
    store.delete_agent(agent_id)


@app.get("/agents/{agent_id}/documents", response_model=list[KnowledgeDocumentResponse])
def list_documents(agent_id: str) -> list[KnowledgeDocumentResponse]:
    store = require_store()
    try:
        documents = store.list_documents(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [serialize_document(agent_id, document) for document in documents]


@app.post("/upload-pdf", response_model=KnowledgeDocumentResponse, status_code=201)
async def upload_pdf(
    file: UploadFile = File(...),
    agent_id: str = Form(...),
) -> KnowledgeDocumentResponse:
    store = require_store()
    try:
        store.require_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_name = (file.filename or "").strip()
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    agent_pdf_dir = store.get_agent_pdf_dir(agent_id)
    agent_pdf_dir.mkdir(parents=True, exist_ok=True)
    target_path = agent_pdf_dir / Path(file_name).name
    payload = await file.read()
    target_path.write_bytes(payload)

    store.mark_document_uploaded(agent_id, target_path.name, status="indexing", source_type="pdf")

    try:
        refresh_agent_index(agent_id)
        documents = store.list_documents(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    for document in documents:
        if document["file_name"] == target_path.name:
            return serialize_document(agent_id, document)

    raise HTTPException(status_code=500, detail="Upload completed but document could not be loaded.")


@app.post("/crawl-website", response_model=CrawlJobResponse, status_code=202)
def crawl_website(request: WebsiteCrawlRequest, background_tasks: BackgroundTasks) -> CrawlJobResponse:
    store = require_store()
    try:
        store.require_agent(request.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_url = normalize_website_url(request.url)
    if not source_url:
        raise HTTPException(status_code=400, detail="Website URL is required.")

    store.upsert_website_source(request.agent_id, display_name=source_url, source_url=source_url, status="indexing")
    job = crawl_job_store.create(agent_id=request.agent_id, source_url=source_url)
    background_tasks.add_task(run_website_crawl_job, job.id)
    return CrawlJobResponse(**job.to_dict())


@app.get("/crawl-website/{job_id}", response_model=CrawlJobResponse)
def get_website_crawl_job(job_id: str) -> CrawlJobResponse:
    return serialize_crawl_job(job_id)


@app.post("/agents/{agent_id}/text-snippets", response_model=KnowledgeDocumentResponse, status_code=201)
def create_text_snippet(agent_id: str, request: TextSnippetCreateRequest) -> KnowledgeDocumentResponse:
    store = require_store()
    try:
        store.require_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document = store.mark_document_uploaded(
        agent_id,
        request.title.strip(),
        status="indexed",
        source_type="text_snippet",
    )
    manual_service = ManualKnowledgeService(store.get_agent_snippet_dir(agent_id), store.get_agent_qa_dir(agent_id))
    try:
        manual_service.save_text_snippet(document["id"], request.title, request.content)
        refresh_agent_index(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save text snippet: {exc}") from exc

    return serialize_document(agent_id, document)


@app.post("/agents/{agent_id}/qa", response_model=KnowledgeDocumentResponse, status_code=201)
def create_qa(agent_id: str, request: QACreateRequest) -> KnowledgeDocumentResponse:
    store = require_store()
    try:
        store.require_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document = store.mark_document_uploaded(
        agent_id,
        request.question.strip(),
        status="indexed",
        source_type="qa",
    )
    manual_service = ManualKnowledgeService(store.get_agent_snippet_dir(agent_id), store.get_agent_qa_dir(agent_id))
    try:
        manual_service.save_qa(document["id"], request.question, request.answer)
        refresh_agent_index(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save Q&A item: {exc}") from exc

    return serialize_document(agent_id, document)


@app.put("/agents/{agent_id}/documents/{document_id}", response_model=KnowledgeDocumentResponse)
def update_document(
    agent_id: str,
    document_id: str,
    request: KnowledgeDocumentUpdateRequest,
) -> KnowledgeDocumentResponse:
    store = require_store()
    try:
        agent = store.require_agent(agent_id)
        document = next(item for item in agent["documents"] if item["id"] == document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' was not found.") from exc

    source_type = document.get("source_type")
    next_name = request.file_name.strip() if request.file_name else document["file_name"]
    if source_type == "pdf":
        if not next_name.lower().endswith(".pdf"):
            next_name = f"{next_name}.pdf"
        current_path = store.get_agent_pdf_dir(agent_id) / document["file_name"]
        target_path = store.get_agent_pdf_dir(agent_id) / Path(next_name).name
        if target_path.exists() and target_path != current_path:
            raise HTTPException(status_code=409, detail=f"A PDF named '{target_path.name}' already exists.")
        if current_path.exists() and current_path != target_path:
            current_path.rename(target_path)
        document = store.update_document(agent_id, document_id, file_name=target_path.name)
        refresh_agent_index(agent_id)
        return serialize_document(agent_id, document)
    if source_type == "website":
        document = store.update_document(agent_id, document_id, file_name=next_name)
        return serialize_document(agent_id, document)

    manual_service = ManualKnowledgeService(store.get_agent_snippet_dir(agent_id), store.get_agent_qa_dir(agent_id))
    if source_type == "text_snippet":
        existing = manual_service.list_text_snippets().get(document_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Text snippet '{document_id}' was not found.")
        content = request.content.strip() if request.content is not None else existing.content
        manual_service.save_text_snippet(document_id, next_name, content)
        document = store.update_document(agent_id, document_id, file_name=next_name)
        refresh_agent_index(agent_id)
        return serialize_document(agent_id, document)

    if source_type == "qa":
        existing = manual_service.list_qa().get(document_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Q&A item '{document_id}' was not found.")
        answer = request.answer.strip() if request.answer is not None else existing.answer
        manual_service.save_qa(document_id, next_name, answer)
        document = store.update_document(agent_id, document_id, file_name=next_name)
        refresh_agent_index(agent_id)
        return serialize_document(agent_id, document)

    document = store.update_document(agent_id, document_id, file_name=next_name)
    return serialize_document(agent_id, document)


@app.delete("/agents/{agent_id}/documents/{document_id}", status_code=204)
def delete_document(agent_id: str, document_id: str) -> None:
    store = require_store()
    try:
        document = store.delete_document(agent_id, document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # FIX: Resolve the doc_id BEFORE deleting the physical file, since for
    # PDFs the doc_id is the file name (still intact at this point).
    doc_id = resolve_doc_id(document)

    # Step 1 — delete the physical file / data from disk (unchanged logic)
    if document.get("source_type") == "pdf":
        pdf_path = store.get_agent_pdf_dir(agent_id) / document["file_name"]
        if pdf_path.exists():
            pdf_path.unlink()
    elif document.get("source_type") == "website":
        WebsiteService(store.get_agent_website_dir(agent_id)).delete_source(
            str(document.get("source_url") or "")
        )
    elif document.get("source_type") == "text_snippet":
        ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id), store.get_agent_qa_dir(agent_id)
        ).delete_text_snippet(document_id)
    elif document.get("source_type") == "qa":
        ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id), store.get_agent_qa_dir(agent_id)
        ).delete_qa(document_id)

    # FIX: Step 2 — surgically remove ONLY this document's chunks from Qdrant
    # using a payload filter on doc_id, instead of dropping and rebuilding
    # the entire collection.
    #
    # Old approach (broken):
    #   refresh_agent_index(agent_id)  ← drops entire collection, then
    #                                     re-embeds ALL remaining documents.
    #   Problems:
    #   • Non-atomic: queries see empty results during the rebuild window.
    #   • Qdrant local-client bug: delete+recreate with the same collection
    #     name can return stale cached vectors on the very next query.
    #   • Data loss risk: if OpenAI API fails mid-rebuild, all vectors are gone.
    #   • Slow: O(all documents) embedding API calls just to remove one.
    #
    # New approach (correct):
    #   pipeline.remove_document(doc_id)  ← single Qdrant filter-delete,
    #                                        O(1), atomic, no rebuild needed.
    try:
        pipeline = build_pipeline(agent_id)
        if doc_id:
            pipeline.remove_document(doc_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove document from knowledge base: {exc}",
        ) from exc


@app.get("/agents/{agent_id}/settings", response_model=AgentSettingsResponse)
def get_settings(agent_id: str) -> AgentSettingsResponse:
    store = require_store()
    try:
        settings = store.get_settings(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentSettingsResponse(**settings)


@app.put("/agents/{agent_id}/settings", response_model=AgentSettingsResponse)
def update_settings(agent_id: str, request: AgentSettingsUpdateRequest) -> AgentSettingsResponse:
    store = require_store()
    try:
        settings = store.update_settings(agent_id, request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentSettingsResponse(**settings)


@app.get("/agents/{agent_id}/conversations", response_model=list[ConversationSummaryResponse])
def list_conversations(agent_id: str) -> list[ConversationSummaryResponse]:
    store = require_store()
    try:
        conversations = store.list_conversations(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [ConversationSummaryResponse(**conversation) for conversation in conversations]


@app.get(
    "/agents/{agent_id}/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(agent_id: str, conversation_id: str) -> ConversationDetailResponse:
    store = require_store()
    try:
        conversation = store.get_conversation(agent_id, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ConversationDetailResponse(**conversation)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    store = require_store()
    try:
        store.require_agent(request.agent_id)
        settings = store.get_settings(request.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        pipeline = build_pipeline(request.agent_id)
        answer = pipeline.answer_question(
            request.question.strip(),
            system_prompt=settings["system_prompt"],
            temperature=settings["temperature"],
        )
        conversation = store.append_conversation_messages(
            agent_id=request.agent_id,
            user_message=request.question.strip(),
            assistant_message=answer,
            conversation_id=request.conversation_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    return ChatResponse(answer=answer, conversation_id=conversation["id"])