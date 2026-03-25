from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.crawl_job_store import CrawlJobStore
from app.manual_knowledge_service import ManualKnowledgeService
from app.models.document import DocumentResponse
from app.rag_pipeline import RAGPipeline
from app.services.admin_store_mongo import AdminStoreMongo
from app.website_service import WebsiteService

router = APIRouter(tags=["documents"])

# In-memory crawl job store (same as original)
crawl_job_store = CrawlJobStore()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def agent_collection_name(agent_id: str) -> str:
    return f"knowledge_base_{agent_id.replace('-', '_')}"


def _build_pipeline(store: AdminStoreMongo, agent_id: str) -> RAGPipeline:
    return RAGPipeline(
        pdf_directory=store.get_agent_pdf_dir(agent_id),
        website_directory=store.get_agent_website_dir(agent_id),
        snippets_directory=store.get_agent_snippet_dir(agent_id),
        qa_directory=store.get_agent_qa_dir(agent_id),
        collection_name=agent_collection_name(agent_id),
    )


def normalize_website_url(url: str) -> str:
    normalized = url.strip()
    if normalized and not urlparse(normalized).scheme:
        normalized = f"https://{normalized}"
    return normalized


def _serialize_doc(doc: dict) -> DocumentResponse:
    return DocumentResponse(**{k: v for k, v in doc.items() if k in DocumentResponse.model_fields})


# ── Request models ────────────────────────────────────────────────────────────

class WebsiteCrawlRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


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


class TextSnippetCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class QACreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class DocumentUpdateRequest(BaseModel):
    file_name: str | None = Field(default=None, min_length=1)
    content: str | None = None
    answer: str | None = None


# ── Background crawl job ──────────────────────────────────────────────────────

def run_website_crawl_job(job_id: str, store: AdminStoreMongo) -> None:
    job = crawl_job_store.get(job_id)
    if job is None:
        return

    website_service = WebsiteService(store.get_agent_website_dir(job.agent_id))
    crawl_job_store.update(job_id, status="running", stage="crawling", message="Starting website crawl.")

    def on_progress(update: dict) -> None:
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
            message=f"Building embeddings for {len(crawl.pages)} pages.",
            discovered_pages=len(crawl.pages),
            indexed_pages=len(crawl.pages),
            current_url=None,
        )
        website_service.save_crawl(crawl)

        pipeline = _build_pipeline(store, job.agent_id)
        pipeline.ingest_documents(recreate=True)

        # Use asyncio to run the async upsert
        loop = asyncio.new_event_loop()
        doc = loop.run_until_complete(
            store.upsert_website_source(
                agent_id=job.agent_id,
                tenant_id=job.agent_id,  # will be overwritten below
                user_id="",
                display_name=crawl.display_name,
                source_url=crawl.source_url,
                status="indexed",
            )
        )
        loop.close()

        crawl_job_store.update(
            job_id,
            status="completed",
            stage="completed",
            message=f"Indexed {len(crawl.pages)} pages successfully.",
            discovered_pages=len(crawl.pages),
            indexed_pages=len(crawl.pages),
            document_id=doc.get("id"),
            document_name=doc.get("file_name"),
        )
    except Exception as exc:
        crawl_job_store.update(
            job_id, status="failed", stage="failed",
            error=str(exc), message="Website crawl failed.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/agents/{agent_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        await store.require_agent(agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    docs = await store.list_documents(agent_id, user.tenant_id)
    return [_serialize_doc(d) for d in docs]


@router.post("/agents/{agent_id}/upload-pdf", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    agent_id: str,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        await store.require_agent(agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_name = (file.filename or "").strip()
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_dir = store.get_agent_pdf_dir(agent_id)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / Path(file_name).name
    target.write_bytes(await file.read())

    doc = await store.mark_document_uploaded(
        agent_id, user.tenant_id, user.id, target.name, "pdf", status="indexing"
    )

    try:
        loop = asyncio.get_event_loop()
        def _ingest():
            pipeline = _build_pipeline(store, agent_id)
            pipeline.ingest_documents(recreate=True)
        await loop.run_in_executor(None, _ingest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    await store.mark_document_uploaded(
        agent_id, user.tenant_id, user.id, target.name, "pdf", status="indexed"
    )
    doc["status"] = "indexed"
    return _serialize_doc(doc)


@router.post("/crawl-website", response_model=CrawlJobResponse, status_code=202)
async def crawl_website(
    request: WebsiteCrawlRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        await store.require_agent(request.agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_url = normalize_website_url(request.url)
    if not source_url:
        raise HTTPException(status_code=400, detail="Website URL is required.")

    await store.upsert_website_source(
        agent_id=request.agent_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
        display_name=source_url,
        source_url=source_url,
        status="indexing",
    )

    job = crawl_job_store.create(agent_id=request.agent_id, source_url=source_url)
    background_tasks.add_task(run_website_crawl_job, job.id, store)
    return CrawlJobResponse(**job.to_dict())


@router.get("/crawl-website/{job_id}", response_model=CrawlJobResponse)
async def get_crawl_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    job = crawl_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Crawl job '{job_id}' not found.")
    return CrawlJobResponse(**job.to_dict())


@router.post("/agents/{agent_id}/text-snippets", response_model=DocumentResponse, status_code=201)
async def create_text_snippet(
    agent_id: str,
    request: TextSnippetCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        await store.require_agent(agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    doc = await store.mark_document_uploaded(
        agent_id, user.tenant_id, user.id,
        request.title.strip(), "text_snippet", status="indexed",
    )
    manual_service = ManualKnowledgeService(
        store.get_agent_snippet_dir(agent_id),
        store.get_agent_qa_dir(agent_id),
    )
    try:
        manual_service.save_text_snippet(doc["id"], request.title, request.content)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _build_pipeline(store, agent_id).ingest_documents(recreate=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save snippet: {exc}") from exc

    return _serialize_doc(doc)


@router.post("/agents/{agent_id}/qa", response_model=DocumentResponse, status_code=201)
async def create_qa(
    agent_id: str,
    request: QACreateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        await store.require_agent(agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    doc = await store.mark_document_uploaded(
        agent_id, user.tenant_id, user.id,
        request.question.strip(), "qa", status="indexed",
    )
    manual_service = ManualKnowledgeService(
        store.get_agent_snippet_dir(agent_id),
        store.get_agent_qa_dir(agent_id),
    )
    try:
        manual_service.save_qa(doc["id"], request.question, request.answer)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _build_pipeline(store, agent_id).ingest_documents(recreate=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save Q&A: {exc}") from exc

    return _serialize_doc(doc)


@router.put("/agents/{agent_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    agent_id: str,
    document_id: str,
    request: DocumentUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    doc = await store.get_document(document_id, agent_id, user.tenant_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    source_type = doc.get("source_type")
    next_name = request.file_name.strip() if request.file_name else doc["file_name"]

    if source_type == "text_snippet":
        manual_service = ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id),
            store.get_agent_qa_dir(agent_id),
        )
        content = request.content.strip() if request.content else (doc.get("content") or "")
        manual_service.save_text_snippet(document_id, next_name, content)

    elif source_type == "qa":
        manual_service = ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id),
            store.get_agent_qa_dir(agent_id),
        )
        answer = request.answer.strip() if request.answer else (doc.get("answer") or "")
        manual_service.save_qa(document_id, next_name, answer)

    elif source_type == "pdf":
        current_path = store.get_agent_pdf_dir(agent_id) / doc["file_name"]
        if not next_name.lower().endswith(".pdf"):
            next_name = f"{next_name}.pdf"
        target_path = store.get_agent_pdf_dir(agent_id) / Path(next_name).name
        if current_path.exists() and current_path != target_path:
            current_path.rename(target_path)

    updated = await store.update_document(document_id, agent_id, user.tenant_id, file_name=next_name)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _build_pipeline(store, agent_id).ingest_documents(recreate=True))

    return _serialize_doc(updated)


@router.delete("/agents/{agent_id}/documents/{document_id}", status_code=204)
async def delete_document(
    agent_id: str,
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        doc = await store.delete_document(document_id, agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_type = doc.get("source_type")

    if source_type == "pdf":
        pdf_path = store.get_agent_pdf_dir(agent_id) / doc["file_name"]
        pdf_path.unlink(missing_ok=True)
    elif source_type == "website":
        WebsiteService(store.get_agent_website_dir(agent_id)).delete_source(
            str(doc.get("source_url") or "")
        )
    elif source_type == "text_snippet":
        ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id),
            store.get_agent_qa_dir(agent_id),
        ).delete_text_snippet(document_id)
    elif source_type == "qa":
        ManualKnowledgeService(
            store.get_agent_snippet_dir(agent_id),
            store.get_agent_qa_dir(agent_id),
        ).delete_qa(document_id)

    try:
        pipeline = _build_pipeline(store, agent_id)
        doc_id = doc.get("source_url") if source_type == "website" else doc.get("file_name", document_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pipeline.remove_document(doc_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove from index: {exc}") from exc
