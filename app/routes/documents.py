from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.models.document import DocumentResponse, DocumentUpdateRequest
from app.models.user import UserRole
from app.rag_pipeline import RAGPipeline
from app.services.admin_store_mongo import AdminStoreMongo
from app.services.chunk_store import ChunkStore
from app.website_service import WebsiteService
from app.manual_knowledge_service import ManualKnowledgeService
from app.pdf_service import PDFService

router = APIRouter(prefix="/agents/{agent_id}", tags=["documents"])

SUPPORTED_FILE_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".txt": "txt",
}
FILE_SOURCE_TYPES = set(SUPPORTED_FILE_TYPES.values())


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _get_chunk_store() -> ChunkStore:
    from app.main import chunk_store
    return chunk_store


def _build_pipeline(store: AdminStoreMongo, agent_id: str) -> RAGPipeline:
    from app.main import agent_collection_name
    return RAGPipeline(
        pdf_directory=store.get_agent_pdf_dir(agent_id),
        website_directory=store.get_agent_website_dir(agent_id),
        snippets_directory=store.get_agent_snippet_dir(agent_id),
        qa_directory=store.get_agent_qa_dir(agent_id),
        collection_name=agent_collection_name(agent_id),
    )


async def _resolve_tenant(
    agent_id: str,
    user: CurrentUser,
    store: AdminStoreMongo,
    *,
    allow_shared: bool = False,
) -> str:
    """
    Return tenant_id for the agent.

    Fast path: for regular users the tenant_id is already in the JWT — skip the DB.
    Slow path: super_admin can access any tenant, so we still need the DB lookup.
    """
    from app.models.user import UserRole
    if user.role != UserRole.SUPER_ADMIN:
        # JWT already carries the correct tenant_id for this user.
        # Just verify the agent belongs to them without a full document fetch.
        if allow_shared:
            agent_doc = await store.get_accessible_agent(agent_id, user.tenant_id, user.id)
            if not agent_doc:
                raise KeyError(f"Agent '{agent_id}' not found.")
            return agent_doc["tenant_id"]

        exists = await store.db.agents.find_one(
            {"_id": agent_id, "tenant_id": user.tenant_id},
            {"_id": 1},   # projection — only _id, minimal I/O
        )
        if not exists:
            raise KeyError(f"Agent '{agent_id}' not found.")
        return user.tenant_id

    # Super admin: fetch the real tenant_id from the agent document
    agent_doc = await store.db.agents.find_one({"_id": agent_id}, {"tenant_id": 1})
    if not agent_doc:
        raise KeyError(f"Agent '{agent_id}' not found.")
    return agent_doc["tenant_id"]


async def _require_conversation_read_access(user: CurrentUser) -> None:
    if await user.has_permission("conversations", "read") or await user.has_permission("chats", "read"):
        return
    raise HTTPException(status_code=403, detail="Your role does not have permission to view chats.")


class WebsitePageResponse(BaseModel):
    index: int
    url: str
    title: str
    text: str


class WebsitePageCreateRequest(BaseModel):
    url: str = Field(..., min_length=1)
    title: str | None = Field(default=None)
    text: str | None = Field(default=None)


class WebsitePageUpdateRequest(BaseModel):
    url: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


def _format_website_page_for_ingestion(page: WebsitePageResponse) -> str:
    parts = [page.title.strip(), page.url.strip(), page.text.strip()]
    return "\n\n".join(part for part in parts if part)


async def _get_website_document(
    agent_id: str,
    document_id: str,
    user: CurrentUser,
    store: AdminStoreMongo,
    *,
    allow_shared: bool = False,
) -> tuple[str, dict]:
    tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=allow_shared)
    document = await store.get_document(document_id, agent_id, tenant_id)
    if document is None:
        raise KeyError(f"Document '{document_id}' not found.")
    if document.get("source_type") != "website":
        raise ValueError("This document is not a website source.")
    return tenant_id, document


async def _reindex_website_document(
    *,
    agent_id: str,
    tenant_id: str,
    document: dict,
    store: AdminStoreMongo,
    cs: ChunkStore,
) -> dict | None:
    source_url = str(document.get("source_url") or "").strip()
    website_service = WebsiteService(store.get_agent_website_dir(agent_id))
    pages = [
        WebsitePageResponse(index=page.index, url=page.url, title=page.title, text=page.text)
        for page in website_service.list_source_pages(source_url)
    ]

    pipeline = _build_pipeline(store, agent_id)
    chunk_ids = await cs.get_chunk_ids_by_document(document["id"])
    if chunk_ids:
        pipeline.remove_document(document["id"], chunk_ids=chunk_ids)
        await cs.delete_chunks_by_document(document["id"])

    if not pages:
        website_service.delete_source(source_url)
        await store.delete_document(document["id"], agent_id, tenant_id)
        return None

    combined_text = "\n\n".join(
        _format_website_page_for_ingestion(page)
        for page in pages
        if page.text.strip()
    ).strip()
    if not combined_text:
        raise ValueError("Website pages must contain readable text.")

    await pipeline.ingest_single_document(
        chunk_store=cs,
        tenant_id=tenant_id,
        agent_id=agent_id,
        document_id=document["id"],
        source_type="website",
        source_name=source_url,
        text=combined_text,
    )

    display_name = pages[0].title.strip() or source_url
    return await store.update_document(
        document["id"],
        agent_id,
        tenant_id,
        file_name=display_name,
        source_url=source_url,
    )


# ── List documents ────────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    agent_id: str,
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("knowledge", "read")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response.headers["Cache-Control"] = "private, max-age=30, stale-while-revalidate=60"
    docs = await store.list_documents(agent_id, tenant_id)
    website_sources = WebsiteService(store.get_agent_website_dir(agent_id)).list_sources()

    # Load manual knowledge records from disk so we can enrich text_snippet /
    # qa documents whose content is NOT stored in MongoDB.
    _manual_service = ManualKnowledgeService(
        snippets_directory=store.get_agent_snippet_dir(agent_id),
        qa_directory=store.get_agent_qa_dir(agent_id),
    )
    snippets_on_disk = _manual_service.list_text_snippets()
    qa_on_disk = _manual_service.list_qa()

    enriched_docs: list[DocumentResponse] = []
    for doc in docs:
        payload = {**doc, "uploaded_at": doc.get("uploaded_at")}
        if doc.get("source_type") == "website":
            summary = website_sources.get(str(doc.get("source_url") or "").strip())
            if summary:
                # Disk data available (local dev or fresh Cloud Run instance)
                payload["page_count"] = summary.page_count
                payload["page_urls"] = summary.page_urls
            else:
                # Disk wiped (Cloud Run restart) — use MongoDB persisted data
                mongo_urls = doc.get("page_urls") or []
                if mongo_urls:
                    payload["page_count"] = len(mongo_urls)
                    payload["page_urls"] = mongo_urls
        elif doc.get("source_type") == "text_snippet":
            # content is stored on disk; fall back to MongoDB if already
            # persisted there (post-fix documents will have it in Mongo too).
            if not payload.get("content"):
                record = snippets_on_disk.get(str(doc.get("id", "")))
                if record:
                    payload["content"] = record.content
        elif doc.get("source_type") == "qa":
            # answer is stored on disk; same fallback strategy as above.
            if not payload.get("answer"):
                record = qa_on_disk.get(str(doc.get("id", "")))
                if record:
                    payload["answer"] = record.answer
        enriched_docs.append(DocumentResponse(**payload))
    return enriched_docs


# ── Upload PDF ────────────────────────────────────────────────────────────────

@router.post("/upload-pdf", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    agent_id: str,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_name = (file.filename or "").strip()
    suffix = Path(file_name).suffix.lower()
    source_type = SUPPORTED_FILE_TYPES.get(suffix)
    if not source_type:
        raise HTTPException(status_code=400, detail="Supported file types: PDF, DOCX, PPTX, TXT.")

    pdf_dir = store.get_agent_pdf_dir(agent_id)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / Path(file_name).name
    target.write_bytes(await file.read())

    doc = await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, target.name, source_type, status="indexing"
    )

    try:
        file_service = PDFService(pdf_directory=pdf_dir)
        extracted_text = file_service.extract_text(target)

        if extracted_text.strip():
            pipeline = _build_pipeline(store, agent_id)
            await pipeline.ingest_single_document(
                chunk_store=cs,
                tenant_id=tenant_id,
                agent_id=agent_id,
                document_id=doc["id"],
                source_type=source_type,
                source_name=target.name,
                text=extracted_text,
            )
        else:
            raise ValueError(f"No text could be extracted from the {source_type.upper()} file.")

    except Exception as exc:
        await store.mark_document_uploaded(
            agent_id, tenant_id, user.id, target.name, source_type, status="failed"
        )
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, target.name, source_type, status="indexed"
    )
    doc["status"] = "indexed"
    doc["source_type"] = source_type
    return DocumentResponse(**doc)


# ── Crawl website ─────────────────────────────────────────────────────────────

@router.post("/crawl-website", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def crawl_website(
    agent_id: str,
    url: str = Form(...),
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    doc = await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, url, "website",
        status="indexing", source_url=url,
    )

    try:
        website_dir = store.get_agent_website_dir(agent_id)
        website_dir.mkdir(parents=True, exist_ok=True)
        website_service = WebsiteService(website_directory=website_dir)
        crawl_result = website_service.crawl(url)
        website_service.save_crawl(crawl_result)

        crawled_docs = crawl_result.pages
        valid_pages = [p for p in crawled_docs if p.text.strip()]
        if not valid_pages:
            raise ValueError("No text could be extracted from the website.")

        pipeline = _build_pipeline(store, agent_id)

        # Ingest each crawled page as a separate document so every page
        # gets its own focused embeddings — fixes partial product retrieval.
        for page in valid_pages:
            page_text = "\n\n".join(filter(None, [
                page.title.strip() if page.title else "",
                page.url.strip(),
                page.text.strip(),
            ]))
            page_doc = await store.mark_document_uploaded(
                agent_id, tenant_id, user.id,
                page.title or page.url, "website",
                status="indexing", source_url=page.url,
            )
            await pipeline.ingest_single_document(
                chunk_store=cs,
                tenant_id=tenant_id,
                agent_id=agent_id,
                document_id=page_doc["id"],
                source_type="website",
                source_name=page.url,
                text=page_text,
            )
            await store.mark_document_uploaded(
                agent_id, tenant_id, user.id,
                page.title or page.url, "website",
                status="indexed", source_url=page.url,
            )

    except Exception as exc:
        await store.mark_document_uploaded(
            agent_id, tenant_id, user.id, url, "website", status="failed"
        )
        raise HTTPException(status_code=500, detail=f"Crawl failed: {exc}") from exc

    # Mark the root crawl entry (the submitted URL) as indexed
    await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, url, "website",
        status="indexed", source_url=url,
    )
    doc["status"] = "indexed"
    return DocumentResponse(**doc)


# ── Add text snippet ──────────────────────────────────────────────────────────

class TextSnippetRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


@router.post("/text-snippets", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def add_text_snippet(
    agent_id: str,
    body: TextSnippetRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    title = body.title
    content = body.content
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snippet_dir = store.get_agent_snippet_dir(agent_id)
    snippet_dir.mkdir(parents=True, exist_ok=True)

    doc = await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, title, "text_snippet", status="indexing"
    )

    try:
        manual_service = ManualKnowledgeService(
            snippets_directory=snippet_dir,
            qa_directory=store.get_agent_qa_dir(agent_id),
        )
        manual_service.save_text_snippet(doc["id"], title, content)

        pipeline = _build_pipeline(store, agent_id)
        await pipeline.ingest_single_document(
            chunk_store=cs,
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=doc["id"],
            source_type="text_snippet",
            source_name=title,
            text=f"{title}\n\n{content}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    # Persist content to MongoDB so it survives Cloud Run restarts and is
    # returned directly by list_documents without needing disk enrichment.
    updated = await store.update_document(
        doc["id"], agent_id, tenant_id,
        file_name=title,
        content=content.strip(),
        status="indexed",
    )
    return DocumentResponse(**updated)


# ── Add Q&A ───────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


@router.post("/qa", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def add_qa(
    agent_id: str,
    body: QARequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    question = body.question
    answer = body.answer
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    qa_dir = store.get_agent_qa_dir(agent_id)
    qa_dir.mkdir(parents=True, exist_ok=True)

    doc = await store.mark_document_uploaded(
        agent_id, tenant_id, user.id, question[:80], "qa", status="indexing"
    )

    try:
        manual_service = ManualKnowledgeService(
            snippets_directory=store.get_agent_snippet_dir(agent_id),
            qa_directory=qa_dir,
        )
        manual_service.save_qa(doc["id"], question, answer)

        pipeline = _build_pipeline(store, agent_id)
        await pipeline.ingest_single_document(
            chunk_store=cs,
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=doc["id"],
            source_type="qa",
            source_name=question[:80],
            text=f"Q: {question}\nA: {answer}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    # Persist answer to MongoDB so it survives Cloud Run restarts.
    updated = await store.update_document(
        doc["id"], agent_id, tenant_id,
        file_name=question[:80],
        answer=answer.strip(),
        status="indexed",
    )
    return DocumentResponse(**updated)


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    agent_id: str,
    document_id: str,
    body: DocumentUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
        doc = await store.get_document(document_id, agent_id, tenant_id)
        if doc is None:
            raise KeyError(f"Document '{document_id}' not found.")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_type = str(doc.get("source_type") or "")
    next_name = body.file_name.strip() if body.file_name else str(doc.get("file_name") or "")

    try:
        if source_type == "text_snippet":
            next_content = (body.content or "").strip()
            if not next_name or not next_content:
                raise ValueError("Text snippets require both title and content.")
            ManualKnowledgeService(
                snippets_directory=store.get_agent_snippet_dir(agent_id),
                qa_directory=store.get_agent_qa_dir(agent_id),
            ).save_text_snippet(document_id, next_name, next_content)

            chunk_ids = await cs.get_chunk_ids_by_document(document_id)
            if chunk_ids:
                _build_pipeline(store, agent_id).remove_document(document_id, chunk_ids=chunk_ids)
                await cs.delete_chunks_by_document(document_id)

            await _build_pipeline(store, agent_id).ingest_single_document(
                chunk_store=cs,
                tenant_id=tenant_id,
                agent_id=agent_id,
                document_id=document_id,
                source_type="text_snippet",
                source_name=next_name,
                text=f"{next_name}\n\n{next_content}",
            )
            updated = await store.update_document(
                document_id, agent_id, tenant_id,
                file_name=next_name,
                content=next_content,
                status="indexed",
            )
            return DocumentResponse(**updated)

        if source_type == "qa":
            next_answer = (body.answer or "").strip()
            if not next_name or not next_answer:
                raise ValueError("Q&A entries require both question and answer.")
            ManualKnowledgeService(
                snippets_directory=store.get_agent_snippet_dir(agent_id),
                qa_directory=store.get_agent_qa_dir(agent_id),
            ).save_qa(document_id, next_name, next_answer)

            chunk_ids = await cs.get_chunk_ids_by_document(document_id)
            if chunk_ids:
                _build_pipeline(store, agent_id).remove_document(document_id, chunk_ids=chunk_ids)
                await cs.delete_chunks_by_document(document_id)

            await _build_pipeline(store, agent_id).ingest_single_document(
                chunk_store=cs,
                tenant_id=tenant_id,
                agent_id=agent_id,
                document_id=document_id,
                source_type="qa",
                source_name=next_name,
                text=f"Q: {next_name}\nA: {next_answer}",
            )
            updated = await store.update_document(
                document_id, agent_id, tenant_id,
                file_name=next_name,
                answer=next_answer,
                status="indexed",
            )
            return DocumentResponse(**updated)

        if source_type in FILE_SOURCE_TYPES:
            if not next_name:
                raise ValueError("File name is required.")
            current_path = store.get_agent_pdf_dir(agent_id) / str(doc.get("file_name") or "")
            expected_suffix = f".{source_type}"
            if not next_name.lower().endswith(expected_suffix):
                next_name = f"{next_name}{expected_suffix}"
            target_path = store.get_agent_pdf_dir(agent_id) / Path(next_name).name
            if current_path.exists() and current_path != target_path:
                current_path.rename(target_path)

            file_service = PDFService(pdf_directory=store.get_agent_pdf_dir(agent_id))
            extracted_text = file_service.extract_text(target_path)
            if not extracted_text.strip():
                raise ValueError(f"No text could be extracted from the {source_type.upper()} file.")

            chunk_ids = await cs.get_chunk_ids_by_document(document_id)
            if chunk_ids:
                _build_pipeline(store, agent_id).remove_document(document_id, chunk_ids=chunk_ids)
                await cs.delete_chunks_by_document(document_id)

            await _build_pipeline(store, agent_id).ingest_single_document(
                chunk_store=cs,
                tenant_id=tenant_id,
                agent_id=agent_id,
                document_id=document_id,
                source_type=source_type,
                source_name=target_path.name,
                text=extracted_text,
            )
            updated = await store.update_document(document_id, agent_id, tenant_id, file_name=target_path.name)
            return DocumentResponse(**updated)

        updated = await store.update_document(document_id, agent_id, tenant_id, file_name=next_name)
        return DocumentResponse(**updated)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Update failed: {exc}") from exc


@router.get("/documents/{document_id}/website-pages", response_model=list[WebsitePageResponse])
async def list_website_pages(
    agent_id: str,
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("knowledge", "read")
    try:
        _, document = await _get_website_document(agent_id, document_id, user, store, allow_shared=True)
        source_url = str(document.get("source_url") or "")
        website_service = WebsiteService(store.get_agent_website_dir(agent_id))

        # Try reading from local disk first (fast path — works locally and
        # immediately after a manual crawl on Cloud Run)
        disk_pages = []
        try:
            disk_pages = website_service.list_source_pages(source_url)
        except (FileNotFoundError, ValueError):
            pass

        if disk_pages:
            return [
                WebsitePageResponse(index=p.index, url=p.url, title=p.title, text=p.text)
                for p in disk_pages
            ]

        # Fallback: read page_urls persisted in MongoDB.
        # This is the only data that survives Cloud Run restarts / ephemeral
        # disk resets — the scheduled recrawl saves page_urls to MongoDB
        # after every successful crawl.
        mongo_page_urls: list[str] = document.get("page_urls") or []
        if mongo_page_urls:
            return [
                WebsitePageResponse(index=i, url=url, title=url, text="")
                for i, url in enumerate(mongo_page_urls)
            ]

        # Nothing found anywhere
        return []

    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/{document_id}/website-pages", response_model=WebsitePageResponse, status_code=status.HTTP_201_CREATED)
async def create_website_page(
    agent_id: str,
    document_id: str,
    body: WebsitePageCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    try:
        tenant_id, document = await _get_website_document(agent_id, document_id, user, store)
        website_service = WebsiteService(store.get_agent_website_dir(agent_id))
        crawled_page = website_service.crawl_single_page(body.url)
        page = website_service.create_source_page(
            str(document.get("source_url") or ""),
            url=crawled_page.url,
            title=crawled_page.title,
            text=crawled_page.text,
        )
        await _reindex_website_document(
            agent_id=agent_id,
            tenant_id=tenant_id,
            document=document,
            store=store,
            cs=cs,
        )
        return WebsitePageResponse(index=page.index, url=page.url, title=page.title, text=page.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/documents/{document_id}/website-pages/{page_index}", response_model=WebsitePageResponse)
async def update_website_page(
    agent_id: str,
    document_id: str,
    page_index: int,
    body: WebsitePageUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "write")
    try:
        tenant_id, document = await _get_website_document(agent_id, document_id, user, store)
        website_service = WebsiteService(store.get_agent_website_dir(agent_id))
        page = website_service.update_source_page(
            str(document.get("source_url") or ""),
            page_index,
            url=body.url,
            title=body.title,
            text=body.text,
        )
        await _reindex_website_document(
            agent_id=agent_id,
            tenant_id=tenant_id,
            document=document,
            store=store,
            cs=cs,
        )
        return WebsitePageResponse(index=page.index, url=page.url, title=page.title, text=page.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/documents/{document_id}/website-pages/{page_index}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_website_page(
    agent_id: str,
    document_id: str,
    page_index: int,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "delete")
    try:
        tenant_id, document = await _get_website_document(agent_id, document_id, user, store)
        website_service = WebsiteService(store.get_agent_website_dir(agent_id))
        website_service.delete_source_page(str(document.get("source_url") or ""), page_index)
        await _reindex_website_document(
            agent_id=agent_id,
            tenant_id=tenant_id,
            document=document,
            store=store,
            cs=cs,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Delete document ───────────────────────────────────────────────────────────

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    agent_id: str,
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "delete")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
        doc = await store.delete_document(document_id, agent_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_type = doc.get("source_type")

    chunk_ids = await cs.get_chunk_ids_by_document(document_id)
    if chunk_ids:
        try:
            pipeline = _build_pipeline(store, agent_id)
            pipeline.remove_document(document_id, chunk_ids=chunk_ids)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to remove from index: {exc}") from exc

    await cs.delete_chunks_by_document(document_id)

    if source_type in FILE_SOURCE_TYPES:
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


# ── Rebuild index ─────────────────────────────────────────────────────────────

@router.post("/rebuild-index")
async def rebuild_index(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    await user.require_permission("knowledge", "manage")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    pipeline = _build_pipeline(store, agent_id)
    count = await pipeline.rebuild_index_from_mongo(cs, agent_id)
    return {"message": f"Rebuilt index with {count} chunks", "chunks": count}


# ── Conversations ─────────────────────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(
    agent_id: str,
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await _require_conversation_read_access(user)
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response.headers["Cache-Control"] = "private, max-age=30, stale-while-revalidate=60"
    return await store.list_conversations(agent_id, tenant_id)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    agent_id: str,
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await _require_conversation_read_access(user)
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=True)
        conv = await store.get_conversation(conversation_id, agent_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return conv


# ── Conversation summary ──────────────────────────────────────────────────────

@router.get("/conversations/{conversation_id}/summary")
async def get_conversation_summary(
    agent_id: str,
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await _require_conversation_read_access(user)
    import os
    import asyncio
    from openai import OpenAI

    try:
        tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    existing_summary, is_stale = await store.get_conversation_summary(
        conversation_id, agent_id, tenant_id
    )

    if existing_summary and not is_stale:
        return {"summary": existing_summary, "cached": True}

    try:
        conv = await store.get_conversation(conversation_id, agent_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    messages = conv.get("messages", [])
    if not messages:
        return {"summary": "No messages in this conversation yet.", "cached": False}

    transcript = ""
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Bot"
        text = msg.get("content", "").strip()
        if text:
            transcript += f"{role}: {text}\n"

    if not transcript.strip():
        return {"summary": "No content to summarize.", "cached": False}

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"summary": "Summary unavailable — OpenAI API key not configured.", "cached": False}

    try:
        client = OpenAI(api_key=api_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. "
                            "Given a chat transcript between a User and a Bot, "
                            "write a concise 2-3 sentence summary of what the user asked about "
                            "and what the bot answered. Write in third person. "
                            "Be specific about the topics discussed."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation:\n\n{transcript[:3000]}"
                    }
                ],
                temperature=0.3,
                max_tokens=150,
            )
        )
        summary = (response.choices[0].message.content or "Unable to generate summary.").strip()
        await store.save_conversation_summary(conversation_id, agent_id, tenant_id, summary)
        return {"summary": summary, "cached": False}

    except Exception as exc:
        return {"summary": f"Unable to generate summary: {str(exc)}", "cached": False}


# ── Agent analytics ───────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_agent_analytics(
    agent_id: str,
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("analytics", "read")
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store, allow_shared=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=120"

    import asyncio as _asyncio
    # Run all 3 independent DB queries in parallel
    convs, fallback_logs = await _asyncio.gather(
        store.list_conversations(agent_id, tenant_id),
        store.get_fallback_logs(agent_id, tenant_id, limit=200),
    )
    total_convs = len(convs)
    # Derive count from logs already fetched — avoids a 4th round-trip
    fallback_count = len(fallback_logs)

    total_messages = sum(c.get("message_count", 0) for c in convs)
    user_messages = total_messages // 2 if total_messages > 0 else 0
    fallback_rate = round((fallback_count / user_messages * 100), 1) if user_messages > 0 else 0

    from collections import Counter
    title_counts = Counter(c.get("title", "").strip() for c in convs if c.get("title"))
    starting_questions = [
        {"question": q, "count": n}
        for q, n in title_counts.most_common(10)
        if q
    ]

    unanswered_counter = Counter(
        log.get("question", "").strip() for log in fallback_logs
    )
    unanswered_questions = [
        {"question": q, "count": n, "last_asked": next(
            (l.get("created_at") for l in fallback_logs if l.get("question", "").strip() == q), None
        )}
        for q, n in unanswered_counter.most_common(20)
        if q
    ]

    return {
        "total_conversations": total_convs,
        "total_messages": total_messages,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "unanswered_questions": unanswered_questions,
        "starting_questions": starting_questions,
    }