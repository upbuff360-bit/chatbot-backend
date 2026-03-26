from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

from app.db.connection import connect, create_indexes, disconnect, get_database
from app.core.dependencies import CurrentUser, get_current_user
from app.crawl_job_store import CrawlJobStore
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo
from app.services.chunk_store import ChunkStore
from app.routes import auth, agents, documents, chat, dashboard, roles, users, plans, billing

# Global instances — injected into routes via dependency functions
store: AdminStoreMongo | None = None
chunk_store: ChunkStore | None = None


def agent_collection_name(agent_id: str) -> str:
    return f"knowledge_base_{agent_id.replace('-', '_')}"


async def _startup_index_agents(s: AdminStoreMongo, cs: ChunkStore) -> None:
    from app.rag_pipeline import RAGPipeline

    all_agents = await s.list_all_agents()
    if not all_agents:
        return

    print(f"Startup: indexing {len(all_agents)} agent(s)...")

    # Phase 1 — fire all MongoDB chunk-existence checks in parallel
    mongo_flags = await asyncio.gather(
        *[cs.has_chunks_for_agent(agent["id"]) for agent in all_agents]
    )

    # Phase 2 — process each agent; Qdrant client calls are sync, keep serial
    for agent, has_mongo_chunks in zip(all_agents, mongo_flags):
        agent_id = agent["id"]
        collection = agent_collection_name(agent_id)

        pipeline = RAGPipeline(
            pdf_directory=s.get_agent_pdf_dir(agent_id),
            website_directory=s.get_agent_website_dir(agent_id),
            snippets_directory=s.get_agent_snippet_dir(agent_id),
            qa_directory=s.get_agent_qa_dir(agent_id),
            collection_name=collection,
        )

        qdrant_exists = pipeline.vector_store.collection_exists()

        # Check how many points are in Qdrant (0 means empty/missing)
        qdrant_count = 0
        if qdrant_exists:
            try:
                info = pipeline.vector_store.client.get_collection(collection)
                qdrant_count = info.points_count or 0
            except Exception:
                qdrant_count = 0

        if has_mongo_chunks and qdrant_count > 0:
            # Both MongoDB and Qdrant have data — skip completely
            chunk_count = await cs.count_chunks_by_agent(agent_id)
            print(f"  SKIP '{agent['name']}' — {chunk_count} mongo chunks, {qdrant_count} qdrant points (already indexed)")

        elif has_mongo_chunks and qdrant_count == 0:
            # MongoDB has chunks but Qdrant is empty — rebuild from MongoDB (no re-upload)
            print(f"  REBUILD '{agent['name']}' from MongoDB...")
            count = await pipeline.rebuild_index_from_mongo(cs, agent_id)
            print(f"  OK '{agent['name']}' — {count} chunks re-indexed from MongoDB")

        elif not has_mongo_chunks and qdrant_count > 0:
            # Qdrant has data but MongoDB doesn't — legacy agent, skip to avoid re-indexing on every restart
            print(f"  SKIP '{agent['name']}' — {qdrant_count} qdrant points (legacy, no MongoDB chunks)")

        else:
            # Nothing anywhere — first time, index from disk
            stats = pipeline.ingest_documents(recreate=True)
            print(f"  OK '{agent['name']}' — {stats.chunks_indexed} chunks indexed from disk")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global store, chunk_store

    await connect()
    await create_indexes()

    db = get_database()
    store = AdminStoreMongo(db=db, agents_root=os.getenv("AGENTS_DATA_ROOT", "./data/agents"))
    chunk_store = ChunkStore(db=db)

    # Ensure MongoDB chunk indexes exist
    await chunk_store.ensure_indexes()

    # Backfill display_id for existing agents (one-time, safe to run multiple times)
    await store.backfill_agent_display_ids()

    # Seed default permissions and roles
    await store.seed_default_permissions()
    await store.seed_default_roles()

    # Smart startup indexing
    await _startup_index_agents(store, chunk_store)

    yield

    await disconnect()


app = FastAPI(
    title="Chatbot SaaS API",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS
_origins = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if o.strip()
]
if "*" in _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routes
app.include_router(auth.router)
app.include_router(agents.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(dashboard.router)
app.include_router(roles.router)
app.include_router(users.router)
app.include_router(plans.router)
app.include_router(billing.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Crawl website ──────────────────────────────────────────────────────────────
from pydantic import BaseModel as _BaseModel


class _CrawlRequest(_BaseModel):
    agent_id: str
    url: str


class _SinglePageCrawlRequest(_BaseModel):
    agent_id: str
    document_id: str
    url: str


class _CrawlJobResponse(_BaseModel):
    id: str
    agent_id: str
    source_url: str
    status: str
    stage: str
    message: str | None = None
    discovered_pages: int = 0
    indexed_pages: int = 0
    current_url: str | None = None
    error: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    source_type: str = "website"


crawl_job_store = CrawlJobStore()


def _normalize_url(url: str) -> str:
    from app.website_service import WebsiteService

    return WebsiteService._normalize_url(url)


async def _run_crawl(job_id: str, agent_id: str, tenant_id: str, user_id: str, url: str) -> None:
    global store, chunk_store
    from app.rag_pipeline import RAGPipeline
    from app.website_service import WebsiteService

    if store is None or chunk_store is None:
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error="Crawl services are not ready.",
            message="Crawl services are not ready.",
        )
        return

    crawl_job_store.update(
        job_id,
        status="running",
        stage="crawling",
        current_url=url,
        message="Starting website crawl.",
        error=None,
    )

    try:
        loop = asyncio.get_running_loop()
        website_dir = store.get_agent_website_dir(agent_id)
        website_dir.mkdir(parents=True, exist_ok=True)
        website_service = WebsiteService(website_directory=website_dir)

        def on_progress(update: dict[str, int | str | None]) -> None:
            crawl_job_store.update(
                job_id,
                stage=str(update.get("stage") or "crawling"),
                discovered_pages=int(update.get("discovered_pages") or 0),
                indexed_pages=int(update.get("indexed_pages") or 0),
                current_url=str(update.get("current_url")) if update.get("current_url") else None,
                message=str(update.get("message")) if update.get("message") else None,
            )

        crawl_result = await loop.run_in_executor(
            None,
            lambda: website_service.crawl(url, progress_callback=on_progress),
        )
        await loop.run_in_executor(None, lambda: website_service.save_crawl(crawl_result))

        stored_pages = website_service.list_source_pages(url)

        combined_text = "\n\n".join(page.text for page in stored_pages if page.text.strip())
        if not combined_text.strip():
            raise ValueError("No text extracted from website.")

        doc = await store.upsert_website_source(
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            display_name=stored_pages[0].title if stored_pages else (crawl_result.display_name or url),
            source_url=url,
            status="indexing",
        )
        crawl_job_store.update(
            job_id,
            stage="indexing",
            message=f"Indexing {len(stored_pages)} page(s)...",
            discovered_pages=len(stored_pages),
            indexed_pages=len(stored_pages),
            current_url=None,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

        pipeline = RAGPipeline(
            pdf_directory=store.get_agent_pdf_dir(agent_id),
            website_directory=website_dir,
            snippets_directory=store.get_agent_snippet_dir(agent_id),
            qa_directory=store.get_agent_qa_dir(agent_id),
            collection_name=agent_collection_name(agent_id),
        )
        chunk_ids = await chunk_store.get_chunk_ids_by_document(doc["id"])
        if chunk_ids:
            pipeline.remove_document(doc["id"], chunk_ids=chunk_ids)
            await chunk_store.delete_chunks_by_document(doc["id"])
        await pipeline.ingest_single_document(
            chunk_store=chunk_store,
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=doc["id"],
            source_type="website",
            source_name=url,
            text=combined_text,
        )

        doc = await store.upsert_website_source(
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            display_name=stored_pages[0].title if stored_pages else (crawl_result.display_name or url),
            source_url=url,
            status="indexed",
        )

        crawl_job_store.update(
            job_id,
            status="completed",
            stage="completed",
            message=f"Indexed {len(stored_pages)} page(s).",
            discovered_pages=len(stored_pages),
            indexed_pages=len(stored_pages),
            current_url=None,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

    except Exception as exc:
        if store is not None:
            await store.upsert_website_source(
                agent_id=agent_id,
                tenant_id=tenant_id,
                user_id=user_id,
                display_name=url,
                source_url=url,
                status="failed",
            )
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            message=str(exc),
        )


async def _run_single_page_crawl(
    job_id: str,
    agent_id: str,
    tenant_id: str,
    document_id: str,
    url: str,
) -> None:
    global store, chunk_store
    from app.rag_pipeline import RAGPipeline
    from app.website_service import WebsiteService

    if store is None or chunk_store is None:
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error="Crawl services are not ready.",
            message="Crawl services are not ready.",
        )
        return

    crawl_job_store.update(
        job_id,
        status="running",
        stage="crawling",
        discovered_pages=1,
        indexed_pages=0,
        current_url=url,
        message="Fetching page content.",
        error=None,
    )

    try:
        loop = asyncio.get_running_loop()
        website_dir = store.get_agent_website_dir(agent_id)
        website_dir.mkdir(parents=True, exist_ok=True)
        website_service = WebsiteService(website_directory=website_dir)

        document = await store.get_document(document_id, agent_id, tenant_id)
        if document is None:
            raise KeyError(f"Document '{document_id}' not found.")
        if document.get("source_type") != "website":
            raise ValueError("This document is not a website source.")

        crawled_page = await loop.run_in_executor(None, lambda: website_service.crawl_single_page(url))
        website_service.create_source_page(
            str(document.get("source_url") or ""),
            url=crawled_page.url,
            title=crawled_page.title,
            text=crawled_page.text,
        )

        crawl_job_store.update(
            job_id,
            stage="indexing",
            discovered_pages=1,
            indexed_pages=1,
            current_url=crawled_page.url,
            message="Indexing crawled page.",
            document_id=document["id"],
            document_name=document["file_name"],
        )

        pages = website_service.list_source_pages(str(document.get("source_url") or ""))
        combined_text = "\n\n".join(
            "\n\n".join(part for part in [page.title.strip(), page.url.strip(), page.text.strip()] if part)
            for page in pages
            if page.text.strip()
        ).strip()
        if not combined_text:
            raise ValueError("Website pages must contain readable text.")

        pipeline = RAGPipeline(
            pdf_directory=store.get_agent_pdf_dir(agent_id),
            website_directory=website_dir,
            snippets_directory=store.get_agent_snippet_dir(agent_id),
            qa_directory=store.get_agent_qa_dir(agent_id),
            collection_name=agent_collection_name(agent_id),
        )

        chunk_ids = await chunk_store.get_chunk_ids_by_document(document["id"])
        if chunk_ids:
            pipeline.remove_document(document["id"], chunk_ids=chunk_ids)
            await chunk_store.delete_chunks_by_document(document["id"])

        await pipeline.ingest_single_document(
            chunk_store=chunk_store,
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=document["id"],
            source_type="website",
            source_name=str(document.get("source_url") or ""),
            text=combined_text,
        )

        updated = await store.update_document(
            document["id"],
            agent_id,
            tenant_id,
            file_name=pages[0].title.strip() or str(document.get("source_url") or ""),
            source_url=str(document.get("source_url") or ""),
        )

        crawl_job_store.update(
            job_id,
            status="completed",
            stage="completed",
            discovered_pages=1,
            indexed_pages=1,
            current_url=crawled_page.url,
            message="Crawled and indexed 1 page.",
            document_id=updated["id"],
            document_name=updated["file_name"],
            source_type="website_page",
        )
    except Exception as exc:
        crawl_job_store.update(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            message=str(exc),
            source_type="website_page",
        )


@app.post("/crawl-website", response_model=_CrawlJobResponse, status_code=202)
async def crawl_website(
    request: _CrawlRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    await user.require_permission("knowledge", "write")

    url = _normalize_url(request.url)
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    if store is None:
        raise HTTPException(status_code=503, detail="Store is not ready.")

    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent = await store.require_agent_any_tenant(request.agent_id)
            tenant_id = agent["tenant_id"]
        else:
            await store.require_agent(request.agent_id, user.tenant_id)
            tenant_id = user.tenant_id
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document = await store.upsert_website_source(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        user_id=user.id,
        display_name=url,
        source_url=url,
        status="indexing",
    )

    job = crawl_job_store.create(agent_id=request.agent_id, source_url=url)
    crawl_job_store.update(
        job.id,
        message="Queued for crawling.",
        current_url=url,
        document_id=document["id"],
        document_name=document["file_name"],
    )
    background_tasks.add_task(_run_crawl, job.id, request.agent_id, tenant_id, user.id, url)
    return _CrawlJobResponse(**crawl_job_store.get(job.id).to_dict())


@app.post("/crawl-website-page", response_model=_CrawlJobResponse, status_code=202)
async def crawl_website_page(
    request: _SinglePageCrawlRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    await user.require_permission("knowledge", "write")

    url = _normalize_url(request.url)
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    if store is None:
        raise HTTPException(status_code=503, detail="Store is not ready.")

    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent = await store.require_agent_any_tenant(request.agent_id)
            tenant_id = agent["tenant_id"]
        else:
            await store.require_agent(request.agent_id, user.tenant_id)
            tenant_id = user.tenant_id
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document = await store.get_document(request.document_id, request.agent_id, tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document '{request.document_id}' not found.")
    if document.get("source_type") != "website":
        raise HTTPException(status_code=400, detail="This document is not a website source.")

    job = crawl_job_store.create(agent_id=request.agent_id, source_url=url)
    crawl_job_store.update(
        job.id,
        message="Queued for page crawl.",
        current_url=url,
        discovered_pages=1,
        indexed_pages=0,
        document_id=document["id"],
        document_name=document["file_name"],
        source_type="website_page",
    )
    background_tasks.add_task(
        _run_single_page_crawl,
        job.id,
        request.agent_id,
        tenant_id,
        request.document_id,
        url,
    )
    return _CrawlJobResponse(**crawl_job_store.get(job.id).to_dict())


@app.get("/crawl-website/{job_id}", response_model=_CrawlJobResponse)
async def get_crawl_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    if not await user.has_permission("knowledge", "read") and not await user.has_permission("knowledge", "write"):
        raise HTTPException(status_code=403, detail="Your role does not have 'knowledge:read' permission.")

    if store is None:
        raise HTTPException(status_code=503, detail="Store is not ready.")

    job = crawl_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    try:
        if user.role == UserRole.SUPER_ADMIN:
            await store.require_agent_any_tenant(job.agent_id)
        else:
            await store.require_agent(job.agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _CrawlJobResponse(**job.to_dict())


@app.get("/widget.js")
async def serve_widget():
    widget_path = os.path.join(os.path.dirname(__file__), "..", "public", "widget.js")
    return FileResponse(widget_path, media_type="application/javascript")
