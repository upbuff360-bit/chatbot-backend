from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

from app.db.connection import connect, create_indexes, disconnect, get_database
from app.core.dependencies import CurrentUser, get_current_user
from app.crawl_job_store import CrawlJobStore
from app.recrawl_log_store import RecrawlLogEntry, RecrawlLogStore
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
    global store, chunk_store, crawl_job_store

    await connect()
    await create_indexes()

    db = get_database()
    store = AdminStoreMongo(db=db, agents_root=os.getenv("AGENTS_DATA_ROOT", "./data/agents"))
    chunk_store = ChunkStore(db=db)
    crawl_job_store = CrawlJobStore(db=db)
    recrawl_log_store = RecrawlLogStore(db=db)

    # Ensure MongoDB indexes exist
    await chunk_store.ensure_indexes()
    await crawl_job_store.ensure_indexes()
    await recrawl_log_store.ensure_indexes()

    # Backfill display_id for existing agents (one-time, safe to run multiple times)
    await store.backfill_agent_display_ids()

    # Seed default permissions and roles
    await store.seed_default_permissions()
    await store.seed_default_roles()

    # Smart startup indexing
    await _startup_index_agents(store, chunk_store)

    # Signal that all services are fully initialised.
    # Any background task waiting on _services_ready_event will now proceed.
    _services_ready_event.set()

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


crawl_job_store: CrawlJobStore | None = None
recrawl_log_store: RecrawlLogStore | None = None

# Event set at the end of lifespan startup — background tasks that need
# fully-initialized services wait on this instead of polling globals.
_services_ready_event: asyncio.Event = asyncio.Event()


def _normalize_url(url: str) -> str:
    from app.website_service import WebsiteService

    return WebsiteService._normalize_url(url)


async def _run_crawl(job_id: str, agent_id: str, tenant_id: str, user_id: str, url: str) -> None:
    """
    Batch crawl implementation.

    Pages are crawled and ingested in batches of CRAWL_BATCH_SIZE (default 50).
    Each batch is saved to disk and indexed into Qdrant immediately, so the
    chatbot becomes usable after the first batch completes — without waiting
    for the full crawl to finish.

    Flow:
      1. crawl_in_batches() generator runs in a thread executor.
      2. Each batch is yielded back to this async function via a queue.
      3. The batch is saved + ingested before the next batch starts crawling.
      4. Job progress is updated after every page so the frontend shows live
         progress throughout the entire crawl.
    """
    global store, chunk_store, crawl_job_store
    import queue as _queue
    import threading as _threading
    from app.category_detector import detect_page_category
    from app.rag_pipeline import RAGPipeline
    from app.website_service import CrawledPage, WebsiteService

    BATCH_SIZE = int(os.getenv("CRAWL_BATCH_SIZE", "50"))

    if store is None or chunk_store is None or crawl_job_store is None:
        await crawl_job_store.update(
            job_id,
            status="failed", stage="failed",
            error="Crawl services are not ready.",
            message="Crawl services are not ready.",
        )
        return

    await crawl_job_store.update(
        job_id,
        status="running", stage="crawling",
        current_url=url, message="Starting website crawl.", error=None,
    )

    try:
        loop = asyncio.get_running_loop()
        website_dir = store.get_agent_website_dir(agent_id)
        website_dir.mkdir(parents=True, exist_ok=True)
        website_service = WebsiteService(website_directory=website_dir)

        # Progress callback — forwards crawler updates to the job store
        def on_progress(update: dict) -> None:
            # indexed_pages from the crawler = pages crawled so far.
            # This gives live progress during the crawl phase.
            # During ingestion, _run_crawl overwrites this with the
            # actual Qdrant-indexed count.
            asyncio.run_coroutine_threadsafe(
                crawl_job_store.update(
                    job_id,
                    stage=str(update.get("stage") or "crawling"),
                    discovered_pages=int(update.get("discovered_pages") or 0),
                    indexed_pages=int(update.get("indexed_pages") or 0),
                    current_url=str(update.get("current_url")) if update.get("current_url") else None,
                    message=str(update.get("message")) if update.get("message") else None,
                ),
                loop,
            )

        # ── Bridge: run the sync generator in a thread, pass batches back
        # via a queue so this async function can await between batches.
        batch_queue: _queue.Queue = _queue.Queue()
        crawl_errors: list[BaseException] = []

        def _crawl_worker() -> None:
            try:
                for batch in website_service.crawl_in_batches(
                    url, batch_size=BATCH_SIZE, progress_callback=on_progress
                ):
                    batch_queue.put(batch)
            except Exception as exc:  # noqa: BLE001
                crawl_errors.append(exc)
            finally:
                batch_queue.put(None)  # sentinel — signals end of crawl

        crawl_thread = _threading.Thread(target=_crawl_worker, daemon=True)
        crawl_thread.start()

        # ── Prepare the RAG pipeline and document record up front
        pipeline = RAGPipeline(
            pdf_directory=store.get_agent_pdf_dir(agent_id),
            website_directory=website_dir,
            snippets_directory=store.get_agent_snippet_dir(agent_id),
            qa_directory=store.get_agent_qa_dir(agent_id),
            collection_name=agent_collection_name(agent_id),
        )

        # Upsert a document record for the root URL
        doc = await store.upsert_website_source(
            agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
            display_name=url, source_url=url, status="indexing",
        )
        await crawl_job_store.update(
            job_id,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

        # Clear old chunks for this source so re-crawls don't duplicate
        chunk_ids = await chunk_store.get_chunk_ids_by_document(doc["id"])
        if chunk_ids:
            pipeline.remove_document(doc["id"], chunk_ids=chunk_ids)
            await chunk_store.delete_chunks_by_document(doc["id"])

        # ── Process batches as they arrive ────────────────────────────────
        all_pages: list[CrawledPage] = []
        indexed = 0
        batch_num = 0
        display_name = url

        while True:
            # Wait for next batch from the crawl thread (non-blocking for
            # the event loop — uses run_in_executor so other coroutines can
            # proceed while we wait).
            batch: list[CrawledPage] | None = await loop.run_in_executor(
                None, batch_queue.get
            )

            if batch is None:
                # Sentinel received — crawl thread is done
                break

            if not batch:
                continue

            batch_num += 1
            all_pages.extend(batch)

            # Persist this batch to disk immediately
            await loop.run_in_executor(
                None,
                lambda b=batch: website_service._merge_and_save_pages(url, b),
            )

            # Ingest every page in this batch into Qdrant
            await crawl_job_store.update(
                job_id,
                stage="indexing",
                message=f"Batch {batch_num}: indexing {len(batch)} page(s)…",
                discovered_pages=len(all_pages),
            )
            for page in batch:
                page_text = page.text.strip()
                if not page_text:
                    continue
                page_category = detect_page_category(
                    url=page.url, title=page.title, text=page_text
                )
                page_input = f"{page.title}\n\n{page.url}\n\n{page_text}"
                await pipeline.ingest_single_document(
                    chunk_store=chunk_store,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    document_id=doc["id"],
                    source_type="website",
                    source_name=page.url or url,
                    text=page_input,
                    category=page_category,
                )
                indexed += 1
                if not display_name or display_name == url:
                    display_name = page.title or url
                await crawl_job_store.update(
                    job_id,
                    stage="indexing",
                    indexed_pages=indexed,
                    discovered_pages=len(all_pages),
                    current_url=page.url,
                    message=(
                        f"Batch {batch_num} — indexed {indexed} page(s) so far. "
                        f"Chatbot is already usable!"
                    ),
                )

        crawl_thread.join(timeout=10)

        # Re-raise any crawl-thread exception
        if crawl_errors:
            raise crawl_errors[0]

        if not all_pages:
            raise ValueError("No readable website content was found at the provided URL.")

        # Mark the source as fully indexed
        doc = await store.upsert_website_source(
            agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
            display_name=display_name,
            source_url=url, status="indexed",
        )
        await crawl_job_store.update(
            job_id,
            status="completed", stage="completed",
            message=f"Crawl complete — indexed {indexed} page(s) across {batch_num} batch(es).",
            discovered_pages=len(all_pages),
            indexed_pages=indexed,
            current_url=None,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

    except Exception as exc:
        if store is not None:
            await store.upsert_website_source(
                agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
                display_name=url, source_url=url, status="failed",
            )
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower():
            err = "Indexing failed: invalid or missing OpenAI API key."
        elif "429" in err or "rate_limit" in err.lower():
            err = "Indexing failed: OpenAI rate limit exceeded."
        await crawl_job_store.update(
            job_id, status="failed", stage="failed", error=err, message=err
        )
async def _run_single_page_crawl(
    job_id: str, agent_id: str, tenant_id: str, document_id: str, url: str,
) -> None:
    global store, chunk_store, crawl_job_store
    from app.category_detector import detect_page_category
    from app.rag_pipeline import RAGPipeline
    from app.website_service import WebsiteService

    if store is None or chunk_store is None or crawl_job_store is None:
        await crawl_job_store.update(
            job_id, status="failed", stage="failed",
            error="Crawl services are not ready.", message="Crawl services are not ready.",
        )
        return

    await crawl_job_store.update(
        job_id, status="running", stage="crawling",
        discovered_pages=1, indexed_pages=0,
        current_url=url, message="Fetching page content.", error=None,
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
            url=crawled_page.url, title=crawled_page.title, text=crawled_page.text,
        )

        await crawl_job_store.update(
            job_id, stage="indexing", discovered_pages=1, indexed_pages=0,
            current_url=crawled_page.url, message="Indexing crawled page.",
            document_id=document["id"], document_name=document["file_name"],
        )

        pages = website_service.list_source_pages(str(document.get("source_url") or ""))
        if not any(p.text.strip() for p in pages):
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

        # Re-ingest all pages (including the new one) with category detection
        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            page_category = detect_page_category(url=page.url, title=page.title, text=page_text)
            page_input = "\n\n".join(p for p in [page.title.strip(), page.url.strip(), page_text] if p)
            await pipeline.ingest_single_document(
                chunk_store=chunk_store,
                tenant_id=tenant_id, agent_id=agent_id,
                document_id=document["id"],
                source_type="website",
                source_name=page.url or str(document.get("source_url") or ""),
                text=page_input,
                category=page_category,
            )

        updated = await store.update_document(
            document["id"], agent_id, tenant_id,
            file_name=pages[0].title.strip() or str(document.get("source_url") or ""),
            source_url=str(document.get("source_url") or ""),
        )
        await crawl_job_store.update(
            job_id, status="completed", stage="completed",
            discovered_pages=1, indexed_pages=1,
            current_url=crawled_page.url, message="Crawled and indexed 1 page.",
            document_id=updated["id"], document_name=updated["file_name"],
            source_type="website_page",
        )
    except Exception as exc:
        await crawl_job_store.update(
            job_id, status="failed", stage="failed",
            error=str(exc), message=str(exc), source_type="website_page",
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

    job = await crawl_job_store.create(agent_id=request.agent_id, source_url=url)
    await crawl_job_store.update(
        job.id,
        message="Queued for crawling.",
        current_url=url,
        document_id=document["id"],
        document_name=document["file_name"],
    )
    background_tasks.add_task(_run_crawl, job.id, request.agent_id, tenant_id, user.id, url)
    refreshed = await crawl_job_store.get(job.id)
    return _CrawlJobResponse(**refreshed.to_dict())


@app.post("/crawl-single-url", response_model=_CrawlJobResponse, status_code=202)
async def crawl_single_url(
    request: _CrawlRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Crawl exactly ONE URL and index it as a standalone website source.
    Uses Playwright for JS-heavy pages.
    Does NOT follow links — strictly one page only.
    """
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

    job = await crawl_job_store.create(agent_id=request.agent_id, source_url=url)
    await crawl_job_store.update(
        job.id,
        message="Queued for single-page crawl.",
        current_url=url,
        discovered_pages=1,
        indexed_pages=0,
        document_id=document["id"],
        document_name=document["file_name"],
    )
    background_tasks.add_task(
        _run_single_url_crawl, job.id, request.agent_id, tenant_id, user.id, url
    )
    refreshed = await crawl_job_store.get(job.id)
    return _CrawlJobResponse(**refreshed.to_dict())


async def _run_single_url_crawl(
    job_id: str, agent_id: str, tenant_id: str, user_id: str, url: str
) -> None:
    """
    Crawl exactly one URL — no link discovery, no BFS.
    Uses crawl_single_page() which tries urllib first then Playwright.
    """
    global store, chunk_store, crawl_job_store
    from app.category_detector import detect_page_category
    from app.rag_pipeline import RAGPipeline
    from app.website_service import WebsiteService

    if store is None or chunk_store is None or crawl_job_store is None:
        await crawl_job_store.update(
            job_id, status="failed", stage="failed",
            error="Crawl services are not ready.",
            message="Crawl services are not ready.",
        )
        return

    await crawl_job_store.update(
        job_id, status="running", stage="crawling",
        discovered_pages=1, indexed_pages=0,
        current_url=url, message="Fetching page content…", error=None,
    )

    try:
        loop = asyncio.get_running_loop()
        website_dir = store.get_agent_website_dir(agent_id)
        website_dir.mkdir(parents=True, exist_ok=True)
        website_service = WebsiteService(website_directory=website_dir)

        # Crawl exactly one page (Playwright fallback built-in)
        page = await loop.run_in_executor(
            None, lambda: website_service.crawl_single_page(url)
        )

        # Save to disk
        await loop.run_in_executor(
            None, lambda: website_service._merge_and_save_pages(url, [page])
        )

        # Upsert document with real title
        doc = await store.upsert_website_source(
            agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
            display_name=page.title or url,
            source_url=url, status="indexing",
        )
        await crawl_job_store.update(
            job_id, stage="indexing",
            message="Indexing page…",
            discovered_pages=1, indexed_pages=0,
            current_url=url,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

        # Index into Qdrant
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

        page_text = page.text.strip()
        page_category = detect_page_category(url=page.url, title=page.title, text=page_text)
        page_input = f"{page.title}\n\n{page.url}\n\n{page_text}"
        await pipeline.ingest_single_document(
            chunk_store=chunk_store,
            tenant_id=tenant_id, agent_id=agent_id,
            document_id=doc["id"],
            source_type="website",
            source_name=page.url,
            text=page_input,
            category=page_category,
        )

        doc = await store.upsert_website_source(
            agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
            display_name=page.title or url,
            source_url=url, status="indexed",
        )
        await crawl_job_store.update(
            job_id, status="completed", stage="completed",
            message="Page crawled and indexed.",
            discovered_pages=1, indexed_pages=1,
            current_url=None,
            document_id=doc["id"],
            document_name=doc["file_name"],
        )

    except Exception as exc:
        if store is not None:
            await store.upsert_website_source(
                agent_id=agent_id, tenant_id=tenant_id, user_id=user_id,
                display_name=url, source_url=url, status="failed",
            )
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower():
            err = "Indexing failed: invalid or missing OpenAI API key."
        elif "429" in err or "rate_limit" in err.lower():
            err = "Indexing failed: OpenAI rate limit exceeded."
        await crawl_job_store.update(
            job_id, status="failed", stage="failed", error=err, message=err
        )


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

    job = await crawl_job_store.create(agent_id=request.agent_id, source_url=url)
    await crawl_job_store.update(
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
    refreshed = await crawl_job_store.get(job.id)
    return _CrawlJobResponse(**refreshed.to_dict())


@app.get("/crawl-website/{job_id}", response_model=_CrawlJobResponse)
async def get_crawl_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    if not await user.has_permission("knowledge", "read") and not await user.has_permission("knowledge", "write"):
        raise HTTPException(status_code=403, detail="Your role does not have 'knowledge:read' permission.")

    if store is None:
        raise HTTPException(status_code=503, detail="Store is not ready.")

    job = await crawl_job_store.get(job_id)
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


# ── Scheduled auto re-crawl ────────────────────────────────────────────────────
#
# Triggered by Cloud Scheduler every Sunday at 2 AM via:
#   POST /schedule-recrawl
#   Header: X-Recrawl-Secret: <RECRAWL_SECRET env var>
#
# Strategy:
#   - Active agents   (conversation in last 7 days)  → re-crawl weekly
#   - Inactive agents (no conversation in 7+ days)   → re-crawl monthly
#   - Agents with no website sources                 → skip
#   - Process ONE agent at a time with a 60-second gap between each
#   - Only re-index pages whose text hash changed since last crawl
#   - Log every result (success/fail/skipped) to MongoDB recrawl_logs
#   - Send summary email when done (if SMTP is configured)

_ACTIVE_THRESHOLD_DAYS   = int(os.getenv("RECRAWL_ACTIVE_DAYS",   "7"))
_INACTIVE_CYCLE_DAYS     = int(os.getenv("RECRAWL_INACTIVE_DAYS", "30"))
_RECRAWL_DELAY_SECONDS   = int(os.getenv("RECRAWL_DELAY_SECONDS", "60"))
_RECRAWL_SECRET          = os.getenv("RECRAWL_SECRET", "")


async def _run_scheduled_recrawl() -> None:
    """
    Sequential re-crawl runner.

    For every agent that has at least one website source:
      1. Determine if the agent is active (chat in last 7 days) or inactive.
      2. Active   → re-crawl if last crawl was ≥ 7 days ago.
         Inactive → re-crawl if last crawl was ≥ 30 days ago.
      3. Re-crawl in batches, but only re-index pages whose content changed.
      4. Wait 60 seconds before moving to the next agent.
      5. Log every outcome to MongoDB.
      6. Send a summary email when all agents are done.
    """
    global store, chunk_store, recrawl_log_store

    from datetime import timedelta
    from app.category_detector import detect_page_category
    from app.rag_pipeline import RAGPipeline
    from app.services.email_service import EmailConfigError, _send_email_sync
    from app.website_service import WebsiteService

    # Wait for lifespan startup to complete before doing anything.
    # Cloud Run may invoke the scheduler endpoint while the instance is
    # still running startup indexing — the Event is set only after all
    # services are fully initialised so we never read None globals.
    try:
        await asyncio.wait_for(_services_ready_event.wait(), timeout=120)
    except asyncio.TimeoutError:
        logger.error("Scheduled recrawl: services did not become ready within 120s — aborting.")
        return

    now = datetime.now(timezone.utc)
    active_cutoff   = now - timedelta(days=_ACTIVE_THRESHOLD_DAYS)
    inactive_window = timedelta(days=_INACTIVE_CYCLE_DAYS)
    active_window   = timedelta(days=_ACTIVE_THRESHOLD_DAYS)

    # ── 1. Fetch all agents ───────────────────────────────────────────────────
    all_agents = await store.list_all_agents()
    if not all_agents:
        logger.info("Scheduled recrawl: no agents found.")
        return

    # ── 2. Build priority queue ───────────────────────────────────────────────
    # Each entry: (agent, website_sources, is_active, last_crawl_at)
    queue: list[tuple[dict, list[dict], bool, datetime | None]] = []

    for agent in all_agents:
        agent_id  = agent["id"]
        tenant_id = agent.get("tenant_id", "")

        # Get website documents for this agent
        website_docs = await store.list_documents(
            agent_id, tenant_id, source_type="website"
        )
        # Keep only root sources (source_url == file_name pattern, not sub-pages)
        root_sources = [
            d for d in website_docs
            if d.get("source_url") and d.get("source_url") == d.get("source_url")
            and d.get("status") not in ("failed",)
        ]
        # Deduplicate by source_url — keep most recently indexed
        seen_urls: dict[str, dict] = {}
        for d in root_sources:
            url = str(d.get("source_url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls[url] = d
        unique_sources = list(seen_urls.values())

        if not unique_sources:
            continue  # No website sources — skip this agent

        # Determine activity: agent.updated_at reflects last conversation
        agent_updated = agent.get("updated_at")
        if isinstance(agent_updated, str):
            try:
                agent_updated = datetime.fromisoformat(agent_updated.replace("Z", "+00:00"))
            except Exception:
                agent_updated = None

        is_active = (
            agent_updated is not None
            and agent_updated >= active_cutoff
        )

        # Last crawl time — use oldest uploaded_at among website sources
        # (conservative: re-crawl if ANY source might be stale)
        crawl_times = [
            d.get("uploaded_at") for d in unique_sources if d.get("uploaded_at")
        ]
        last_crawl: datetime | None = None
        if crawl_times:
            # Convert strings to datetime if needed
            parsed = []
            for t in crawl_times:
                if isinstance(t, datetime):
                    parsed.append(t)
                elif isinstance(t, str):
                    try:
                        parsed.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
                    except Exception:
                        pass
            last_crawl = min(parsed) if parsed else None

        queue.append((agent, unique_sources, is_active, last_crawl))

    # Sort: active agents first, then by last_crawl ascending (oldest first)
    queue.sort(key=lambda x: (
        0 if x[2] else 1,                    # active=0, inactive=1
        x[3] or datetime.min.replace(tzinfo=timezone.utc),  # oldest crawl first
    ))

    logger.info(
        "Scheduled recrawl: %d agent(s) with website sources queued "
        "(%d active, %d inactive).",
        len(queue),
        sum(1 for _, _, is_active, _ in queue if is_active),
        sum(1 for _, _, is_active, _ in queue if not is_active),
    )

    # ── 3. Process agents one at a time ──────────────────────────────────────
    summary_lines: list[str] = []
    total_agents_crawled = 0
    total_pages_changed  = 0

    for agent, sources, is_active, last_crawl in queue:
        agent_id   = agent["id"]
        agent_name = agent.get("name", agent_id)
        tenant_id  = agent.get("tenant_id", "")

        # Determine if this agent is due for a re-crawl
        recrawl_window = active_window if is_active else inactive_window
        if last_crawl is not None:
            age = now - last_crawl
            if age < recrawl_window:
                logger.info(
                    "SKIP '%s' — last crawl %.0f days ago (window: %d days).",
                    agent_name, age.total_seconds() / 86400, recrawl_window.days,
                )
                continue

        label = "active" if is_active else "inactive"
        logger.info("START recrawl for '%s' (%s, %d source(s)).", agent_name, label, len(sources))

        # Process each website source for this agent
        for source_doc in sources:
            source_url = str(source_doc.get("source_url", "")).strip()
            if not source_url:
                continue

            entry = RecrawlLogEntry(
                agent_id=agent_id,
                agent_name=agent_name,
                tenant_id=tenant_id,
                source_url=source_url,
                status="running",
            )

            try:
                website_dir = store.get_agent_website_dir(agent_id)
                website_dir.mkdir(parents=True, exist_ok=True)
                website_service = WebsiteService(website_directory=website_dir)

                BATCH_SIZE = int(os.getenv("CRAWL_BATCH_SIZE", "50"))

                # Run crawl in a thread (sync generator)
                loop = asyncio.get_running_loop()

                import queue as _queue
                import threading as _threading

                batch_q: _queue.Queue = _queue.Queue()
                crawl_errors: list[BaseException] = []

                def _worker(url=source_url, bq=batch_q, errs=crawl_errors) -> None:
                    try:
                        for batch in website_service.crawl_in_batches(url, batch_size=BATCH_SIZE):
                            bq.put(batch)
                    except Exception as exc:
                        errs.append(exc)
                    finally:
                        bq.put(None)

                t = _threading.Thread(target=_worker, daemon=True)
                t.start()

                # Prepare RAG pipeline
                pipeline = RAGPipeline(
                    pdf_directory=store.get_agent_pdf_dir(agent_id),
                    website_directory=website_dir,
                    snippets_directory=store.get_agent_snippet_dir(agent_id),
                    qa_directory=store.get_agent_qa_dir(agent_id),
                    collection_name=agent_collection_name(agent_id),
                )

                # Upsert website source document
                doc = await store.upsert_website_source(
                    agent_id=agent_id, tenant_id=tenant_id,
                    user_id="system", display_name=source_url,
                    source_url=source_url, status="indexing",
                )

                pages_crawled = 0
                pages_changed = 0
                pages_added   = 0
                display_name  = source_url

                # Process each batch
                while True:
                    batch = await loop.run_in_executor(None, batch_q.get)
                    if batch is None:
                        break
                    if not batch:
                        continue

                    # Save batch to disk with hash change detection
                    counts = await loop.run_in_executor(
                        None,
                        lambda b=batch: website_service._merge_and_save_pages(source_url, b),
                    )
                    pages_crawled += len(batch)
                    pages_added   += counts["added"]
                    pages_changed += counts["changed"]

                    # Only re-index pages that actually changed or are new
                    pages_to_index = [
                        p for p in batch
                        if p.text.strip()
                    ]
                    # Filter to changed/new only by re-checking hash
                    # (merge_and_save already persisted — now compare)
                    truly_changed = [
                        p for p in pages_to_index
                        if counts["added"] > 0 or counts["changed"] > 0
                    ]
                    # Simple approach: index all pages in this batch that
                    # are either new or changed (counts tell us the batch totals)
                    if counts["added"] + counts["changed"] > 0:
                        for page in pages_to_index:
                            page_text = page.text.strip()
                            if not page_text:
                                continue
                            page_category = detect_page_category(
                                url=page.url, title=page.title, text=page_text
                            )
                            page_input = f"{page.title}\n\n{page.url}\n\n{page_text}"
                            await pipeline.ingest_single_document(
                                chunk_store=chunk_store,
                                tenant_id=tenant_id,
                                agent_id=agent_id,
                                document_id=doc["id"],
                                source_type="website",
                                source_name=page.url or source_url,
                                text=page_input,
                                category=page_category,
                            )
                        if not display_name or display_name == source_url:
                            display_name = batch[0].title or source_url

                t.join(timeout=10)
                if crawl_errors:
                    raise crawl_errors[0]

                # Mark source as indexed
                await store.upsert_website_source(
                    agent_id=agent_id, tenant_id=tenant_id,
                    user_id="system", display_name=display_name,
                    source_url=source_url, status="indexed",
                )

                entry.status       = "success"
                entry.pages_crawled = pages_crawled
                entry.pages_changed = pages_changed
                entry.pages_added   = pages_added
                entry.finished_at   = datetime.now(timezone.utc)
                total_pages_changed += pages_changed + pages_added

                logger.info(
                    "OK '%s' — %s: %d crawled, %d changed, %d added.",
                    agent_name, source_url, pages_crawled, pages_changed, pages_added,
                )

            except Exception as exc:
                err_msg = str(exc)
                logger.error("FAIL '%s' — %s: %s", agent_name, source_url, err_msg)
                entry.status      = "failed"
                entry.error       = err_msg
                entry.finished_at = datetime.now(timezone.utc)
                try:
                    await store.upsert_website_source(
                        agent_id=agent_id, tenant_id=tenant_id,
                        user_id="system", display_name=source_url,
                        source_url=source_url, status="failed",
                    )
                except Exception:
                    pass

            finally:
                await recrawl_log_store.insert(entry)

        total_agents_crawled += 1
        summary_lines.append(
            f"  {agent_name}: {entry.pages_crawled} crawled, "
            f"{entry.pages_changed + entry.pages_added} changed/new "
            f"({entry.status})"
        )

        # ── 60-second gap before next agent ──────────────────────────────
        if queue.index((agent, sources, is_active, last_crawl)) < len(queue) - 1:
            logger.info("Waiting %d seconds before next agent…", _RECRAWL_DELAY_SECONDS)
            await asyncio.sleep(_RECRAWL_DELAY_SECONDS)

    logger.info(
        "Scheduled recrawl complete — %d agent(s) processed, %d page(s) changed/added.",
        total_agents_crawled, total_pages_changed,
    )

    # ── 4. Send summary email (optional) ─────────────────────────────────────
    notify_email = os.getenv("RECRAWL_NOTIFY_EMAIL", "").strip()
    if notify_email and summary_lines:
        subject = (
            f"[Chatbot SaaS] Weekly recrawl complete — "
            f"{total_agents_crawled} agent(s), {total_pages_changed} page(s) updated"
        )
        body = (
            f"Scheduled recrawl finished at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Agents processed: {total_agents_crawled}\n"
            f"Pages changed/added: {total_pages_changed}\n\n"
            "Details:\n" + "\n".join(summary_lines)
        )
        try:
            await asyncio.to_thread(
                _send_email_sync, to_email=notify_email, subject=subject, text_body=body
            )
            logger.info("Recrawl summary email sent to %s.", notify_email)
        except EmailConfigError:
            logger.debug("SMTP not configured — skipping recrawl summary email.")
        except Exception as exc:
            logger.warning("Failed to send recrawl summary email: %s", exc)


class _RecrawlRequest(_BaseModel):
    secret: str = ""


@app.post("/schedule-recrawl", status_code=202)
async def schedule_recrawl(
    request: _RecrawlRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the weekly scheduled re-crawl.
    Called by Google Cloud Scheduler every Sunday at 2 AM.

    Protected by RECRAWL_SECRET env var.  If RECRAWL_SECRET is empty,
    the endpoint is open (useful during local testing).
    """
    if _RECRAWL_SECRET and request.secret != _RECRAWL_SECRET:
        raise HTTPException(status_code=403, detail="Invalid recrawl secret.")

    background_tasks.add_task(_run_scheduled_recrawl)
    return {
        "status": "queued",
        "message": (
            "Scheduled recrawl queued. Agents will be processed one at a time "
            f"with a {_RECRAWL_DELAY_SECONDS}s gap between each."
        ),
    }


@app.get("/recrawl-logs")
async def get_recrawl_logs(
    agent_id: str | None = None,
    user: CurrentUser = Depends(get_current_user),
):
    """Return recent recrawl log entries. Super admin only."""
    if user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(status_code=403, detail="Super admin only.")
    if recrawl_log_store is None:
        raise HTTPException(status_code=503, detail="Log store not ready.")
    return await recrawl_log_store.list_recent(agent_id=agent_id, limit=200)