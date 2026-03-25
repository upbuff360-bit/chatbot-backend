from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.db.connection import connect, create_indexes, disconnect, get_database
from app.services.admin_store_mongo import AdminStoreMongo
from app.routes import auth, agents, documents, chat, dashboard

logger = logging.getLogger(__name__)

store: AdminStoreMongo | None = None


def agent_collection_name(agent_id: str) -> str:
    return f"knowledge_base_{agent_id.replace('-', '_')}"


def _build_pipeline(agent_id: str):
    from app.rag_pipeline import RAGPipeline
    return RAGPipeline(
        pdf_directory=store.get_agent_pdf_dir(agent_id),
        website_directory=store.get_agent_website_dir(agent_id),
        snippets_directory=store.get_agent_snippet_dir(agent_id),
        qa_directory=store.get_agent_qa_dir(agent_id),
        collection_name=agent_collection_name(agent_id),
    )


async def _index_all_agents() -> None:
    if store is None:
        return

    cursor = store.db.agents.find({}, {"_id": 1, "name": 1})
    agent_docs = await cursor.to_list(length=1000)

    if not agent_docs:
        print("No agents found — skipping startup indexing.")
        return

    print(f"Startup: indexing {len(agent_docs)} agent(s)...")
    loop = asyncio.get_event_loop()

    for agent_doc in agent_docs:
        agent_id   = agent_doc["_id"]
        agent_name = agent_doc.get("name", agent_id)

        def _ingest(aid: str = agent_id, name: str = agent_name):
            try:
                pipeline = _build_pipeline(aid)
                stats = pipeline.ingest_documents(recreate=True)
                return f"  OK '{name}' — {stats.chunks_indexed} chunks indexed"
            except Exception as exc:
                return f"  SKIP '{name}' — {exc}"

        result = await loop.run_in_executor(None, _ingest)
        print(result)

    print("Startup indexing complete.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global store

    await connect()
    await create_indexes()

    db = get_database()
    store = AdminStoreMongo(db=db, agents_root=os.getenv("AGENTS_DATA_ROOT", "./data/agents"))

    await _index_all_agents()

    yield

    await disconnect()


app = FastAPI(
    title="Chatbot SaaS API",
    version="3.0.0",
    lifespan=lifespan,
)

_origins = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(agents.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(dashboard.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/widget.js", include_in_schema=False)
async def serve_widget():
    """Serve the embeddable chat widget — no auth required."""
    widget_path = Path(__file__).parent.parent / "public" / "widget.js"
    if not widget_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="widget.js not found. Place it in chatbot/public/widget.js")
    return FileResponse(str(widget_path), media_type="application/javascript")

