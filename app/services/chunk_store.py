from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChunkStore:
    """
    Handles all MongoDB operations for the `chunks` collection.
    This is the source of truth for all knowledge base text content.
    Qdrant stores only chunk_ids — full text lives here.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._chunks = db.chunks

    async def ensure_indexes(self) -> None:
        await self._chunks.create_index("agent_id")
        await self._chunks.create_index("document_id")
        await self._chunks.create_index("tenant_id")

    async def save_chunks(
        self,
        tenant_id: str,
        agent_id: str,
        document_id: str,
        source_type: str,
        source_name: str,
        chunks: list[str],
        category: str | None = None,
    ) -> list[str]:
        """Save chunks to MongoDB. Returns list of chunk IDs in same order."""
        if not chunks:
            return []

        now = _now()
        docs = []
        ids: list[str] = []

        for i, content in enumerate(chunks):
            chunk_id = str(uuid4())
            ids.append(chunk_id)
            doc: dict = {
                "_id": chunk_id,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "document_id": document_id,
                "source_type": source_type,
                "source_name": source_name,
                "chunk_index": i,
                "content": content,
                "created_at": now,
            }
            if category:
                doc["category"] = category
            docs.append(doc)

        await self._chunks.insert_many(docs)
        return ids

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by IDs preserving Qdrant relevance order."""
        if not chunk_ids:
            return []
        cursor = self._chunks.find({"_id": {"$in": chunk_ids}})
        results = await cursor.to_list(length=len(chunk_ids))
        result_map = {r["_id"]: r for r in results}
        return [result_map[cid] for cid in chunk_ids if cid in result_map]

    async def delete_chunks_by_document(self, document_id: str) -> int:
        result = await self._chunks.delete_many({"document_id": document_id})
        return result.deleted_count

    async def delete_chunks_by_agent(self, agent_id: str) -> int:
        result = await self._chunks.delete_many({"agent_id": agent_id})
        return result.deleted_count

    async def get_all_chunks_by_agent(self, agent_id: str) -> list[dict]:
        """Get all chunks for rebuilding Qdrant index from MongoDB."""
        cursor = self._chunks.find({"agent_id": agent_id})
        return await cursor.to_list(length=None)

    async def count_chunks_by_agent(self, agent_id: str) -> int:
        return await self._chunks.count_documents({"agent_id": agent_id})

    async def has_chunks_for_agent(self, agent_id: str) -> bool:
        doc = await self._chunks.find_one({"agent_id": agent_id}, {"_id": 1})
        return doc is not None

    async def get_chunk_ids_by_document(self, document_id: str) -> list[str]:
        cursor = self._chunks.find({"document_id": document_id}, {"_id": 1})
        docs = await cursor.to_list(length=None)
        return [d["_id"] for d in docs]