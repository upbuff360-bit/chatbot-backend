from __future__ import annotations

from datetime import datetime, timezone
import re
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
        await self._chunks.create_index([("agent_id", 1), ("chunk_type", 1), ("category", 1)])

    async def save_chunks(
        self,
        tenant_id: str,
        agent_id: str,
        document_id: str,
        source_type: str,
        source_name: str,
        chunks: list[str],
        category: str | None = None,
        chunk_type: str = "detail",
    ) -> list[str]:
        """
        Save chunks to MongoDB. Returns list of chunk IDs in same order.

        chunk_type — "summary" for the compact per-product summary chunk,
                     "detail"  for normal sentence-boundary content chunks.
        Summary chunks are fetched first for list queries; detail chunks are
        used for specific single-product follow-up questions.
        """
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
                "chunk_type": chunk_type,
                "created_at": now,
            }
            if category:
                doc["category"] = category
            docs.append(doc)

        await self._chunks.insert_many(docs)
        return ids

    async def get_summary_chunk_ids_by_agent(
        self,
        agent_id: str,
        *,
        category: str | None = None,
    ) -> list[str]:
        """
        Return IDs of summary chunks for this agent across every source type.

        category:
          - None      -> return both product and service summary chunks
          - "product" -> return only product summary chunks
          - "service" -> return only service summary chunks
        """
        query: dict = {"agent_id": agent_id, "chunk_type": "summary"}
        if category:
            query["category"] = category
        cursor = self._chunks.find(
            query,
            {"_id": 1},
        )
        docs = await cursor.to_list(length=None)
        return [d["_id"] for d in docs]

    async def get_text_snippet_and_qa_chunks_by_agent(self, agent_id: str) -> list[dict]:
        """
        Return ALL chunks for text_snippet and qa source types for this agent.

        These are manually curated by the user so they must ALWAYS appear in
        list-question answers, regardless of vector similarity scores or URL
        patterns.  Unlike crawled product pages they have no summary chunk and
        their source_name is a plain title (not a URL), so they are invisible
        to get_summary_chunk_ids_by_agent's URL-pattern filter.
        """
        cursor = self._chunks.find(
            {"agent_id": agent_id, "source_type": {"$in": ["text_snippet", "qa"]}},
            {"_id": 1, "content": 1, "source_name": 1, "source_type": 1},
        )
        return await cursor.to_list(length=None)

    async def get_detail_chunks_by_agent_and_category(
        self,
        agent_id: str,
        *,
        category: str,
        limit: int = 40,
        source_types: list[str] | None = None,
        exclude_source_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Return detail chunks for a specific category.

        This is used as a stable fallback for list-style product/service queries
        so the answer can still be complete even when vector search happens to
        miss the most catalog-like chunk for a particular phrasing.
        """
        query: dict = {
            "agent_id": agent_id,
            "chunk_type": "detail",
            "category": category,
        }
        if source_types:
            query["source_type"] = {"$in": source_types}
        if exclude_source_types:
            query["source_type"] = {
                **(query.get("source_type") or {}),
                "$nin": exclude_source_types,
            }

        cursor = self._chunks.find(
            query,
            {
                "_id": 1,
                "content": 1,
                "source_name": 1,
                "category": 1,
                "chunk_index": 1,
                "source_type": 1,
            },
        ).sort([("source_type", 1), ("source_name", 1), ("chunk_index", 1)]).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_contact_chunks_by_agent(
        self,
        agent_id: str,
        *,
        limit: int = 3,
    ) -> list[dict]:
        """
        Return chunks that contain explicit contact details for this agent.

        Used to stabilize pricing/recommendation handoff answers so email and
        phone details appear consistently when they truly exist in knowledge.
        """
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        phone_pattern = re.compile(r"(?:\+\d[\d\s().-]{7,}\d|\b\d{10,}\b)")
        placeholder_pattern = re.compile(
            r"(?i)\b(?:email|mail|e-mail|phone|mobile|contact)\s*:\s*(?:\[\s*not available\s*\]|not available|n/?a|na|unknown|-)\b"
        )
        projection = {
            "_id": 1,
            "content": 1,
            "source_name": 1,
            "source_type": 1,
            "chunk_type": 1,
            "chunk_index": 1,
        }
        sort_order = [("chunk_type", 1), ("source_type", 1), ("source_name", 1), ("chunk_index", 1)]

        email_docs = await self._chunks.find(
            {
                "agent_id": agent_id,
                "content": {"$regex": email_pattern.pattern, "$options": "i"},
            },
            projection,
        ).sort(sort_order).limit(max(limit, 4)).to_list(length=max(limit, 4))

        phone_docs = await self._chunks.find(
            {
                "agent_id": agent_id,
                "content": {"$regex": phone_pattern.pattern},
            },
            projection,
        ).sort(sort_order).limit(max(limit, 4)).to_list(length=max(limit, 4))

        combined: list[dict] = []
        seen_ids: set[str] = set()
        for bucket in (email_docs, phone_docs):
            for doc in bucket:
                doc_id = str(doc.get("_id"))
                content = str(doc.get("content") or "").strip()
                if not content or doc_id in seen_ids:
                    continue
                if placeholder_pattern.search(content):
                    continue
                seen_ids.add(doc_id)
                combined.append(doc)
                if len(combined) >= limit:
                    return combined

        return combined

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
