from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class CrawlJob:
    id: str
    agent_id: str
    source_url: str
    status: str
    stage: str
    discovered_pages: int = 0
    indexed_pages: int = 0
    current_url: str | None = None
    message: str | None = None
    error: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    source_type: str = "website"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CrawlJob":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


class CrawlJobStore:
    """
    MongoDB-backed crawl job store.
    Jobs survive server restarts and --reload hot-reloads.
    TTL index auto-purges completed/failed jobs after 24 hours.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["crawl_jobs"]

    async def ensure_indexes(self) -> None:
        await self._col.create_index("agent_id")
        # Auto-delete jobs older than 24 hours
        await self._col.create_index(
            "created_at",
            expireAfterSeconds=86400,
        )

    async def create(self, agent_id: str, source_url: str) -> CrawlJob:
        from uuid import uuid4
        job = CrawlJob(
            id=str(uuid4()),
            agent_id=agent_id,
            source_url=source_url,
            status="queued",
            stage="queued",
            message="Waiting to start crawl.",
        )
        doc = job.to_dict()
        doc["_id"] = doc.pop("id")
        doc["created_at"] = _now()
        await self._col.insert_one(doc)
        return job

    async def get(self, job_id: str) -> CrawlJob | None:
        doc = await self._col.find_one({"_id": job_id})
        if doc is None:
            return None
        doc["id"] = doc.pop("_id")
        doc.pop("created_at", None)
        return CrawlJob.from_dict(doc)

    async def update(self, job_id: str, **changes: Any) -> CrawlJob | None:
        if not changes:
            return await self.get(job_id)
        await self._col.update_one(
            {"_id": job_id},
            {"$set": changes},
        )
        return await self.get(job_id)