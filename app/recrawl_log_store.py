from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RecrawlLogEntry:
    """One log entry per agent per scheduled recrawl run."""
    agent_id: str
    agent_name: str
    tenant_id: str
    source_url: str
    status: str              # "success" | "failed" | "skipped"
    pages_crawled: int = 0
    pages_changed: int = 0
    pages_added: int = 0
    pages_removed: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecrawlLogStore:
    """
    MongoDB-backed store for scheduled recrawl audit logs.
    TTL index auto-purges entries older than 90 days.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["recrawl_logs"]

    async def ensure_indexes(self) -> None:
        await self._col.create_index("agent_id")
        await self._col.create_index("tenant_id")

        # Drop the old plain started_at index if it exists — it conflicts
        # with the TTL index we want to create on the same field.
        try:
            await self._col.drop_index("started_at_1")
        except Exception:
            pass  # already gone or never existed — safe to ignore

        # TTL index on started_at — auto-deletes logs older than 90 days.
        # Also serves as the sort index for started_at queries.
        await self._col.create_index(
            "started_at",
            expireAfterSeconds=90 * 24 * 3600,
            name="ttl_90d",
        )

    async def insert(self, entry: RecrawlLogEntry) -> str:
        doc = entry.to_dict()
        doc["started_at"] = entry.started_at
        doc["finished_at"] = entry.finished_at
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_recent(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query: dict = {}
        if agent_id:
            query["agent_id"] = agent_id
        cursor = self._col.find(query, sort=[("started_at", -1)])
        docs = await cursor.to_list(length=limit)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs