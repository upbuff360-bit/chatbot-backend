from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


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


class CrawlJobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, CrawlJob] = {}

    def create(self, agent_id: str, source_url: str) -> CrawlJob:
        job = CrawlJob(
            id=str(uuid4()),
            agent_id=agent_id,
            source_url=source_url,
            status="queued",
            stage="queued",
            message="Waiting to start crawl.",
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> CrawlJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> CrawlJob:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            return job
