from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


LeadSource = Literal["chat", "widget"]


class LeadDocument(BaseModel):
    id: str = Field(alias="_id")
    tenant_id: str
    agent_id: str
    conversation_id: str
    source: LeadSource = "chat"
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    interest: str | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


class LeadResponse(BaseModel):
    id: str
    conversation_id: str
    source: LeadSource
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    interest: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime
