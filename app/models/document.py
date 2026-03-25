from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "docx", "pptx", "txt", "website", "text_snippet", "qa"]


class DocumentDocument(BaseModel):
    """Mirrors the MongoDB document."""
    id: str = Field(alias="_id")
    tenant_id: str
    agent_id: str
    user_id: str
    file_name: str
    source_type: SourceType
    status: str = "indexed"
    source_url: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    page_count: Optional[int] = None
    page_urls: Optional[list[str]] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    uploaded_at: datetime
    status: str
    source_type: str
    source_url: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    page_count: Optional[int] = None
    page_urls: Optional[list[str]] = None


class DocumentUpdateRequest(BaseModel):
    file_name: Optional[str] = Field(default=None, min_length=1)
    content: Optional[str] = None
    answer: Optional[str] = None


class ConversationMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationDocument(BaseModel):
    id: str = Field(alias="_id")
    tenant_id: str
    agent_id: str
    user_id: str
    title: str
    messages: list[ConversationMessage] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
