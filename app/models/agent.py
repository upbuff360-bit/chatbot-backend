from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_SYSTEM_PROMPT = (
    "You are a question-answering assistant.\n"
    "Answer strictly using the provided context.\n"
    "If the answer is not in the context, respond:\n"
    "\"I'd love to help, but I don't have that detail in my current information just yet. If you'd like, ask me about something else and I'll do my best to help.\"\n"
    "Do not guess or fabricate information."
)


class AgentSettings(BaseModel):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    welcome_message: str = "Hi, I'm your AI assistant. Ask me anything about this knowledge base."
    display_name: str = ""
    website_name: str = ""
    website_url: str = ""
    primary_color: str = "#0f172a"
    secondary_color: str = "#f8fafc"
    appearance: str = "light"
    lead_capture_enabled: bool = False


class AgentDocument(BaseModel):
    """Mirrors the MongoDB document."""
    id: str = Field(alias="_id")
    tenant_id: str
    user_id: str
    name: str
    shared_with_user_ids: list[str] = []
    settings: AgentSettings = Field(default_factory=AgentSettings)
    document_count: int = 0
    conversation_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


# ── API shapes ────────────────────────────────────────────────────────────────
class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AgentUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AgentSettingsUpdateRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=0.0, le=1.0)
    welcome_message: str = Field(..., min_length=1)
    display_name: str = ""
    website_name: str = ""
    website_url: str = ""
    primary_color: str = "#0f172a"
    secondary_color: str = "#f8fafc"
    appearance: str = "light"
    lead_capture_enabled: bool = False


class AgentResponse(BaseModel):
    id: str
    display_id: str | None = None
    name: str
    tenant_id: str
    user_id: str
    document_count: int
    conversation_count: int
    created_at: datetime
    can_manage: bool = True
    is_shared: bool = False


class AgentShareRequest(BaseModel):
    user_id: str | None = None
    email: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "AgentShareRequest":
        if self.user_id and self.email:
            raise ValueError("Provide either user_id or email, not both.")
        if not self.user_id and not self.email:
            raise ValueError("Provide either user_id or email.")
        return self


class AgentShareResponse(BaseModel):
    mode: Literal["shared", "invited"]
    email: str
    user_id: str | None = None
    message: str


class AgentShareCandidateResponse(BaseModel):
    id: str
    email: str
    name: str | None = None


class AgentSettingsResponse(BaseModel):
    system_prompt: str
    temperature: float
    welcome_message: str
    display_name: str
    website_name: str
    website_url: str
    primary_color: str
    secondary_color: str
    appearance: str
    lead_capture_enabled: bool
