from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    OWNER = "owner"
    ADMIN = "admin"
    VIEWER = "viewer"
    CUSTOMER = "customer"


# ── DB document shape (internal) ─────────────────────────────────────────────
class UserDocument(BaseModel):
    """Mirrors the MongoDB document exactly."""
    id: str = Field(alias="_id")
    email: str
    hashed_password: str
    tenant_id: str
    name: Optional[str] = None
    role: str = "owner"    # plain string — supports custom role names beyond the enum
    plan: str = "free"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


# ── API request/response shapes ──────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    name: Optional[str] = None
    invite_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3)


class ForgotPasswordResponse(BaseModel):
    message: str


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=12)
    password: str = Field(..., min_length=8)


class PasswordResetPreviewResponse(BaseModel):
    email: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    role: str          # plain string — custom roles aren't in the UserRole enum
    tenant_id: str
    plan: Optional[str] = "free"  # None-safe — admin-created users may have null plan in DB
    created_at: datetime

    model_config = {"populate_by_name": True}


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class InvitationPreviewResponse(BaseModel):
    email: str
    agent_name: str
    inviter_email: Optional[str] = None
