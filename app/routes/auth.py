from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import create_access_token, hash_password, verify_password
from app.db.connection import get_database
from app.models.user import (
    AuthResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    InvitationPreviewResponse,
    LoginRequest,
    PasswordResetPreviewResponse,
    ResetPasswordRequest,
    SignupRequest,
    UserResponse,
)
from app.services.admin_store_mongo import AdminStoreMongo
from app.services.email_service import (
    EmailConfigError,
    build_password_reset_url,
    send_password_reset_email,
)

router = APIRouter(prefix="/auth", tags=["auth"])

PASSWORD_RESET_TTL = timedelta(hours=1)


def _get_store() -> AdminStoreMongo:
    from app.main import store

    return store


def _password_resets():
    return get_database().password_resets


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest, store: AdminStoreMongo = Depends(_get_store)):
    email = request.email.strip()
    existing = await store.get_user_by_email(email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    if request.invite_token:
        invite = await store.get_agent_invitation_by_token(request.invite_token)
        if not invite:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This invite is invalid or has expired.",
            )
        if invite["invited_email"].lower() != email.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This invite is only valid for {invite['invited_email']}.",
            )

    tenant_id = str(uuid4())
    hashed = hash_password(request.password)

    user = await store.create_user(
        email=email,
        hashed_password=hashed,
        tenant_id=tenant_id,
        role="owner",
        name=request.name or email.split("@")[0],
    )
    await store.accept_pending_agent_invites(email, user["_id"])

    token = create_access_token(
        {
            "sub": user["_id"],
            "email": user["email"],
            "name": user.get("name", ""),
            "tenant_id": tenant_id,
            "role": "owner",
        }
    )

    return AuthResponse(
        access_token=token,
        user=UserResponse(
            id=user["_id"],
            email=user["email"],
            name=user.get("name"),
            role=user["role"],
            tenant_id=user["tenant_id"],
            plan=user["plan"],
            created_at=user["created_at"],
        ),
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, store: AdminStoreMongo = Depends(_get_store)):
    user = await store.get_user_by_email(request.email.strip())
    if user is None or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(
        {
            "sub": user["_id"],
            "email": user["email"],
            "name": user.get("name", user["email"].split("@")[0]),
            "tenant_id": user["tenant_id"],
            "role": user["role"],
        }
    )

    return AuthResponse(
        access_token=token,
        user=UserResponse(
            id=user["_id"],
            email=user["email"],
            name=user.get("name"),
            role=user["role"],
            tenant_id=user["tenant_id"],
            plan=user["plan"],
            created_at=user["created_at"],
        ),
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    store: AdminStoreMongo = Depends(_get_store),
):
    email = request.email.strip()
    generic_message = "If an account exists for this email, a password reset link has been sent."
    user = await store.get_user_by_email(email)
    if user is None:
        return ForgotPasswordResponse(message=generic_message)

    email = user["email"]

    now = datetime.now(timezone.utc)
    token = secrets.token_urlsafe(32)
    expires_at = now + PASSWORD_RESET_TTL

    await _password_resets().delete_many({"email": email})
    await _password_resets().insert_one(
        {
            "_id": str(uuid4()),
            "token": token,
            "user_id": user["_id"],
            "email": email,
            "created_at": now,
            "expires_at": expires_at,
        }
    )

    try:
        await send_password_reset_email(
            to_email=email,
            reset_url=build_password_reset_url(token, email),
        )
    except EmailConfigError as exc:
        await _password_resets().delete_many({"email": email})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ForgotPasswordResponse(message=generic_message)


@router.get("/password-reset/{token}", response_model=PasswordResetPreviewResponse)
async def get_password_reset_preview(token: str, store: AdminStoreMongo = Depends(_get_store)):
    now = datetime.now(timezone.utc)
    reset_doc = await _password_resets().find_one({"token": token, "expires_at": {"$gt": now}})
    if not reset_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This password reset link is invalid or has expired.",
        )

    user = await store.get_user_by_email(reset_doc["email"])
    if user is None:
        await _password_resets().delete_one({"_id": reset_doc["_id"]})
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This password reset link is invalid or has expired.",
        )

    return PasswordResetPreviewResponse(email=user["email"])


@router.post("/reset-password", response_model=ForgotPasswordResponse)
async def reset_password(
    request: ResetPasswordRequest,
    store: AdminStoreMongo = Depends(_get_store),
):
    now = datetime.now(timezone.utc)
    reset_doc = await _password_resets().find_one({"token": request.token, "expires_at": {"$gt": now}})
    if not reset_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This password reset link is invalid or has expired.",
        )

    user = await store.get_user_by_email(reset_doc["email"])
    if user is None:
        await _password_resets().delete_one({"_id": reset_doc["_id"]})
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This password reset link is invalid or has expired.",
        )

    await get_database().users.update_one(
        {"_id": user["_id"]},
        {"$set": {"hashed_password": hash_password(request.password), "updated_at": now}},
    )
    await _password_resets().delete_many({"email": reset_doc["email"]})

    return ForgotPasswordResponse(message="Your password has been reset successfully.")


@router.get("/invitations/{token}", response_model=InvitationPreviewResponse)
async def get_invitation(token: str, store: AdminStoreMongo = Depends(_get_store)):
    invite = await store.get_agent_invitation_by_token(token)
    if not invite:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This invite is invalid or has expired.")
    return InvitationPreviewResponse(
        email=invite["invited_email"],
        agent_name=invite["agent_name"],
        inviter_email=invite.get("inviter_email"),
    )


@router.get("/me", response_model=UserResponse)
async def me(store: AdminStoreMongo = Depends(_get_store)):
    raise HTTPException(status_code=501, detail="Use GET /auth/me with Bearer token.")
