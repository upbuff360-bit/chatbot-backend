from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import CurrentUser, get_current_user
from app.models.agent import (
    AgentCreateRequest,
    AgentShareCandidateResponse,
    AgentShareRequest,
    AgentShareResponse,
    AgentResponse,
    AgentSettingsResponse,
    AgentSettingsUpdateRequest,
    AgentUpdateRequest,
)
from app.models.user import UserRole
from app.services.email_service import EmailConfigError, build_agent_invite_url, send_agent_invitation_email
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/agents", tags=["agents"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _serialize(doc: dict) -> AgentResponse:
    return AgentResponse(
        id=doc["id"],
        display_id=doc.get("display_id"),
        name=doc["name"],
        tenant_id=doc["tenant_id"],
        user_id=doc["user_id"],
        document_count=doc.get("document_count", 0),
        conversation_count=doc.get("conversation_count", 0),
        created_at=doc["created_at"],
        can_manage=doc.get("can_manage", True),
        is_shared=doc.get("is_shared", False),
    )


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    if not (
        await user.has_permission("agents", "read")
        or await user.has_permission("chats", "read")
        or await user.has_permission("leads", "read")
    ):
        raise HTTPException(status_code=403, detail="Your role does not have permission to view agents.")
    # Super admin sees all agents across all tenants
    if user.role == UserRole.SUPER_ADMIN:
        agents = await store.list_all_agents()
    else:
        agents = await store.list_agents(user.tenant_id, user.id)
    # Filter out agents missing required fields
    valid = [a for a in agents if a.get("name") and a.get("tenant_id")]
    return [_serialize(a) for a in valid]


# @router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
# async def create_agent(
#     request: AgentCreateRequest,
#     user: CurrentUser = Depends(get_current_user),
#     store: AdminStoreMongo = Depends(_get_store),
# ):
#     await user.require_permission("agents", "write")
#     agent = await store.create_agent(request.name, user.tenant_id, user.id)
#     return _serialize(agent)
@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("agents", "write")
    if user.role != UserRole.SUPER_ADMIN:
        subscription = await store.get_user_subscription(user.id)
        if not subscription:
            raise HTTPException(
                status_code=402,
                detail="No active plan assigned. Please contact your administrator to assign a plan before creating agents."
            )
    agent = await store.create_agent(request.name, user.tenant_id, user.id)
    return _serialize(agent)
    

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent = await store.get_agent_by_id(agent_id)
        else:
            agent = await store.require_accessible_agent(agent_id, user.tenant_id, user.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _serialize(agent)


@router.get("/{agent_id}/share-candidates", response_model=list[AgentShareCandidateResponse])
async def share_candidates(
    agent_id: str,
    query: str = "",
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("agents", "write")
    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent = await store.get_agent_by_id(agent_id)
            tenant_id = agent["tenant_id"]
        else:
            await store.require_agent(agent_id, user.tenant_id)
            tenant_id = user.tenant_id
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    matches = await store.search_users_by_email(
        query,
        exclude_user_id=user.id,
        exclude_tenant_id=tenant_id,
        exclude_agent_id=agent_id,
    )
    return [AgentShareCandidateResponse(**item) for item in matches]


@router.post("/{agent_id}/share", response_model=AgentShareResponse, status_code=status.HTTP_201_CREATED)
async def share_agent(
    agent_id: str,
    request: AgentShareRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("agents", "write")
    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent = await store.get_agent_by_id(agent_id)
            tenant_id = agent["tenant_id"]
        else:
            await store.require_agent(agent_id, user.tenant_id)
            tenant_id = user.tenant_id

        if request.user_id:
            result = await store.share_agent(agent_id, tenant_id, request.user_id)
            return AgentShareResponse(
                mode="shared",
                email=result["email"],
                user_id=result["user_id"],
                message=f"Agent shared with {result['email']}.",
            )

        invite = await store.create_agent_invitation(
            agent_id=agent_id,
            owner_tenant_id=tenant_id,
            invited_by_user_id=user.id,
            invited_email=request.email or "",
        )
        invite_url = build_agent_invite_url(invite["token"], invite["invited_email"])
        try:
            await send_agent_invitation_email(
                to_email=invite["invited_email"],
                agent_name=invite["agent_name"],
                inviter_email=user.email,
                invite_url=invite_url,
            )
        except Exception:
            await store.delete_agent_invitation(invite["id"])
            raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmailConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to send invite email: {exc}") from exc
    return AgentShareResponse(
        mode="invited",
        email=invite["invited_email"],
        user_id=None,
        message=f"Invite email sent to {invite['invited_email']}.",
    )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("agents", "write")
    try:
        # Super admin can update any agent
        tenant_id = user.tenant_id
        if user.role == UserRole.SUPER_ADMIN:
            existing = await store.get_agent_by_id(agent_id)
            tenant_id = existing["tenant_id"]
        agent = await store.update_agent(agent_id, tenant_id, request.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _serialize(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("agents", "delete")
    try:
        if user.role == UserRole.SUPER_ADMIN:
            existing = await store.get_agent_by_id(agent_id)
            tenant_id = existing["tenant_id"]
        else:
            await store.require_agent(agent_id, user.tenant_id)
            tenant_id = user.tenant_id
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await store.delete_agent(agent_id, tenant_id)


@router.get("/{agent_id}/settings", response_model=AgentSettingsResponse)
async def get_settings(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        if user.role == UserRole.SUPER_ADMIN:
            existing = await store.get_agent_by_id(agent_id)
            tenant_id = existing["tenant_id"]
        else:
            existing = await store.require_accessible_agent(agent_id, user.tenant_id, user.id)
            tenant_id = existing["tenant_id"]
        settings = await store.get_settings(agent_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentSettingsResponse(**settings)


@router.put("/{agent_id}/settings", response_model=AgentSettingsResponse)
async def update_settings(
    agent_id: str,
    request: AgentSettingsUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("settings", "write")
    try:
        if user.role == UserRole.SUPER_ADMIN:
            existing = await store.get_agent_by_id(agent_id)
            tenant_id = existing["tenant_id"]
        else:
            tenant_id = user.tenant_id
        settings = await store.update_settings(agent_id, tenant_id, request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentSettingsResponse(**settings)
