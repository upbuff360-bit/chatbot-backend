from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import CurrentUser, get_current_user
from app.models.lead import LeadResponse
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/agents/{agent_id}/leads", tags=["leads"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


async def _require_lead_read_access(user: CurrentUser) -> None:
    if not (
        await user.has_permission("leads", "read")
        or await user.has_permission("agents", "read")
        or await user.has_permission("chats", "read")
    ):
        raise HTTPException(status_code=403, detail="Your role does not have permission to view leads.")


async def _resolve_tenant(agent_id: str, user: CurrentUser, store: AdminStoreMongo) -> str:
    if user.role == UserRole.SUPER_ADMIN:
        agent = await store.get_agent_by_id(agent_id)
        return agent["tenant_id"]

    agent = await store.get_accessible_agent(agent_id, user.tenant_id, user.id)
    if not agent:
        raise KeyError(f"Agent '{agent_id}' not found.")
    return agent["tenant_id"]


@router.get("", response_model=list[LeadResponse])
async def list_leads(
    agent_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await _require_lead_read_access(user)
    try:
        tenant_id = await _resolve_tenant(agent_id, user, store)
        leads = await store.list_leads(agent_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [LeadResponse(**lead) for lead in leads]
