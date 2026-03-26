from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.core.permissions_registry import normalize_permission_names
from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/roles", tags=["roles"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _require_super_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Only super admin can access this."""
    if user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(status_code=403, detail="Super admin access required.")
    return user


def _require_manager(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Owner and above can manage roles and permissions (except super_admin role)."""
    if user.role not in (UserRole.SUPER_ADMIN, UserRole.CUSTOMER):
        raise HTTPException(status_code=403, detail="Owner or super admin access required.")
    return user


# ── Permissions ───────────────────────────────────────────────────────────────

class CreatePermissionRequest(BaseModel):
    name: str
    description: str = ""
    resource: str
    action: str


@router.get("/permissions")
async def list_permissions(
    response: Response,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "read")
    response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
    return await store.list_permissions()


@router.post("/permissions", status_code=201)
async def create_permission(
    request: CreatePermissionRequest,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "manage")
    try:
        return await store.create_permission(
            name=request.name,
            description=request.description,
            resource=request.resource,
            action=request.action,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/permissions/{permission_id}", status_code=204)
async def delete_permission(
    permission_id: str,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "manage")
    await store.delete_permission(permission_id)


# ── Roles ─────────────────────────────────────────────────────────────────────

class CreateRoleRequest(BaseModel):
    name: str
    description: str = ""
    permissions: list[str] = []


class UpdateRoleRequest(BaseModel):
    name: str
    description: str = ""
    permissions: list[str] = []


@router.post("/reseed-permissions", status_code=200)
async def reseed_permissions(
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """
    Re-run permission seeding from the registry.
    Safe to call multiple times — only inserts missing permissions.
    Use this after adding a new page to permissions_registry.py.
    """
    await store.seed_default_permissions()
    permissions = await store.list_permissions()
    return {"message": f"Permissions synced. Total: {len(permissions)}.", "count": len(permissions)}


@router.get("")
async def list_roles(
    response: Response,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "read")
    response.headers["Cache-Control"] = "no-store"

    roles = await store.list_roles()

    # Permissions are now stored as name strings directly (e.g. "agents:read").
    # Any legacy ID refs are resolved here for backward compat.
    all_perms = await store.list_permissions()
    id_to_name = {p["id"]: p["name"] for p in all_perms}

    for role in roles:
        for field in ("permissions", "seed_permissions", "extra_permissions"):
            raw: list[str] = role.get(field, [])
            if raw == ["*"]:
                role[field] = ["*"]
            else:
                # If the value contains ":" it's already a name; otherwise resolve ID
                role[field] = normalize_permission_names([
                    ref if ":" in ref else id_to_name.get(ref, ref)
                    for ref in raw
                ])

    return roles


@router.post("", status_code=201)
async def create_role(
    request: CreateRoleRequest,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "manage")
    try:
        return await store.create_role(
            name=request.name,
            description=request.description,
            permissions=request.permissions,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.put("/{role_id}")
async def update_role(
    role_id: str,
    request: UpdateRoleRequest,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "manage")
    try:
        return await store.update_role(
            role_id=role_id,
            name=request.name,
            description=request.description,
            permissions=request.permissions,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{role_id}", status_code=204)
async def delete_role(
    role_id: str,
    user: CurrentUser = Depends(_require_manager),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("roles", "manage")
    try:
        await store.delete_role(role_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
