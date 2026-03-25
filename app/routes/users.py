from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.permissions_registry import normalize_permission_names
from app.core.dependencies import CurrentUser, get_current_user
from app.core.security import hash_password
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/users", tags=["users"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _require_super_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    # Check both the parsed enum and the raw role_key string so custom-named
    # super_admin roles also pass (role_key preserves the original JWT string).
    if user.role != UserRole.SUPER_ADMIN and user.role_key != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin access required.")
    return user


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "customer"          # default to the system customer role
    plan: str = "free"


class UpdateUserRequest(BaseModel):
    email: str | None = None
    role: str | None = None
    plan: str | None = None
    password: str | None = None


@router.get("/me/subscription")
async def get_my_subscription(
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Any logged-in user can fetch their own subscription."""
    sub = await store.get_user_subscription(user.id)
    return {"subscription": sub}


class UpdateProfileRequest(BaseModel):
    name:    str | None = None
    phone:   str | None = None
    address: str | None = None
    avatar:  str | None = None   # base64 data-url (max ~200 KB enforced in route)


@router.get("/me")
async def get_my_profile(
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Return the authenticated user's own profile (no super admin required)."""
    doc = await store._users.find_one({"_id": user.id}, {"hashed_password": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.put("/me")
async def update_my_profile(
    request: UpdateProfileRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Update the authenticated user's own name, phone, address, and avatar."""
    if request.avatar and len(request.avatar) > 300_000:
        raise HTTPException(status_code=413, detail="Avatar image must be under 200 KB.")

    import datetime
    update: dict = {"updated_at": datetime.datetime.now(datetime.timezone.utc)}
    if request.name    is not None: update["name"]    = request.name.strip()
    if request.phone   is not None: update["phone"]   = request.phone.strip()
    if request.address is not None: update["address"] = request.address.strip()
    if request.avatar  is not None: update["avatar"]  = request.avatar

    doc = await store._users.find_one_and_update(
        {"_id": user.id},
        {"$set": update},
        return_document=True,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    doc.pop("hashed_password", None)
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/me/permissions")
async def get_my_permissions(
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    """
    Returns the resolved permission name strings for the current user's role.
    Accessible to any authenticated user — no roles:read required.
    Used by the sidebar and permission-gated pages to determine what to show.
    """
    if user.role_key == "super_admin" or user.role == UserRole.SUPER_ADMIN:
        return {"permissions": ["*"], "role": user.role_key}

    # 1. Try JWT role key first (fast path — works when JWT is fresh)
    role_doc = (
        await store._roles.find_one({"key": user.role_key})
        or await store._roles.find_one({"name": user.role_key})
    )

    # 2. JWT may be stale — user's role was changed after they logged in.
    #    Fall back to the actual role stored on the user document in MongoDB.
    if not role_doc:
        user_doc = await store._users.find_one({"_id": user.id}, {"role": 1})
        actual_role = user_doc.get("role", "") if user_doc else ""
        if actual_role and actual_role != user.role_key:
            role_doc = (
                await store._roles.find_one({"key": actual_role})
                or await store._roles.find_one({"name": actual_role})
            )

    # 3. Last resort: use the customer system role (always exists)
    if not role_doc:
        role_doc = await store._roles.find_one({"key": "customer", "is_system": True})

    # 4. Final fallback: any non-super-admin system role
    if not role_doc:
        role_doc = await store._roles.find_one(
            {"is_system": True, "is_super_admin": {"$ne": True}}
        )

    if not role_doc:
        return {"permissions": [], "role": user.role_key}

    raw_perms: list[str] = role_doc.get("permissions", [])
    if raw_perms == ["*"]:
        return {"permissions": ["*"], "role": user.role_key}

    # Resolve any legacy UUID refs to names
    all_perm_docs = await store._permissions.find({}, {"_id": 1, "name": 1}).to_list(length=1000)
    id_to_name = {str(p["_id"]): p["name"] for p in all_perm_docs}
    resolved = [
        ref if ":" in ref else id_to_name.get(ref, "")
        for ref in raw_perms
    ]
    resolved = [p for p in resolved if ":" in p]  # drop unresolved/empty refs
    resolved = normalize_permission_names(resolved)

    return {"permissions": resolved, "role": user.role_key}


@router.get("")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("users", "read")
    return await store.list_all_users(skip=skip, limit=limit)


@router.post("", status_code=201)
async def create_user(
    request: CreateUserRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("users", "write")
    if len(request.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    try:
        hashed = hash_password(request.password)
        return await store.admin_create_user(
            email=request.email,
            hashed_password=hashed,
            role=request.role,
            plan=request.plan,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("users", "write")
    try:
        if request.password:
            if len(request.password) < 8:
                raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
            hashed = hash_password(request.password)
            await store._users.update_one(
                {"_id": user_id},
                {"$set": {"hashed_password": hashed}},
            )
        return await store.update_user(
            user_id=user_id,
            email=request.email,
            role=request.role,
            plan=request.plan,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("users", "delete")
    if user_id == user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    try:
        await store.delete_user(user_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


# ── Plan assignment ───────────────────────────────────────────────────────────

class AssignPlanRequest(BaseModel):
    plan_id: str


@router.post("/{user_id}/assign-plan")
async def assign_plan(
    user_id: str,
    request: AssignPlanRequest,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Assign a subscription plan to a user and initialise usage counters."""
    plan = await store.db.plans.find_one({"_id": request.plan_id})
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")

    target = await store._users.find_one({"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")
    if target.get("role") == "super_admin":
        raise HTTPException(status_code=403, detail="Cannot assign plan to super admin.")

    subscription = await store.assign_plan_to_user(user_id, plan)
    return {"message": "Plan assigned successfully.", "subscription": subscription}


@router.get("/{user_id}/subscription")
async def get_subscription(
    user_id: str,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    sub = await store.get_user_subscription(user_id)
    if not sub:
        return {"subscription": None}
    return {"subscription": sub}
