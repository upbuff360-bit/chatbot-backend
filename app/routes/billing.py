from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/billing", tags=["billing"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _require_super_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(status_code=403, detail="Super admin access required.")
    return user


def _format_record(user: dict) -> dict | None:
    sub = user.get("subscription")
    if not sub:
        return None
    return {
        "user_id":               str(user.get("_id", user.get("id", ""))),
        "email":                 user.get("email", ""),
        "plan_id":               sub.get("plan_id", ""),
        "plan_name":             sub.get("plan_name", ""),
        "selling_price":         sub.get("selling_price", 0),
        "duration_months":       sub.get("duration_months", 1),
        "cycle_start_date":      sub.get("cycle_start_date", ""),
        "cycle_end_date":        sub.get("cycle_end_date", ""),
        "assigned_at":           sub.get("assigned_at", ""),
        "remaining_messages":    sub.get("remaining_messages", 0),
        "monthly_message_limit": sub.get("monthly_message_limit", 0),
        "billing_status":        sub.get("billing_status", "active"),
    }


# ── List endpoints ────────────────────────────────────────────────────────────

@router.get("/all")
async def get_all_billing(
    response: Response,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("billing", "read")
    response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=120"
    users = await store._users.find(
        {"subscription": {"$exists": True}},
        {"email": 1, "subscription": 1}
    ).to_list(length=500)
    return [r for r in (_format_record(u) for u in users) if r]


@router.get("/me")
async def get_my_billing(
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("billing", "read")
    response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=120"
    doc = await store._users.find_one(
        {"_id": user.id},
        {"email": 1, "subscription": 1}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    record = _format_record(doc)
    return [record] if record else []


# ── Billing actions ───────────────────────────────────────────────────────────

@router.patch("/{user_id}/pause")
async def pause_billing(
    user_id: str,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Pause billing — agents stop responding."""
    await store.set_billing_status(user_id, "paused")
    return {"message": "Billing paused. Agents will not respond until resumed."}


@router.patch("/{user_id}/stop")
async def stop_billing(
    user_id: str,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Stop billing — agents permanently stopped until renewed."""
    await store.set_billing_status(user_id, "stopped")
    return {"message": "Billing stopped. Assign a new plan to resume."}


@router.patch("/{user_id}/resume")
async def resume_billing(
    user_id: str,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Resume a paused billing."""
    await store.set_billing_status(user_id, "active")
    return {"message": "Billing resumed. Agents are active."}


class RenewRequest(BaseModel):
    plan_id: str
    duration_months: int = 1


@router.post("/{user_id}/renew")
async def renew_billing(
    user_id: str,
    request: RenewRequest,
    user: CurrentUser = Depends(_require_super_admin),
    store: AdminStoreMongo = Depends(_get_store),
):
    """Renew plan — reassigns plan and resets cycle."""
    plan = await store.db.plans.find_one({"_id": request.plan_id})
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    subscription = await store.assign_plan_to_user(
        user_id, plan, duration_months=request.duration_months
    )
    return {"message": "Plan renewed successfully.", "subscription": subscription}