from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/plans", tags=["plans"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _require_owner(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role not in (UserRole.SUPER_ADMIN, UserRole.CUSTOMER):
        raise HTTPException(status_code=403, detail="Owner or super admin access required.")
    return user


class PlanRequest(BaseModel):
    name: str
    description: str = ""
    totalMessages: int
    totalTokens: int
    estimatedCost: float
    sellingPrice: float
    chatTokenLimit: int = 0
    summaryTokenLimit: int = 0
    tokensPerMessage: int = 800


@router.get("")
async def list_plans(
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("plans", "read")
    response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
    docs = await store.db.plans.find({}).sort("createdAt", -1).to_list(length=100)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs


@router.post("", status_code=201)
async def create_plan(
    request: PlanRequest,
    user: CurrentUser = Depends(_require_owner),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("plans", "write")
    from datetime import datetime, timezone
    from uuid import uuid4
    doc = {
        "_id":              str(uuid4()),
        "name":             request.name,
        "description":      request.description,
        "totalMessages":    request.totalMessages,
        "totalTokens":      request.totalTokens,
        "estimatedCost":    request.estimatedCost,
        "sellingPrice":     request.sellingPrice,
        "chatTokenLimit":   request.chatTokenLimit or request.totalMessages * 500,
        "summaryTokenLimit":request.summaryTokenLimit or request.totalMessages * 300,
        "tokensPerMessage": request.tokensPerMessage,
        "createdAt":        datetime.now(timezone.utc).isoformat(),
    }
    await store.db.plans.insert_one(doc)
    doc["id"] = doc.pop("_id")
    return doc


@router.put("/{plan_id}")
async def update_plan(
    plan_id: str,
    request: PlanRequest,
    user: CurrentUser = Depends(_require_owner),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("plans", "write")
    from datetime import datetime, timezone
    update = {
        "name":              request.name,
        "description":       request.description,
        "totalMessages":     request.totalMessages,
        "totalTokens":       request.totalTokens,
        "estimatedCost":     request.estimatedCost,
        "sellingPrice":      request.sellingPrice,
        "chatTokenLimit":    request.chatTokenLimit or request.totalMessages * 500,
        "summaryTokenLimit": request.summaryTokenLimit or request.totalMessages * 300,
        "tokensPerMessage":  request.tokensPerMessage,
        "updatedAt":         datetime.now(timezone.utc).isoformat(),
    }
    result = await store.db.plans.find_one_and_update(
        {"_id": plan_id}, {"$set": update}, return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Plan not found.")
    result["id"] = result.pop("_id")
    return result


@router.delete("/{plan_id}", status_code=204)
async def delete_plan(
    plan_id: str,
    user: CurrentUser = Depends(_require_owner),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("plans", "delete")
    await store.db.plans.delete_one({"_id": plan_id})