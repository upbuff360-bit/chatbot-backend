from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ── Original summary (kept for backward-compat) ───────────────────────────────

class DashboardResponse(BaseModel):
    total_agents: int
    total_documents: int
    total_conversations: int
    recent_activity: list


@router.get("/summary", response_model=DashboardResponse)
async def dashboard_summary(
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    summary = await store.dashboard_summary(user.tenant_id, user.id)
    return DashboardResponse(**summary)


# ── Full dashboard ────────────────────────────────────────────────────────────

@router.get("/full")
async def dashboard_full(
    response: Response,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    await user.require_permission("dashboard", "read")
    response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=120"

    tid = user.tenant_id
    uid = user.id
    is_super = user.role == UserRole.SUPER_ADMIN

    agents_cursor = (
        store._agents.find({} if is_super else {"tenant_id": tid})
        .sort("updated_at", -1)
    )
    agents_coro       = agents_cursor.to_list(length=500)
    subscription_coro = store.get_user_subscription(uid)
    scoped_subscriptions_coro = (
        store._users.find(
            {"subscription": {"$exists": True, "$ne": None}},
            {"subscription": 1, "role": 1},
        ).to_list(length=1000)
        if is_super
        else asyncio.sleep(0, result=[])
    )
    activity_cursor   = store._activity.find(
        {} if is_super else {"tenant_id": tid},
        sort=[("timestamp", -1)],
        limit=20,
    )
    activity_coro = activity_cursor.to_list(length=20)

    agents_raw, subscription, scoped_subscription_docs, recent_activity_raw = await asyncio.gather(
        agents_coro, subscription_coro, scoped_subscriptions_coro, activity_coro
    )

    agent_ids    = [a["_id"] for a in agents_raw]
    total_agents = len(agent_ids)

    conv_match = (
        {"tenant_id": {"$in": list({a.get("tenant_id") for a in agents_raw})}}
        if is_super else {"tenant_id": tid}
    )

    conv_pipeline = [
        {"$match": conv_match},
        {"$addFields": {"message_count": {"$size": {"$ifNull": ["$messages", []]}}}},
        {"$project": {"agent_id": 1, "title": 1, "updated_at": 1, "created_at": 1, "message_count": 1}},
        {"$sort": {"updated_at": -1}},
        {"$limit": 500},
    ]

    doc_match = {"agent_id": {"$in": agent_ids}} if agent_ids else {"agent_id": "__none__"}

    convs_raw, total_documents = await asyncio.gather(
        store._conversations.aggregate(conv_pipeline).to_list(length=500),
        store._documents.count_documents(doc_match),
    )

    total_conversations = len(convs_raw)
    total_messages      = sum(c.get("message_count", 0) for c in convs_raw)
    avg_messages        = round(total_messages / total_conversations, 1) if total_conversations else 0

    now              = _now()
    active_threshold = now - timedelta(minutes=30)
    week_ago         = now - timedelta(days=7)
    month_ago        = now - timedelta(days=30)

    def parse_dt(val):
        if not val: return None
        if isinstance(val, datetime):
            return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
        try:
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except Exception:
            return None

    active_chats   = 0
    last7_convs    = 0
    last30_convs   = 0
    user_messages  = 0
    ai_messages    = 0
    hourly_counts: dict[int, int] = defaultdict(int)
    daily_counts:  dict[str, int] = defaultdict(int)

    for c in convs_raw:
        c["id"] = str(c.pop("_id", c.get("id", "")))
        updated = parse_dt(c.get("updated_at"))
        mc = c.get("message_count", 0)
        if updated:
            if updated >= active_threshold: active_chats += 1
            if updated >= week_ago:         last7_convs  += 1
            if updated >= month_ago:        last30_convs += 1
            hourly_counts[updated.hour] += 1
            daily_counts[_day_key(updated)] += 1
        user_messages += mc // 2
        ai_messages   += mc - (mc // 2)

    trend_14d = [
        {
            "date":  _day_key(d := now - timedelta(days=i)),
            "label": d.strftime("%b %d"),
            "value": daily_counts.get(_day_key(d), 0),
        }
        for i in range(13, -1, -1)
    ]

    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
    if peak_hour is not None:
        peak_label = "12 AM" if peak_hour == 0 else f"{peak_hour} AM" if peak_hour < 12 else "12 PM" if peak_hour == 12 else f"{peak_hour - 12} PM"
    else:
        peak_label = "—"

    agent_name_map = {a["_id"]: a.get("name", "Unknown agent") for a in agents_raw}
    recent_conversations = [
        {
            "id":            c["id"],
            "title":         c.get("title", "Untitled"),
            "agent_id":      c.get("agent_id", ""),
            "agent_name":    agent_name_map.get(c.get("agent_id", ""), "Unknown agent"),
            "message_count": c.get("message_count", 0),
            "updated_at":    str(c.get("updated_at", "")),
        }
        for c in convs_raw[:20]
    ]

    active_agent_ids = {
        c.get("agent_id") for c in convs_raw
        if (upd := parse_dt(c.get("updated_at"))) and upd >= week_ago
    }
    active_agents   = len(active_agent_ids)
    inactive_agents = total_agents - active_agents

    most_recent_agent_id = convs_raw[0].get("agent_id") if convs_raw else None
    most_recent_agent    = agent_name_map.get(most_recent_agent_id) if most_recent_agent_id else None

    agent_conv_counts: dict[str, int] = defaultdict(int)
    for c in convs_raw:
        agent_conv_counts[c.get("agent_id", "")] += 1
    top_agents_list = [
        {"agent_id": aid, "name": agent_name_map.get(aid, "Unknown"), "conversations": cnt}
        for aid, cnt in sorted(agent_conv_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    ]

    plan_info = None
    if subscription:
        exp       = parse_dt(subscription.get("cycle_end_date"))
        days_left = (exp - now).days if exp else None
        total_tokens     = subscription.get("chat_token_limit", 0) + subscription.get("summary_token_limit", 0)
        used_tokens      = subscription.get("chat_tokens_used", 0) + subscription.get("summary_tokens_used", 0)
        remaining_tokens = total_tokens - used_tokens
        used_pct         = round(subscription.get("used_messages", 0) / max(subscription.get("monthly_message_limit", 1), 1) * 100, 1)
        token_pct        = round(used_tokens / max(total_tokens, 1) * 100, 1)
        plan_info = {
            "plan_name":                subscription.get("plan_name", "Free"),
            "billing_status":           subscription.get("billing_status", "active"),
            "monthly_message_limit":    subscription.get("monthly_message_limit", 0),
            "used_messages":            subscription.get("used_messages", 0),
            "remaining_messages":       subscription.get("remaining_messages", 0),
            "message_used_pct":         used_pct,
            "chat_token_limit":         subscription.get("chat_token_limit", 0),
            "chat_tokens_used":         subscription.get("chat_tokens_used", 0),
            "chat_tokens_remaining":    subscription.get("chat_tokens_remaining", 0),
            "summary_token_limit":      subscription.get("summary_token_limit", 0),
            "summary_tokens_used":      subscription.get("summary_tokens_used", 0),
            "summary_tokens_remaining": subscription.get("summary_tokens_remaining", 0),
            "total_tokens":             total_tokens,
            "used_tokens":              used_tokens,
            "remaining_tokens":         remaining_tokens,
            "token_used_pct":           token_pct,
            "cycle_start_date":         subscription.get("cycle_start_date", ""),
            "cycle_end_date":           subscription.get("cycle_end_date", ""),
            "days_remaining":           days_left,
            "high_usage_warning":       used_pct >= 80 or token_pct >= 80,
            "expiry_warning":           days_left is not None and days_left <= 7,
        }

    total_chat_token_limit = 0
    total_chat_tokens_used = 0
    total_summary_token_limit = 0
    total_summary_tokens_used = 0
    total_chat_tokens_remaining = 0
    total_summary_tokens_remaining = 0

    if is_super:
        for doc in scoped_subscription_docs:
            sub = doc.get("subscription") or {}
            total_chat_token_limit += sub.get("chat_token_limit", 0)
            total_chat_tokens_used += sub.get("chat_tokens_used", 0)
            total_summary_token_limit += sub.get("summary_token_limit", 0)
            total_summary_tokens_used += sub.get("summary_tokens_used", 0)
            total_chat_tokens_remaining += sub.get("chat_tokens_remaining", 0)
            total_summary_tokens_remaining += sub.get("summary_tokens_remaining", 0)
    elif subscription:
        total_chat_token_limit = subscription.get("chat_token_limit", 0)
        total_chat_tokens_used = subscription.get("chat_tokens_used", 0)
        total_summary_token_limit = subscription.get("summary_token_limit", 0)
        total_summary_tokens_used = subscription.get("summary_tokens_used", 0)
        total_chat_tokens_remaining = subscription.get("chat_tokens_remaining", 0)
        total_summary_tokens_remaining = subscription.get("summary_tokens_remaining", 0)

    token_total = total_chat_token_limit + total_summary_token_limit
    token_used = total_chat_tokens_used + total_summary_tokens_used
    token_remaining = max(token_total - token_used, 0)
    token_summary = {
        "chat_token_limit": total_chat_token_limit,
        "chat_tokens_used": total_chat_tokens_used,
        "chat_tokens_remaining": total_chat_tokens_remaining,
        "summary_token_limit": total_summary_token_limit,
        "summary_tokens_used": total_summary_tokens_used,
        "summary_tokens_remaining": total_summary_tokens_remaining,
        "total_tokens": token_total,
        "used_tokens": token_used,
        "remaining_tokens": token_remaining,
        "token_used_pct": round(token_used / max(token_total, 1) * 100, 1),
        "scope": "system" if is_super else "user",
    }

    recent_activity = [
        {
            "id":          str(a.pop("_id", "")),
            "type":        a.get("type", ""),
            "description": a.get("description", ""),
            "agent_id":    a.get("agent_id", ""),
            "timestamp":   str(a.get("timestamp", "")),
        }
        for a in recent_activity_raw
    ]

    return {
        "role": user.role,
        "kpis": {
            "total_agents":         total_agents,
            "total_documents":      total_documents,
            "total_conversations":  total_conversations,
            "active_chats":         active_chats,
            "total_messages":       total_messages,
            "avg_messages":         avg_messages,
            "last7_conversations":  last7_convs,
            "last30_conversations": last30_convs,
            "user_messages":        user_messages,
            "ai_messages":          ai_messages,
        },
        "plan":                   plan_info,
        "trend_14d":              trend_14d,
        "peak_hour":              peak_label,
        "token_summary":          token_summary,
        "recent_conversations":   recent_conversations,
        "agents_summary": {
            "total":     total_agents,
            "active":    active_agents,
            "inactive":  inactive_agents,
            "most_recent": most_recent_agent,
            "top_agents":  top_agents_list,
        },
        "recent_activity": recent_activity,
    }
