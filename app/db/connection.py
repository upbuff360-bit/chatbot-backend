from __future__ import annotations

import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        raise RuntimeError("MongoDB client not initialised. Call connect() first.")
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[os.environ["MONGO_DB_NAME"]]


async def connect() -> None:
    global _client
    uri = os.environ["MONGO_URI"]
    _client = AsyncIOMotorClient(uri)
    # Verify connection
    await _client.admin.command("ping")


async def disconnect() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def create_indexes() -> None:
    db = get_database()

    # users — unique email, tenant scoping
    await db.users.create_index("email", unique=True)
    await db.users.create_index("tenant_id")

    # agents — tenant + user scoping
    await db.agents.create_index("tenant_id")
    await db.agents.create_index("user_id")
    await db.agents.create_index("shared_with_user_ids")
    await db.agents.create_index([("tenant_id", 1), ("user_id", 1)])

    # documents — agent + tenant scoping
    await db.documents.create_index("agent_id")
    await db.documents.create_index("tenant_id")
    await db.documents.create_index([("agent_id", 1), ("tenant_id", 1)])

    # conversations — agent scoping
    await db.conversations.create_index("agent_id")
    await db.conversations.create_index("tenant_id")
    await db.conversations.create_index([("agent_id", 1), ("tenant_id", 1)])

    # leads — tenant + agent + conversation scoping
    await db.leads.create_index("agent_id")
    await db.leads.create_index("tenant_id")
    await db.leads.create_index([("tenant_id", 1), ("agent_id", 1), ("created_at", -1)])
    await db.leads.create_index([("tenant_id", 1), ("agent_id", 1), ("conversation_id", 1)], unique=True)
    await db.leads.create_index([("tenant_id", 1), ("agent_id", 1), ("email_normalized", 1)], sparse=True)
    await db.leads.create_index([("tenant_id", 1), ("agent_id", 1), ("phone_normalized", 1)], sparse=True)

    # crawl_jobs
    await db.crawl_jobs.create_index("agent_id")
    await db.crawl_jobs.create_index("tenant_id")

    # agent invitations
    await db.agent_invitations.create_index("token", unique=True)
    await db.agent_invitations.create_index([("invited_email", 1), ("status", 1)])
    await db.agent_invitations.create_index([("agent_id", 1), ("owner_tenant_id", 1), ("status", 1)])
    await db.agent_invitations.create_index("expires_at", expireAfterSeconds=0)

    # password resets
    await db.password_resets.create_index("token", unique=True)
    await db.password_resets.create_index("email")
    await db.password_resets.create_index("expires_at", expireAfterSeconds=0)

    # activity — required for dashboard_summary (tenant filter + timestamp sort)
    await db.activity.create_index("tenant_id")
    await db.activity.create_index([("tenant_id", 1), ("timestamp", -1)])

    # fallback_logs — required for analytics (agent + tenant filter + created_at sort)
    await db.fallback_logs.create_index([("agent_id", 1), ("tenant_id", 1)])
    await db.fallback_logs.create_index([("agent_id", 1), ("tenant_id", 1), ("created_at", -1)])
