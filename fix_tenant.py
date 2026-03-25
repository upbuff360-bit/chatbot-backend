"""
fix_tenant.py — reassigns all agents, documents and conversations
to the admin@chatbot.com user's tenant_id.

Run from chatbot/ project root:
    python fix_tenant.py
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

TARGET_EMAIL = os.environ.get("MIGRATE_OWNER_EMAIL", "admin@chatbot.com")


async def main():
    client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]

    # Find the target user
    user = await db.users.find_one({"email": TARGET_EMAIL})
    if not user:
        print(f"ERROR: User '{TARGET_EMAIL}' not found in MongoDB.")
        client.close()
        return

    tenant_id = user["tenant_id"]
    user_id   = user["_id"]
    print(f"\nTarget user  : {TARGET_EMAIL}")
    print(f"tenant_id    : {tenant_id}")
    print(f"user_id      : {user_id}")

    # Update all agents
    result = await db.agents.update_many(
        {},
        {"$set": {"tenant_id": tenant_id, "user_id": user_id}}
    )
    print(f"\nAgents updated    : {result.modified_count}")

    # Update all documents
    result = await db.documents.update_many(
        {},
        {"$set": {"tenant_id": tenant_id, "user_id": user_id}}
    )
    print(f"Documents updated : {result.modified_count}")

    # Update all conversations
    result = await db.conversations.update_many(
        {},
        {"$set": {"tenant_id": tenant_id, "user_id": user_id}}
    )
    print(f"Conversations updated : {result.modified_count}")

    # Verify
    agent_count = await db.agents.count_documents({"tenant_id": tenant_id})
    print(f"\nAgents now visible to '{TARGET_EMAIL}': {agent_count}")
    print("Done! Refresh the frontend — all agents should appear now.")

    client.close()


asyncio.run(main())
