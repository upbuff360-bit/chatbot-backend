"""
migrate.py — One-time migration from state.json flat file → MongoDB

Run from your chatbot/ project root:
    python migrate.py

What it does:
    1. Reads persist/admin/state.json
    2. Creates one MongoDB user (owner) for the whole workspace
    3. Migrates all agents, their settings, documents, and conversations
    4. Prints a summary at the end

After running:
    - Use the email/password printed at the end to log in
    - All your existing data is preserved in MongoDB
    - The original state.json is NOT deleted (kept as backup)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# ── Validate env ──────────────────────────────────────────────────────────────
MONGO_URI    = os.environ.get("MONGO_URI", "")
MONGO_DB     = os.environ.get("MONGO_DB_NAME", "chatbot_saas")
STATE_PATH   = Path(os.environ.get("STATE_JSON_PATH", "./persist/admin/state.json"))

if not MONGO_URI:
    print("ERROR: MONGO_URI not set in .env")
    sys.exit(1)

if not STATE_PATH.exists():
    print(f"ERROR: state.json not found at {STATE_PATH}")
    print("Set STATE_JSON_PATH in .env if it's in a different location.")
    sys.exit(1)

# ── Migration owner account ───────────────────────────────────────────────────
# Change these before running if you want a specific email/password
OWNER_EMAIL    = os.environ.get("MIGRATE_OWNER_EMAIL", "admin@chatbot.local")
OWNER_PASSWORD = os.environ.get("MIGRATE_OWNER_PASSWORD", "changeme123")


async def run_migration():
    from motor.motor_asyncio import AsyncIOMotorClient
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]

    print(f"\n{'='*55}")
    print("  Chatbot SaaS — state.json → MongoDB Migration")
    print(f"{'='*55}")
    print(f"  Source : {STATE_PATH}")
    print(f"  Target : {MONGO_URI} / {MONGO_DB}")
    print(f"{'='*55}\n")

    # ── Load state.json ───────────────────────────────────────────────────────
    with STATE_PATH.open("r", encoding="utf-8") as f:
        state = json.load(f)

    agents_in_file = state.get("agents", [])
    print(f"Found {len(agents_in_file)} agent(s) in state.json\n")

    # ── Create or reuse owner user ────────────────────────────────────────────
    tenant_id = str(uuid4())
    existing_user = await db.users.find_one({"email": OWNER_EMAIL})

    if existing_user:
        print(f"[SKIP] User '{OWNER_EMAIL}' already exists — reusing tenant_id")
        user_id   = existing_user["_id"]
        tenant_id = existing_user["tenant_id"]
    else:
        user_id = str(uuid4())
        await db.users.insert_one({
            "_id"            : user_id,
            "email"          : OWNER_EMAIL,
            "hashed_password": pwd_context.hash(OWNER_PASSWORD),
            "tenant_id"      : tenant_id,
            "role"           : "customer",
            "plan"           : "free",
            "created_at"     : _now(),
            "updated_at"     : _now(),
        })
        print(f"[OK] Created owner user: {OWNER_EMAIL}")

    print(f"     tenant_id : {tenant_id}\n")

    # ── Migrate agents ────────────────────────────────────────────────────────
    agents_migrated      = 0
    documents_migrated   = 0
    conversations_migrated = 0

    for agent in agents_in_file:
        old_agent_id = agent["id"]
        agent_name   = agent.get("name", "Unnamed agent")

        # Check if already migrated (idempotent)
        existing = await db.agents.find_one({
            "tenant_id": tenant_id,
            "name"     : agent_name,
        })
        if existing:
            print(f"[SKIP] Agent '{agent_name}' already in MongoDB")
            new_agent_id = existing["_id"]
        else:
            new_agent_id = str(uuid4())
            settings     = agent.get("settings", {})

            # Ensure all new settings fields have defaults
            settings.setdefault("display_name",    "")
            settings.setdefault("website_name",     "")
            settings.setdefault("website_url",      "")
            settings.setdefault("primary_color",    "#0f172a")
            settings.setdefault("secondary_color",  "#f8fafc")
            settings.setdefault("appearance",       "light")

            await db.agents.insert_one({
                "_id"               : new_agent_id,
                "tenant_id"         : tenant_id,
                "user_id"           : user_id,
                "name"              : agent_name,
                "settings"          : settings,
                "document_count"    : len(agent.get("documents", [])),
                "conversation_count": len(agent.get("conversations", [])),
                "created_at"        : _parse_date(agent.get("created_at")),
                "updated_at"        : _now(),
            })
            agents_migrated += 1
            print(f"[OK] Migrated agent: '{agent_name}' → {new_agent_id}")

        # ── Migrate documents ─────────────────────────────────────────────────
        for doc in agent.get("documents", []):
            existing_doc = await db.documents.find_one({
                "agent_id"  : new_agent_id,
                "file_name" : doc.get("file_name"),
                "source_type": doc.get("source_type", "pdf"),
            })
            if existing_doc:
                continue

            await db.documents.insert_one({
                "_id"        : str(uuid4()),
                "tenant_id"  : tenant_id,
                "agent_id"   : new_agent_id,
                "user_id"    : user_id,
                "file_name"  : doc.get("file_name", ""),
                "source_type": doc.get("source_type", "pdf"),
                "status"     : doc.get("status", "indexed"),
                "source_url" : doc.get("source_url"),
                "content"    : doc.get("content"),
                "question"   : doc.get("question"),
                "answer"     : doc.get("answer"),
                "page_count" : doc.get("page_count"),
                "page_urls"  : doc.get("page_urls"),
                "uploaded_at": _parse_date(doc.get("uploaded_at")),
            })
            documents_migrated += 1

        print(f"     └─ {len(agent.get('documents', []))} document(s) migrated")

        # ── Migrate conversations ─────────────────────────────────────────────
        for conv in agent.get("conversations", []):
            existing_conv = await db.conversations.find_one({
                "agent_id": new_agent_id,
                "title"   : conv.get("title"),
            })
            if existing_conv:
                continue

            messages = [
                {
                    "id"       : msg.get("id", str(uuid4())),
                    "role"     : msg.get("role", "user"),
                    "content"  : msg.get("content", ""),
                    "timestamp": _parse_date(msg.get("timestamp")),
                }
                for msg in conv.get("messages", [])
            ]

            await db.conversations.insert_one({
                "_id"       : str(uuid4()),
                "tenant_id" : tenant_id,
                "agent_id"  : new_agent_id,
                "user_id"   : user_id,
                "title"     : conv.get("title", "Conversation"),
                "messages"  : messages,
                "created_at": _parse_date(conv.get("created_at")),
                "updated_at": _parse_date(conv.get("updated_at")),
            })
            conversations_migrated += 1

        print(f"     └─ {len(agent.get('conversations', []))} conversation(s) migrated")

    # ── Create indexes ────────────────────────────────────────────────────────
    await db.users.create_index("email", unique=True)
    await db.users.create_index("tenant_id")
    await db.agents.create_index("tenant_id")
    await db.agents.create_index([("tenant_id", 1), ("user_id", 1)])
    await db.documents.create_index([("agent_id", 1), ("tenant_id", 1)])
    await db.conversations.create_index([("agent_id", 1), ("tenant_id", 1)])

    client.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Migration complete!")
    print(f"{'='*55}")
    print(f"  Agents       : {agents_migrated} migrated")
    print(f"  Documents    : {documents_migrated} migrated")
    print(f"  Conversations: {conversations_migrated} migrated")
    print(f"\n  Login credentials:")
    print(f"  Email    : {OWNER_EMAIL}")
    print(f"  Password : {OWNER_PASSWORD}")
    print(f"\n  ⚠  Change your password after first login!")
    print(f"{'='*55}\n")


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _parse_date(value):
    from datetime import datetime, timezone
    if not value:
        return _now()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        return _now()


if __name__ == "__main__":
    asyncio.run(run_migration())
