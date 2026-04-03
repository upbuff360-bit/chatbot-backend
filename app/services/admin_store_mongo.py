from __future__ import annotations

"""
AdminStoreMongo — drop-in replacement for AdminStore.

All methods are async and require (tenant_id, ...) so that every query
is scoped to the correct tenant. The class holds no state — all reads
and writes go directly to MongoDB via the shared motor client.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.permissions_registry import (
    RESOURCE_PERMISSION_ALIASES,
    is_visible_permission_resource,
    normalize_permission_name,
    normalize_permission_names,
    normalize_permission_resource,
)
from app.models.agent import AgentSettings, DEFAULT_SYSTEM_PROMPT
from app.models.document import ConversationMessage


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


def _normalize_email(value: str) -> str:
    return value.strip().lower()


class AdminStoreMongo:
    def __init__(self, db: AsyncIOMotorDatabase, agents_root: str | Path = "./data/agents") -> None:
        self.db = db
        self.agents_root = Path(agents_root)

        # Collection handles — treated like table references
        self._users = db.users
        self._agents = db.agents
        self._documents = db.documents
        self._conversations = db.conversations
        self._activity = db.activity
        self._fallback_logs = db.fallback_logs
        self._agent_invitations = db.agent_invitations
        self._roles       = db.roles
        self._permissions = db.permissions

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _agent_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id

    def get_agent_pdf_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "pdfs"

    def get_agent_website_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "websites"

    def get_agent_snippet_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "text_snippets"

    def get_agent_qa_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "qa"

    async def _log_activity(
        self,
        tenant_id: str,
        agent_id: str,
        activity_type: str,
        description: str,
    ) -> None:
        await self._activity.insert_one({
            "_id": _new_id(),
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "type": activity_type,
            "description": description,
            "timestamp": _now(),
        })

    # ── Users ─────────────────────────────────────────────────────────────────

    async def create_user(
        self,
        email: str,
        hashed_password: str,
        tenant_id: str,
        role: str = "customer",
        plan: str = "free",
        name: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        doc = {
            "_id": _new_id(),
            "email": email,
            "name": name or email.split("@")[0],
            "hashed_password": hashed_password,
            "tenant_id": tenant_id,
            "role": role,
            "plan": plan,
            "created_at": now,
            "updated_at": now,
        }
        await self._users.insert_one(doc)
        return doc

    async def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        doc = await self._users.find_one({"email": email})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def get_user_by_id(self, user_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        doc = await self._users.find_one({"_id": user_id, "tenant_id": tenant_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_all_users(self, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        """List all users across all tenants — super admin only. Paginated via skip/limit."""
        cursor = (
            self._users.find({}, {"hashed_password": 0})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def update_user(
        self,
        user_id: str,
        email: str | None = None,
        role: str | None = None,
        plan: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        user = await self._users.find_one({"_id": user_id})
        if not user:
            raise KeyError(f"User '{user_id}' not found.")
        if user.get("role") == "super_admin":
            raise ValueError("Cannot modify a super admin user.")
        update: dict = {"updated_at": _now()}
        if email:
            update["email"] = email
        if name:
            update["name"] = name
        if role:
            update["role"] = role
        if plan:
            update["plan"] = plan
        doc = await self._users.find_one_and_update(
            {"_id": user_id},
            {"$set": update},
            return_document=True,
        )
        doc.pop("hashed_password", None)
        doc["id"] = doc.pop("_id")
        return doc

    async def delete_user(self, user_id: str) -> None:
        user = await self._users.find_one({"_id": user_id})
        if not user:
            raise KeyError(f"User '{user_id}' not found.")
        if user.get("role") == "super_admin":
            raise ValueError("Cannot delete a super admin user.")
        await self._users.delete_one({"_id": user_id})

    async def admin_create_user(
        self,
        email: str,
        hashed_password: str,
        role: str,
        plan: str = "free",
    ) -> dict[str, Any]:
        """Create a user under super admin — assigns a new tenant_id."""
        from uuid import uuid4
        existing = await self.get_user_by_email(email)
        if existing:
            raise ValueError(f"User with email '{email}' already exists.")
        tenant_id = str(uuid4())
        doc = await self.create_user(
            email=email,
            hashed_password=hashed_password,
            tenant_id=tenant_id,
            role=role,
            plan=plan,
        )
        doc.pop("hashed_password", None)
        doc["id"] = doc.pop("_id")
        return doc

    # ── Agents ────────────────────────────────────────────────────────────────

    async def list_all_agents(self, limit: int = 2000) -> list[dict[str, Any]]:
        """List all agents across all tenants. Capped at `limit` to prevent unbounded memory use."""
        cursor = self._agents.find({}, sort=[("created_at", -1)])
        docs = await cursor.to_list(length=limit)
        result = []
        for d in docs:
            d["id"] = d.pop("_id")
            result.append(d)
        return result

    async def list_agents(self, tenant_id: str, user_id: str) -> list[dict[str, Any]]:
        cursor = self._agents.find(
            {
                "$or": [
                    {"tenant_id": tenant_id},
                    {"shared_with_user_ids": user_id},
                ]
            },
            sort=[("created_at", -1)],
        )
        agents = await cursor.to_list(length=1000)
        for a in agents:
            a["can_manage"] = a.get("tenant_id") == tenant_id
            a["is_shared"] = a.get("tenant_id") != tenant_id
            a["id"] = a.pop("_id")
        return agents

    async def get_agent(self, agent_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        doc = await self._agents.find_one({"_id": agent_id, "tenant_id": tenant_id})
        if doc:
            doc["can_manage"] = True
            doc["is_shared"] = False
            doc["id"] = doc.pop("_id")
        return doc

    async def get_accessible_agent(self, agent_id: str, tenant_id: str, user_id: str) -> Optional[dict[str, Any]]:
        doc = await self._agents.find_one(
            {
                "_id": agent_id,
                "$or": [
                    {"tenant_id": tenant_id},
                    {"shared_with_user_ids": user_id},
                ],
            }
        )
        if doc:
            doc["can_manage"] = doc.get("tenant_id") == tenant_id
            doc["is_shared"] = doc.get("tenant_id") != tenant_id
            doc["id"] = doc.pop("_id")
        return doc

    async def get_agent_by_id(self, agent_id: str) -> dict[str, Any]:
        """Fetch agent by ID only — no tenant restriction (super admin use)."""
        doc = await self._agents.find_one({"_id": agent_id})
        if not doc:
            raise KeyError(f"Agent '{agent_id}' not found.")
        doc["id"] = doc.pop("_id")
        return doc
        return doc

    async def require_agent(self, agent_id: str, tenant_id: str) -> dict[str, Any]:
        doc = await self.get_agent(agent_id, tenant_id)
        if doc is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        return doc

    async def require_accessible_agent(self, agent_id: str, tenant_id: str, user_id: str) -> dict[str, Any]:
        doc = await self.get_accessible_agent(agent_id, tenant_id, user_id)
        if doc is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        return doc

    async def require_agent_any_tenant(self, agent_id: str) -> dict[str, Any]:
        """Fetch agent by ID only — no tenant restriction."""
        doc = await self._agents.find_one({"_id": agent_id})
        if doc is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        doc["id"] = doc.pop("_id")
        return doc

    async def backfill_agent_display_ids(self) -> None:
        """One-time migration: assign display_id (agt-101 format) to agents that don't have one."""
        cursor = self._agents.find({"display_id": {"$exists": False}}, {"_id": 1}).sort("created_at", 1)
        agents = await cursor.to_list(length=None)
        for i, agent in enumerate(agents, start=101):
            await self._agents.update_one(
                {"_id": agent["_id"]},
                {"$set": {"display_id": f"agt-{i}"}}
            )

    async def _new_agent_id(self) -> str:
        """Generate a short sequential agent ID like agt-101, agt-102..."""
        # Fetch all agt- IDs and find the max numerically
        cursor = self._agents.find(
            {"_id": {"$regex": "^agt-"}},
            {"_id": 1}
        )
        docs = await cursor.to_list(length=None)
        max_num = 100
        for doc in docs:
            try:
                num = int(doc["_id"].split("-")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
        return f"agt-{max_num + 1}"

    async def create_agent(self, name: str, tenant_id: str, user_id: str) -> dict[str, Any]:
        settings = AgentSettings()
        now = _now()
        agent_id = await self._new_agent_id()
        doc = {
            "_id": agent_id,
            "display_id": agent_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "name": name.strip(),
            "shared_with_user_ids": [],
            "settings": settings.model_dump(),
            "document_count": 0,
            "conversation_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        await self._agents.insert_one(doc)
        await self._log_activity(tenant_id, doc["_id"], "agent_created", f"Created agent '{name}'.")

        # ensure filesystem dirs exist
        for sub in ["pdfs", "websites", "text_snippets", "qa"]:
            (self._agent_dir(doc["_id"]) / sub).mkdir(parents=True, exist_ok=True)

        doc["id"] = doc.pop("_id")
        return doc

    async def update_agent(self, agent_id: str, tenant_id: str, name: str) -> dict[str, Any]:
        result = await self._agents.find_one_and_update(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$set": {"name": name.strip(), "updated_at": _now()}},
            return_document=True,
        )
        if result is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        result["id"] = result.pop("_id")
        return result

    async def search_users_by_email(
        self,
        query: str,
        *,
        exclude_user_id: str | None = None,
        exclude_tenant_id: str | None = None,
        exclude_agent_id: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        pattern = query.strip()
        if not pattern:
            return []

        mongo_query: dict[str, Any] = {
            "email": {"$regex": pattern, "$options": "i"},
            "role": {"$ne": "super_admin"},
        }
        if exclude_user_id:
            mongo_query["_id"] = {"$ne": exclude_user_id}

        cursor = self._users.find(
            mongo_query,
            {"email": 1, "name": 1, "tenant_id": 1},
        ).sort("email", 1).limit(limit)
        users = await cursor.to_list(length=limit)

        shared_user_ids: set[str] = set()
        if exclude_agent_id:
            agent_doc = await self._agents.find_one({"_id": exclude_agent_id}, {"shared_with_user_ids": 1})
            if agent_doc:
                shared_user_ids = set(agent_doc.get("shared_with_user_ids", []))

        results: list[dict[str, Any]] = []
        for user in users:
            user_id = str(user["_id"])
            if exclude_tenant_id and user.get("tenant_id") == exclude_tenant_id:
                continue
            if user_id in shared_user_ids:
                continue
            results.append({
                "id": user_id,
                "email": user["email"],
                "name": user.get("name"),
            })
        return results

    async def create_agent_invitation(
        self,
        *,
        agent_id: str,
        owner_tenant_id: str,
        invited_by_user_id: str,
        invited_email: str,
    ) -> dict[str, Any]:
        normalized_email = _normalize_email(invited_email)
        existing_user = await self._users.find_one(
            {"email": normalized_email},
            {"email": 1, "tenant_id": 1, "role": 1},
        )
        if existing_user:
            if existing_user.get("role") == "super_admin":
                raise ValueError("Super admin accounts cannot be invited to shared agents.")
            if existing_user.get("tenant_id") == owner_tenant_id:
                raise ValueError("This user already has access to the agent in the same workspace.")
            raise ValueError("This email is already registered. Select the existing account from the dropdown.")

        agent = await self._agents.find_one({"_id": agent_id, "tenant_id": owner_tenant_id}, {"name": 1})
        if not agent:
            raise KeyError(f"Agent '{agent_id}' not found.")

        now = _now()
        expires_at = now + timedelta(days=7)
        token = secrets.token_urlsafe(32)
        existing_invite = await self._agent_invitations.find_one(
            {
                "agent_id": agent_id,
                "owner_tenant_id": owner_tenant_id,
                "invited_email": normalized_email,
                "status": "pending",
            }
        )
        if existing_invite:
            invite_id = existing_invite["_id"]
            await self._agent_invitations.update_one(
                {"_id": invite_id},
                {
                    "$set": {
                        "token": token,
                        "invited_by_user_id": invited_by_user_id,
                        "updated_at": now,
                        "expires_at": expires_at,
                    }
                },
            )
        else:
            invite_id = _new_id()
            await self._agent_invitations.insert_one(
                {
                    "_id": invite_id,
                    "agent_id": agent_id,
                    "owner_tenant_id": owner_tenant_id,
                    "invited_by_user_id": invited_by_user_id,
                    "invited_email": normalized_email,
                    "token": token,
                    "status": "pending",
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": expires_at,
                }
            )

        await self._log_activity(
            owner_tenant_id,
            agent_id,
            "agent_invited",
            f"Sent an agent invite for '{agent.get('name', agent_id)}' to {normalized_email}.",
        )
        return {
            "id": invite_id,
            "token": token,
            "agent_id": agent_id,
            "agent_name": agent.get("name", agent_id),
            "invited_email": normalized_email,
            "expires_at": expires_at,
        }

    async def delete_agent_invitation(self, invitation_id: str) -> None:
        await self._agent_invitations.delete_one({"_id": invitation_id})

    async def get_agent_invitation_by_token(self, token: str) -> Optional[dict[str, Any]]:
        invite = await self._agent_invitations.find_one({"token": token, "status": "pending"})
        if not invite:
            return None
        if invite.get("expires_at") and invite["expires_at"].replace(tzinfo=timezone.utc) <= _now():
            await self._agent_invitations.update_one(
                {"_id": invite["_id"]},
                {"$set": {"status": "expired", "updated_at": _now()}},
            )
            return None

        agent = await self._agents.find_one({"_id": invite["agent_id"]}, {"name": 1})
        inviter = await self._users.find_one({"_id": invite["invited_by_user_id"]}, {"email": 1})
        invite["agent_name"] = agent.get("name", invite["agent_id"]) if agent else invite["agent_id"]
        invite["inviter_email"] = inviter.get("email") if inviter else None
        invite["id"] = str(invite.pop("_id"))
        return invite

    async def accept_pending_agent_invites(self, email: str, user_id: str) -> int:
        normalized_email = _normalize_email(email)
        invites = await self._agent_invitations.find(
            {
                "invited_email": normalized_email,
                "status": "pending",
                "expires_at": {"$gt": _now()},
            }
        ).to_list(length=100)
        if not invites:
            return 0

        accepted = 0
        for invite in invites:
            agent = await self._agents.find_one({"_id": invite["agent_id"]}, {"name": 1, "shared_with_user_ids": 1})
            if not agent:
                await self._agent_invitations.update_one(
                    {"_id": invite["_id"]},
                    {"$set": {"status": "cancelled", "updated_at": _now()}},
                )
                continue

            await self._agents.update_one(
                {"_id": invite["agent_id"]},
                {
                    "$addToSet": {"shared_with_user_ids": user_id},
                    "$set": {"updated_at": _now()},
                },
            )
            await self._agent_invitations.update_one(
                {"_id": invite["_id"]},
                {
                    "$set": {
                        "status": "accepted",
                        "accepted_by_user_id": user_id,
                        "accepted_at": _now(),
                        "updated_at": _now(),
                    }
                },
            )
            await self._log_activity(
                invite["owner_tenant_id"],
                invite["agent_id"],
                "agent_invite_accepted",
                f"{normalized_email} accepted the invite for '{agent.get('name', invite['agent_id'])}'.",
            )
            accepted += 1

        return accepted

    async def share_agent(
        self,
        agent_id: str,
        owner_tenant_id: str,
        target_user_id: str,
    ) -> dict[str, Any]:
        target_user = await self._users.find_one({"_id": target_user_id}, {"email": 1, "tenant_id": 1, "role": 1})
        if not target_user:
            raise KeyError(f"User '{target_user_id}' not found.")
        if target_user.get("role") == "super_admin":
            raise ValueError("Super admin accounts cannot be invited to shared agents.")

        if target_user.get("tenant_id") == owner_tenant_id:
            raise ValueError("This user already has access to the agent in the same workspace.")

        agent = await self._agents.find_one({"_id": agent_id, "tenant_id": owner_tenant_id})
        if not agent:
            raise KeyError(f"Agent '{agent_id}' not found.")

        if target_user_id in agent.get("shared_with_user_ids", []):
            raise ValueError("This agent has already been shared with that user.")

        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": owner_tenant_id},
            {
                "$addToSet": {"shared_with_user_ids": target_user_id},
                "$set": {"updated_at": _now()},
            },
        )
        await self._log_activity(
            owner_tenant_id,
            agent_id,
            "agent_shared",
            f"Shared agent '{agent.get('name', agent_id)}' with {target_user.get('email', 'a user')}.",
        )
        return {
            "user_id": target_user_id,
            "email": target_user["email"],
        }

    async def delete_agent(self, agent_id: str, tenant_id: str) -> None:
        import shutil
        from app.main import agent_collection_name

        # 1. Delete agent record
        await self._agents.delete_one({"_id": agent_id, "tenant_id": tenant_id})

        # 2. Delete all documents metadata
        await self._documents.delete_many({"agent_id": agent_id, "tenant_id": tenant_id})

        # 3. Delete all conversations
        await self._conversations.delete_many({"agent_id": agent_id, "tenant_id": tenant_id})

        # 4. Delete activity logs
        await self._activity.delete_many({"agent_id": agent_id, "tenant_id": tenant_id})

        # 5. Delete fallback logs
        await self._fallback_logs.delete_many({"agent_id": agent_id, "tenant_id": tenant_id})

        # 6. Delete MongoDB chunks via ChunkStore
        try:
            from app.main import chunk_store as cs
            if cs:
                await cs.delete_chunks_by_agent(agent_id)
        except Exception:
            pass

        # 7. Delete Qdrant vector collection
        try:
            from app.vector_store import VectorStore
            collection = agent_collection_name(agent_id)
            vs = VectorStore(collection_name=collection)
            vs.delete_collection()
        except Exception:
            pass

        # 8. Delete agent data folder from disk
        try:
            agent_dir = self._agent_dir(agent_id)
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
        except Exception:
            pass
        await self._log_activity(tenant_id, agent_id, "agent_deleted", f"Deleted agent '{agent_id}'.")

    # ── Documents (knowledge sources) ─────────────────────────────────────────

    async def list_documents(
        self,
        agent_id: str,
        tenant_id: str,
        source_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query: dict = {"agent_id": agent_id, "tenant_id": tenant_id}
        if source_type:
            query["source_type"] = source_type
        cursor = self._documents.find(query, sort=[("uploaded_at", -1)])
        docs = await cursor.to_list(length=limit)
        for d in docs:
            d["id"] = d.pop("_id")
        return docs

    async def mark_document_uploaded(
        self,
        agent_id: str,
        tenant_id: str,
        user_id: str,
        file_name: str,
        source_type: str,
        status: str = "indexed",
        source_url: Optional[str] = None,
    ) -> dict[str, Any]:
        # Upsert — if the same file_name + source_type exists, update it
        now = _now()
        existing = await self._documents.find_one({
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "file_name": file_name,
            "source_type": source_type,
        })
        if existing:
            await self._documents.update_one(
                {"_id": existing["_id"]},
                {"$set": {"status": status, "source_url": source_url}},
            )
            existing["status"] = status
            existing["id"] = existing.pop("_id")
            return existing

        doc = {
            "_id": _new_id(),
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "file_name": file_name,
            "source_type": source_type,
            "status": status,
            "source_url": source_url,
            "uploaded_at": now,
        }
        await self._documents.insert_one(doc)

        # bump document_count on agent
        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$inc": {"document_count": 1}, "$set": {"updated_at": now}},
        )
        await self._log_activity(
            tenant_id, agent_id, "document_uploaded", f"Uploaded '{file_name}'."
        )
        doc["id"] = doc.pop("_id")
        return doc

    async def upsert_website_source(
        self,
        agent_id: str,
        tenant_id: str,
        user_id: str,
        display_name: str,
        source_url: str,
        status: str = "indexed",
        page_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Upsert a website source document in MongoDB.

        page_urls: persisted in MongoDB so the Knowledge tab can display
        crawled pages even after Cloud Run restarts wipe the ephemeral disk.
        """
        now = _now()
        existing = await self._documents.find_one({
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "source_type": "website",
            "source_url": source_url,
        })

        update_fields: dict = {"file_name": display_name, "status": status}
        if page_urls is not None:
            update_fields["page_urls"] = page_urls

        if existing:
            await self._documents.update_one(
                {"_id": existing["_id"]},
                {"$set": update_fields},
            )
            existing.update(update_fields)
            existing["id"] = existing.pop("_id")
            return existing

        doc = {
            "_id": _new_id(),
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "file_name": display_name,
            "source_type": "website",
            "source_url": source_url,
            "status": status,
            "uploaded_at": now,
        }
        if page_urls is not None:
            doc["page_urls"] = page_urls
        await self._documents.insert_one(doc)
        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$inc": {"document_count": 1}, "$set": {"updated_at": now}},
        )
        doc["id"] = doc.pop("_id")
        return doc

    async def get_document(
        self, document_id: str, agent_id: str, tenant_id: str
    ) -> Optional[dict[str, Any]]:
        doc = await self._documents.find_one({
            "_id": document_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        if doc:
            doc["id"] = doc.pop("_id")
        return doc

    async def update_document(
        self,
        document_id: str,
        agent_id: str,
        tenant_id: str,
        file_name: str,
        source_url: Optional[str] = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {"file_name": file_name}
        if source_url is not None:
            updates["source_url"] = source_url
        result = await self._documents.find_one_and_update(
            {"_id": document_id, "agent_id": agent_id, "tenant_id": tenant_id},
            {"$set": updates},
            return_document=True,
        )
        if result is None:
            raise KeyError(f"Document '{document_id}' not found.")
        result["id"] = result.pop("_id")
        return result

    async def delete_document(
        self, document_id: str, agent_id: str, tenant_id: str
    ) -> dict[str, Any]:
        doc = await self._documents.find_one_and_delete({
            "_id": document_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        if doc is None:
            raise KeyError(f"Document '{document_id}' not found.")

        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$inc": {"document_count": -1}, "$set": {"updated_at": _now()}},
        )
        await self._log_activity(
            tenant_id, agent_id, "document_deleted", f"Deleted '{doc['file_name']}'."
        )
        doc["id"] = doc.pop("_id")
        return doc

    async def sync_documents(
        self, agent_id: str, tenant_id: str, user_id: str, file_names: list[str]
    ) -> list[dict[str, Any]]:
        """Reconcile PDF files on disk with the documents collection."""
        existing_cursor = self._documents.find({
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "source_type": "pdf",
        })
        existing = {d["file_name"]: d async for d in existing_cursor}

        for file_name in file_names:
            if file_name not in existing:
                await self.mark_document_uploaded(
                    agent_id, tenant_id, user_id, file_name, "pdf"
                )

        # update document_count to actual count
        count = await self._documents.count_documents({
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$set": {"document_count": count, "updated_at": _now()}},
        )

        return await self.list_documents(agent_id, tenant_id)

    # ── Agent Settings ────────────────────────────────────────────────────────

    async def get_settings(self, agent_id: str, tenant_id: str) -> dict[str, Any]:
        doc = await self._agents.find_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"settings": 1},
        )
        if doc is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        return doc.get("settings", AgentSettings().model_dump())

    async def update_settings(
        self, agent_id: str, tenant_id: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        result = await self._agents.find_one_and_update(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$set": {"settings": settings, "updated_at": _now()}},
            return_document=True,
            projection={"settings": 1},
        )
        if result is None:
            raise KeyError(f"Agent '{agent_id}' not found.")
        await self._log_activity(
            tenant_id, agent_id, "settings_updated", "Updated agent settings."
        )
        return result["settings"]

    # ── Conversations ─────────────────────────────────────────────────────────

    async def append_conversation_messages(
        self,
        agent_id: str,
        tenant_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        now = _now()
        new_messages = [
            {"id": _new_id(), "role": "user", "content": user_message, "timestamp": now},
            {"id": _new_id(), "role": "assistant", "content": assistant_message, "timestamp": _now()},
        ]

        if conversation_id:
            result = await self._conversations.find_one_and_update(
                {"_id": conversation_id, "agent_id": agent_id, "tenant_id": tenant_id},
                {
                    "$push": {"messages": {"$each": new_messages}},
                    "$set": {"updated_at": _now(), "summary_stale": True},
                },
                return_document=True,
            )
            if result:
                result["id"] = result.pop("_id")
                return result

        # New conversation
        conv_id = _new_id()
        doc = {
            "_id": conv_id,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "title": user_message.strip()[:60] or "New conversation",
            "messages": new_messages,
            "summary": None,         # AI-generated summary, saved after first generation
            "summary_stale": True,   # True = needs (re)generation
            "created_at": now,
            "updated_at": _now(),
        }
        await self._conversations.insert_one(doc)
        await self._agents.update_one(
            {"_id": agent_id, "tenant_id": tenant_id},
            {"$inc": {"conversation_count": 1}, "$set": {"updated_at": _now()}},
        )
        doc["id"] = doc.pop("_id")
        return doc

    async def list_conversations(self, agent_id: str, tenant_id: str) -> list[dict[str, Any]]:
        # Use aggregation to count messages without loading full message arrays
        pipeline = [
            {"$match": {"agent_id": agent_id, "tenant_id": tenant_id}},
            {"$addFields": {"message_count": {"$size": {"$ifNull": ["$messages", []]}}}},
            {"$project": {"messages": 0}},
            {"$sort": {"updated_at": -1}},
            {"$limit": 500},
        ]
        convs = await self._conversations.aggregate(pipeline).to_list(length=500)
        for c in convs:
            c["id"] = c.pop("_id")
        return convs

    async def get_conversation_summary(
        self, conversation_id: str, agent_id: str, tenant_id: str
    ) -> tuple[str | None, bool]:
        """Returns (summary_text, is_stale). stale=True means needs regeneration."""
        doc = await self._conversations.find_one(
            {"_id": conversation_id, "agent_id": agent_id, "tenant_id": tenant_id},
            {"summary": 1, "summary_stale": 1},
        )
        if not doc:
            return None, True
        return doc.get("summary"), doc.get("summary_stale", True)

    async def save_conversation_summary(
        self, conversation_id: str, agent_id: str, tenant_id: str, summary: str
    ) -> None:
        """Save generated summary and mark as fresh."""
        await self._conversations.update_one(
            {"_id": conversation_id, "agent_id": agent_id, "tenant_id": tenant_id},
            {"$set": {"summary": summary, "summary_stale": False}},
        )

    async def get_conversation(
        self, conversation_id: str, agent_id: str, tenant_id: str
    ) -> dict[str, Any]:
        doc = await self._conversations.find_one({
            "_id": conversation_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        if doc is None:
            raise KeyError(f"Conversation '{conversation_id}' not found.")
        doc["id"] = doc.pop("_id")
        return doc

    # ── Fallback / Analytics ──────────────────────────────────────────────

    async def log_fallback(
        self,
        agent_id: str,
        tenant_id: str,
        question: str,
        conversation_id: str | None = None,
    ) -> None:
        """Log a question the bot could not answer from context."""
        await self._fallback_logs.insert_one({
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "question": question,
            "conversation_id": conversation_id,
            "created_at": _now(),
        })

    async def get_fallback_logs(self, agent_id: str, tenant_id: str, limit: int = 50) -> list[dict]:
        """Get questions the bot could not answer, most recent first."""
        cursor = self._fallback_logs.find(
            {"agent_id": agent_id, "tenant_id": tenant_id},
            sort=[("created_at", -1)],
        ).limit(limit)
        docs = await cursor.to_list(length=limit)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def get_fallback_count(self, agent_id: str, tenant_id: str) -> int:
        return await self._fallback_logs.count_documents(
            {"agent_id": agent_id, "tenant_id": tenant_id}
        )

    # ── Subscription / Plan Assignment ───────────────────────────────────

    async def assign_plan_to_user(self, user_id: str, plan: dict, duration_months: int = 1) -> dict:
        """
        Assign a subscription plan to a user.
        Copies plan config and initialises usage counters.
        duration_months: number of 30-day cycles (1–12)
        """
        from datetime import timedelta
        duration_months = max(1, min(12, duration_months))
        now = _now()
        cycle_start = now
        cycle_end   = now + timedelta(days=30 * duration_months)

        subscription = {
            "plan_id":                    str(plan["_id"]),
            "plan_name":                  plan["name"],
            "monthly_message_limit":      plan["totalMessages"],
            "chat_token_limit":           plan.get("chatTokenLimit",    plan["totalMessages"] * 500),
            "summary_token_limit":        plan.get("summaryTokenLimit", plan["totalMessages"] * 300),
            "average_tokens_per_message": plan.get("tokensPerMessage",  800),
            # Usage counters
            "used_messages":              0,
            "remaining_messages":         plan["totalMessages"],
            "chat_tokens_used":           0,
            "chat_tokens_remaining":      plan.get("chatTokenLimit",    plan["totalMessages"] * 500),
            "summary_tokens_used":        0,
            "summary_tokens_remaining":   plan.get("summaryTokenLimit", plan["totalMessages"] * 300),
            # Cycle
            "cycle_start_date":           cycle_start.isoformat(),
            "cycle_end_date":             cycle_end.isoformat(),
            "duration_months":            duration_months,
            "assigned_at":                now.isoformat(),
            "selling_price":              plan.get("sellingPrice", 0),
            "billing_status":             "active",
        }

        await self._users.update_one(
            {"_id": user_id},
            {"$set": {"subscription": subscription, "updated_at": now}},
        )
        return subscription

    async def get_user_subscription(self, user_id: str) -> dict | None:
        doc = await self._users.find_one({"_id": user_id}, {"subscription": 1})
        return doc.get("subscription") if doc else None

    async def check_and_reset_cycle(self, user_id: str) -> dict | None:
        """Reset usage if cycle has expired. Returns updated subscription."""
        from datetime import datetime, timedelta, timezone
        sub = await self.get_user_subscription(user_id)
        if not sub:
            return None
        now = datetime.now(timezone.utc)
        cycle_end = datetime.fromisoformat(sub["cycle_end_date"].replace("Z", "+00:00"))             if "+" not in sub["cycle_end_date"]             else datetime.fromisoformat(sub["cycle_end_date"])
        if now > cycle_end:
            new_start = cycle_end
            new_end   = cycle_end + timedelta(days=30)
            reset = {
                "subscription.remaining_messages":       sub["monthly_message_limit"],
                "subscription.used_messages":            0,
                "subscription.chat_tokens_used":         0,
                "subscription.chat_tokens_remaining":    sub["chat_token_limit"],
                "subscription.summary_tokens_used":      0,
                "subscription.summary_tokens_remaining": sub["summary_token_limit"],
                "subscription.cycle_start_date":         new_start.isoformat(),
                "subscription.cycle_end_date":           new_end.isoformat(),
                "updated_at": _now(),
            }
            await self._users.update_one({"_id": user_id}, {"$set": reset})
            return await self.get_user_subscription(user_id)
        return sub

    async def consume_tokens(
        self,
        user_id:        str,
        agent_id:       str,
        input_tokens:   int,
        output_tokens:  int,
        summary_tokens: int,
    ) -> dict:
        """
        Validate limits and deduct tokens from user subscription.
        Returns {"allowed": bool, "reason": str, "sub": dict}
        """
        sub = await self.check_and_reset_cycle(user_id)
        if not sub:
            return {"allowed": True, "reason": "no_plan", "sub": None}

        chat_tokens_used = input_tokens + output_tokens
        avg              = sub.get("average_tokens_per_message", 800) or 800
        total_tokens     = chat_tokens_used + summary_tokens
        message_cost     = max(round(total_tokens / avg, 2), 1)

        # Validate
        if sub["remaining_messages"] <= 0:
            return {"allowed": False, "reason": "Message limit reached for this cycle.", "sub": sub}
        if sub["chat_tokens_remaining"] < chat_tokens_used:
            return {"allowed": False, "reason": "Chat token limit reached for this cycle.", "sub": sub}
        if sub["summary_tokens_remaining"] < summary_tokens:
            return {"allowed": False, "reason": "Summary token limit reached for this cycle.", "sub": sub}

        # Deduct
        update = {
            "subscription.chat_tokens_used":          sub["chat_tokens_used"]         + chat_tokens_used,
            "subscription.chat_tokens_remaining":     sub["chat_tokens_remaining"]     - chat_tokens_used,
            "subscription.summary_tokens_used":       sub["summary_tokens_used"]       + summary_tokens,
            "subscription.summary_tokens_remaining":  sub["summary_tokens_remaining"]  - summary_tokens,
            "subscription.used_messages":             sub["used_messages"]             + message_cost,
            "subscription.remaining_messages":        sub["remaining_messages"]        - message_cost,
            "updated_at": _now(),
        }
        await self._users.update_one({"_id": user_id}, {"$set": update})

        # Per-agent usage tracking
        await self._activity.update_one(
            {"agent_id": agent_id, "user_id": user_id, "type": "token_usage"},
            {"$inc": {
                "messages_used":      message_cost,
                "chat_tokens_used":   chat_tokens_used,
                "summary_tokens_used": summary_tokens,
            },
            "$setOnInsert": {"created_at": _now()}},
            upsert=True,
        )

        return {"allowed": True, "reason": "ok", "sub": sub}

    # ── Billing Status ────────────────────────────────────────────────────

    async def set_billing_status(self, user_id: str, status: str) -> dict:
        """Set billing status: active | paused | stopped"""
        doc = await self._users.find_one_and_update(
            {"_id": user_id},
            {"$set": {"subscription.billing_status": status, "updated_at": _now()}},
            return_document=True,
        )
        if not doc:
            raise KeyError(f"User '{user_id}' not found.")
        return doc.get("subscription", {})

    async def get_billing_status(self, user_id: str) -> str:
        """Returns billing status for a user: active | paused | stopped | no_plan"""
        doc = await self._users.find_one({"_id": user_id}, {"subscription": 1})
        if not doc or not doc.get("subscription"):
            return "no_plan"
        return doc["subscription"].get("billing_status", "active")

    # ── Roles & Permissions ───────────────────────────────────────────────    # ── Billing Status ────────────────────────────────────────────────────

    async def set_billing_status(self, user_id: str, status: str) -> dict:
        """Set billing status: active | paused | stopped"""
        doc = await self._users.find_one_and_update(
            {"_id": user_id},
            {"$set": {"subscription.billing_status": status, "updated_at": _now()}},
            return_document=True,
        )
        if not doc:
            raise KeyError(f"User '{user_id}' not found.")
        return doc.get("subscription", {})


    # ── Roles & Permissions ───────────────────────────────────────────────

    async def list_permissions(self) -> list[dict]:
        docs = await self._permissions.find({}).sort("name", 1).to_list(length=200)
        visible_docs: list[dict] = []
        for d in docs:
            if not is_visible_permission_resource(d.get("resource", "")):
                continue
            d["id"] = str(d.pop("_id"))
            visible_docs.append(d)
        return visible_docs

    async def create_permission(self, name: str, description: str, resource: str, action: str) -> dict:
        normalized_resource = normalize_permission_resource(resource)
        normalized_name = normalize_permission_name(name)
        existing = await self._permissions.find_one({"name": normalized_name})
        if existing:
            raise ValueError(f"Permission '{normalized_name}' already exists.")
        doc = {
            "_id": _new_id(),
            "name": normalized_name,
            "description": description,
            "resource": normalized_resource,
            "action": action,
            "created_at": _now(),
        }
        await self._permissions.insert_one(doc)
        doc["id"] = doc.pop("_id")
        return doc

    async def delete_permission(self, permission_id: str) -> None:
        await self._permissions.delete_one({"_id": permission_id})
        # Remove from all roles
        await self._roles.update_many({}, {"$pull": {"permissions": permission_id}})

    async def list_roles(self) -> list[dict]:
        docs = await self._roles.find({}).sort("name", 1).to_list(length=100)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def create_role(self, name: str, description: str, permissions: list[str]) -> dict:
        existing = await self._roles.find_one({"name": name})
        if existing:
            raise ValueError(f"Role '{name}' already exists.")
        normalized_permissions = normalize_permission_names(permissions)
        # key is a stable slug — alphanumeric + underscores only, never changes after creation
        import re as _re
        key = _re.sub(r"[^a-z0-9_]", "", name.lower().strip().replace(" ", "_"))
        doc = {
            "_id": _new_id(),
            "name": name,
            "key":  key,
            "description": description,
            "permissions": normalized_permissions,
            "is_system": False,
            "created_at": _now(),
        }
        await self._roles.insert_one(doc)
        doc["id"] = doc.pop("_id")
        return doc

    async def update_role(self, role_id: str, name: str, description: str, permissions: list[str]) -> dict:
        existing = await self._roles.find_one({"_id": role_id})
        if not existing:
            raise KeyError(f"Role '{role_id}' not found.")
        if existing.get("is_super_admin"):
            raise ValueError("Cannot edit the super admin role.")
        normalized_permissions = normalize_permission_names(permissions)

        # permissions from the frontend are always name strings (e.g. "agents:read").
        # For system roles: track which are seed-assigned vs manually added so
        # the next server restart only replaces seed_permissions, not extra.
        # For custom roles: permissions is fully authoritative.
        update: dict = {
            "name":        name,
            "description": description,
            "permissions": normalized_permissions,   # store name strings directly
            "updated_at":  _now(),
        }
        if existing.get("is_system"):
            seed_perms = normalize_permission_names(existing.get("seed_permissions", []))
            # extra = selected by user AND not already covered by seed
            extra_perms = [p for p in normalized_permissions if p not in seed_perms]
            update["extra_permissions"] = extra_perms

        doc = await self._roles.find_one_and_update(
            {"_id": role_id},
            {"$set": update},
            return_document=True,
        )
        doc["id"] = doc.pop("_id")
        return doc

    async def delete_role(self, role_id: str) -> None:
        role = await self._roles.find_one({"_id": role_id})
        if not role:
            raise KeyError(f"Role not found.")
        if role.get("is_system"):
            raise ValueError("System-defined roles cannot be deleted.")
        await self._roles.delete_one({"_id": role_id})

    async def seed_default_permissions(self) -> None:
        """
        Seed permissions from the central registry.

        To add permissions for a new page, edit:
            app/core/permissions_registry.py  ← add one entry, restart, done.

        This function is registry-driven — no changes needed here.
        """
        from app.core.permissions_registry import build_permission_defaults
        defaults = build_permission_defaults()
        # Per-permission upsert: only insert permissions that don't exist yet.
        # This means adding a new resource to `defaults` and restarting the server
        # is enough to make it appear in the permissions list — no manual DB work needed.
        existing_names = {
            d["name"]
            for d in await self._permissions.find({}, {"name": 1}).to_list(length=1000)
        }
        new_perms = [
            {"_id": _new_id(), "created_at": _now(), **perm}
            for perm in defaults
            if perm["name"] not in existing_names
        ]
        if new_perms:
            await self._permissions.insert_many(new_perms)

    async def seed_default_roles(self) -> None:
        """
        Create default system roles using actual permission IDs from the permissions collection.
        Roles are built dynamically based on available seeded permissions.
        """

        # Backfill key field on any existing roles that are missing it
        async for role_doc in self._roles.find({"key": {"$exists": False}}):
            fallback_key = role_doc.get("name", "").lower().strip().replace(" ", "_")
            await self._roles.update_one(
                {"_id": role_doc["_id"]},
                {"$set": {"key": fallback_key}},
            )

        # ── ONE-TIME MIGRATION: convert UUID permission refs → name strings ──────
        # Before the name-string refactor, permissions were stored as UUIDs.
        # This block converts any remaining UUID refs to names on every restart.
        # It is idempotent: name strings (containing ":") are left untouched.
        all_perm_docs = await self._permissions.find({}, {"_id": 1, "name": 1}).to_list(length=1000)
        id_to_name: dict[str, str] = {str(p["_id"]): p["name"] for p in all_perm_docs}

        async for role_doc in self._roles.find({}):
            fields_to_migrate = ["permissions", "seed_permissions", "extra_permissions"]
            updates: dict = {}
            for field in fields_to_migrate:
                refs: list[str] = role_doc.get(field, [])
                if not refs or refs == ["*"]:
                    continue
                # Check if any ref is a UUID (doesn't contain ":")
                if any(":" not in r for r in refs):
                    migrated = [
                        id_to_name.get(r, r) if ":" not in r else r
                        for r in refs
                    ]
                    # Drop any refs that couldn't be resolved (stale IDs)
                    migrated = [m for m in migrated if ":" in m]
                    updates[field] = normalize_permission_names(migrated)
                elif any(normalize_permission_name(r) != r for r in refs):
                    updates[field] = normalize_permission_names(refs)
            if updates:
                await self._roles.update_one(
                    {"_id": role_doc["_id"]},
                    {"$set": updates},
                )
        # ── End migration ─────────────────────────────────────────────────────────

        # Always run on every restart — no short-circuit.
        # Seed data changes (name, description, permissions) are applied immediately.
        # Fetch all valid permission names for validation
        perm_docs = await self._permissions.find({}, {"name": 1}).to_list(length=500)
        valid_names: set[str] = {p["name"] for p in perm_docs}

        def resolve(*names: str) -> list[str]:
            """Return permission name strings (not IDs) — names are the stable format."""
            return [n for n in names if n in valid_names]

        # System-defined roles — exactly 2, cannot be deleted.
        # super_admin: all permissions, cannot be edited at all.
        # customer: default starting permissions, name/permissions editable via UI.
        defaults = [
            {
                "name": "super_admin",
                "key":  "super_admin",
                "description": "Unrestricted access to everything including roles, permissions, and all tenants.",
                "permissions": ["*"],
                "is_system": True,
                "is_super_admin": True,
            },
            {
                "name": "customer",
                "key":  "customer",
                "description": "Standard customer role — edit name and permissions freely via the Roles UI.",
                "permissions": resolve(
                    # Agents
                    "agents:read", "agents:write",
                    # Chats
                    "chats:read",
                    # Dashboard
                    "dashboard:read",
                ),
                "is_system": True,
            },
        ]

        for role in defaults:
            # Match by key (stable) so renames in the DB don't prevent updates
            existing = (
                await self._roles.find_one({"key": role["key"]})
                or await self._roles.find_one({"name": role["name"]})
            )

            seed_perms = normalize_permission_names(role["permissions"])

            # seed_perms are now NAME strings (e.g. ["agents:read", "billing:read"])
            # so everything is the same format as what the UI sends and stores.

            if not existing:
                # First insert: seed owns everything, no manual additions yet
                doc = {
                    "_id": _new_id(),
                    "created_at": _now(),
                    "seed_permissions":  seed_perms,  # names from seed
                    "extra_permissions": [],           # nothing manual yet
                    "permissions":       seed_perms,  # effective = seed + extra
                    **role,
                }
                await self._roles.insert_one(doc)
            else:
                # Role exists.
                # seed_permissions: always replaced with current seed (authoritative).
                # extra_permissions: what the user added via UI beyond the seed — preserved.
                # permissions: union of both, deduplicated, seed order first.
                extra_perms = normalize_permission_names(existing.get("extra_permissions", []))

                if "*" in seed_perms:
                    effective = ["*"]
                else:
                    effective = list(dict.fromkeys(
                        seed_perms + [p for p in extra_perms if p not in seed_perms]
                    ))

                await self._roles.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {
                        "key":               role["key"],
                        "seed_permissions":  seed_perms,   # overwrite with current seed
                        "extra_permissions": extra_perms,  # preserve manual additions
                        "permissions":       effective,    # effective = seed + extra
                        "is_system":         True,
                        "updated_at":        _now(),
                    }},
                )

        deprecated_resources = list(RESOURCE_PERMISSION_ALIASES.keys())
        if deprecated_resources:
            await self._permissions.delete_many({"resource": {"$in": deprecated_resources}})

    # ── Dashboard ─────────────────────────────────────────────────────────────

    async def dashboard_summary(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        import asyncio

        total_agents_coro = self._agents.count_documents(
            {"tenant_id": tenant_id, "user_id": user_id}
        )
        total_documents_coro = self._documents.count_documents(
            {"tenant_id": tenant_id}
        )
        total_conversations_coro = self._conversations.count_documents(
            {"tenant_id": tenant_id}
        )
        recent_cursor = self._activity.find(
            {"tenant_id": tenant_id},
            sort=[("timestamp", -1)],
            limit=8,
        )
        recent_activity_coro = recent_cursor.to_list(length=8)

        total_agents, total_documents, total_conversations, recent_activity = (
            await asyncio.gather(
                total_agents_coro,
                total_documents_coro,
                total_conversations_coro,
                recent_activity_coro,
            )
        )

        for a in recent_activity:
            a["id"] = a.pop("_id")

        return {
            "total_agents": total_agents,
            "total_documents": total_documents,
            "total_conversations": total_conversations,
            "recent_activity": recent_activity,
        }
