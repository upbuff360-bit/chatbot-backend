from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.prompt_builder import SYSTEM_PROMPT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdminStore:
    def __init__(
        self,
        state_path: str | Path = "./persist/admin/state.json",
        agents_root: str | Path = "./data/agents",
    ) -> None:
        self.state_path = Path(state_path)
        self.agents_root = Path(agents_root)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.agents_root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._save_state({"agents": [], "recent_activity": []})

    def list_agents(self) -> list[dict[str, Any]]:
        agents = self._state()["agents"]
        return sorted(
            [self._agent_summary(agent) for agent in agents],
            key=lambda agent: agent["created_at"],
            reverse=True,
        )

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        for agent in self._state()["agents"]:
            if agent["id"] == agent_id:
                return agent
        return None

    def require_agent(self, agent_id: str) -> dict[str, Any]:
        agent = self.get_agent(agent_id)
        if agent is None:
            raise KeyError(f"Agent '{agent_id}' was not found.")
        return agent

    def create_agent(self, name: str) -> dict[str, Any]:
        state = self._state()
        timestamp = _utc_now()
        agent = {
            "id": str(uuid4()),
            "name": name.strip(),
            "created_at": timestamp,
            "settings": {
                "system_prompt": SYSTEM_PROMPT,
                "temperature": 0.2,
                "welcome_message": "Hi, I'm your AI assistant. Ask me anything about this knowledge base.",
            },
            "documents": [],
            "conversations": [],
        }
        state["agents"].append(agent)
        state["recent_activity"].insert(
            0,
            self._activity("agent_created", agent["id"], f"Created agent '{agent['name']}'.", timestamp),
        )
        self._save_state(state)
        self.get_agent_pdf_dir(agent["id"]).mkdir(parents=True, exist_ok=True)
        self.get_agent_website_dir(agent["id"]).mkdir(parents=True, exist_ok=True)
        self.get_agent_snippet_dir(agent["id"]).mkdir(parents=True, exist_ok=True)
        self.get_agent_qa_dir(agent["id"]).mkdir(parents=True, exist_ok=True)
        return self._agent_summary(agent)

    def update_agent(self, agent_id: str, name: str) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        previous_name = agent["name"]
        agent["name"] = name.strip()
        state["recent_activity"].insert(
            0,
            self._activity("agent_updated", agent_id, f"Renamed agent '{previous_name}' to '{agent['name']}'."),
        )
        self._save_state(state)
        return self._agent_summary(agent)

    def delete_agent(self, agent_id: str) -> None:
        state = self._state()
        agent = self.require_agent(agent_id)
        state["agents"] = [item for item in state["agents"] if item["id"] != agent_id]
        state["recent_activity"].insert(
            0,
            self._activity("agent_deleted", agent_id, f"Deleted agent '{agent['name']}'."),
        )
        self._save_state(state)

    def list_documents(self, agent_id: str) -> list[dict[str, Any]]:
        agent = self.require_agent(agent_id)
        return sorted(agent["documents"], key=lambda doc: doc["uploaded_at"], reverse=True)

    def sync_documents(self, agent_id: str, file_names: list[str]) -> list[dict[str, Any]]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        existing_by_name = {
            document["file_name"]: document
            for document in agent["documents"]
            if document.get("source_type", "pdf") == "pdf"
        }
        website_documents = [
            document
            for document in agent["documents"]
            if document.get("source_type", "pdf") != "pdf"
        ]
        documents: list[dict[str, Any]] = []

        for file_name in sorted(file_names):
            existing = existing_by_name.get(file_name)
            document = existing or {
                "id": str(uuid4()),
                "file_name": file_name,
                "uploaded_at": _utc_now(),
                "status": "indexed",
                "source_type": "pdf",
                "source_url": None,
            }
            document["status"] = "indexed"
            document["source_type"] = "pdf"
            document["source_url"] = None
            documents.append(document)

        agent["documents"] = website_documents + documents
        self._save_state(state)
        return sorted(agent["documents"], key=lambda doc: doc["uploaded_at"], reverse=True)

    def mark_document_uploaded(
        self,
        agent_id: str,
        file_name: str,
        status: str = "uploaded",
        source_type: str = "pdf",
        source_url: str | None = None,
    ) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        for document in agent["documents"]:
            if document["file_name"] == file_name and document.get("source_type", "pdf") == source_type:
                document["status"] = status
                document["source_type"] = source_type
                document["source_url"] = source_url
                self._save_state(state)
                return document

        document = {
            "id": str(uuid4()),
            "file_name": file_name,
            "uploaded_at": _utc_now(),
            "status": status,
            "source_type": source_type,
            "source_url": source_url,
        }
        agent["documents"].append(document)
        state["recent_activity"].insert(
            0,
            self._activity("document_uploaded", agent_id, f"Uploaded '{file_name}'."),
        )
        self._save_state(state)
        return document

    def upsert_website_source(
        self,
        agent_id: str,
        display_name: str,
        source_url: str,
        status: str = "indexed",
    ) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)

        for document in agent["documents"]:
            if document.get("source_type") == "website" and document.get("source_url") == source_url:
                document["file_name"] = display_name
                document["status"] = status
                self._save_state(state)
                return document

        document = {
            "id": str(uuid4()),
            "file_name": display_name,
            "uploaded_at": _utc_now(),
            "status": status,
            "source_type": "website",
            "source_url": source_url,
        }
        agent["documents"].append(document)
        state["recent_activity"].insert(
            0,
            self._activity("document_uploaded", agent_id, f"Added website '{display_name}'."),
        )
        self._save_state(state)
        return document

    def update_document(
        self,
        agent_id: str,
        document_id: str,
        *,
        file_name: str,
        source_url: str | None = None,
    ) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        document = self._require_document_in_state(agent, document_id)
        previous_name = document["file_name"]
        document["file_name"] = file_name.strip()
        if source_url is not None:
            document["source_url"] = source_url
        state["recent_activity"].insert(
            0,
            self._activity("document_updated", agent_id, f"Updated '{previous_name}' to '{document['file_name']}'."),
        )
        self._save_state(state)
        return dict(document)

    def delete_document(self, agent_id: str, document_id: str) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        document = self._require_document_in_state(agent, document_id)
        agent["documents"] = [item for item in agent["documents"] if item["id"] != document_id]
        state["recent_activity"].insert(
            0,
            self._activity("document_deleted", agent_id, f"Deleted '{document['file_name']}'."),
        )
        self._save_state(state)
        return dict(document)

    def get_settings(self, agent_id: str) -> dict[str, Any]:
        agent = self.require_agent(agent_id)
        return dict(agent["settings"])

    def update_settings(self, agent_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        agent["settings"] = {
            **agent["settings"],
            **settings,
        }
        state["recent_activity"].insert(
            0,
            self._activity("settings_updated", agent_id, f"Updated settings for '{agent['name']}'."),
        )
        self._save_state(state)
        return dict(agent["settings"])

    def append_conversation_messages(
        self,
        agent_id: str,
        user_message: str,
        assistant_message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._state()
        agent = self._require_agent_in_state(state, agent_id)
        timestamp = _utc_now()
        conversation = None

        if conversation_id is not None:
            for item in agent["conversations"]:
                if item["id"] == conversation_id:
                    conversation = item
                    break

        if conversation is None:
            conversation = {
                "id": str(uuid4()),
                "title": user_message.strip()[:60] or "New conversation",
                "created_at": timestamp,
                "updated_at": timestamp,
                "messages": [],
            }
            agent["conversations"].insert(0, conversation)

        conversation["messages"].extend(
            [
                {"id": str(uuid4()), "role": "user", "content": user_message, "timestamp": timestamp},
                {"id": str(uuid4()), "role": "assistant", "content": assistant_message, "timestamp": _utc_now()},
            ]
        )
        conversation["updated_at"] = _utc_now()
        state["recent_activity"].insert(
            0,
            self._activity("conversation_updated", agent_id, f"Conversation updated for '{agent['name']}'."),
        )
        self._save_state(state)
        return self._conversation_detail(conversation)

    def list_conversations(self, agent_id: str) -> list[dict[str, Any]]:
        agent = self.require_agent(agent_id)
        return sorted(
            [self._conversation_summary(conversation) for conversation in agent["conversations"]],
            key=lambda conversation: conversation["updated_at"],
            reverse=True,
        )

    def get_conversation(self, agent_id: str, conversation_id: str) -> dict[str, Any]:
        agent = self.require_agent(agent_id)
        for conversation in agent["conversations"]:
            if conversation["id"] == conversation_id:
                return self._conversation_detail(conversation)
        raise KeyError(f"Conversation '{conversation_id}' was not found.")

    def dashboard_summary(self) -> dict[str, Any]:
        state = self._state()
        agents = state["agents"]
        return {
            "total_agents": len(agents),
            "total_documents": sum(len(agent["documents"]) for agent in agents),
            "total_conversations": sum(len(agent["conversations"]) for agent in agents),
            "recent_activity": state["recent_activity"][:8],
        }

    def get_agent_pdf_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id / "pdfs"

    def get_agent_website_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id / "websites"

    def get_agent_snippet_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id / "text_snippets"

    def get_agent_qa_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id / "qa"

    def _state(self) -> dict[str, Any]:
        with self.state_path.open("r", encoding="utf-8") as state_file:
            return json.load(state_file)

    def _save_state(self, state: dict[str, Any]) -> None:
        with self.state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2)

    def _activity(
        self,
        activity_type: str,
        agent_id: str,
        description: str,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": str(uuid4()),
            "type": activity_type,
            "agent_id": agent_id,
            "description": description,
            "timestamp": timestamp or _utc_now(),
        }

    def _require_agent_in_state(self, state: dict[str, Any], agent_id: str) -> dict[str, Any]:
        for agent in state["agents"]:
            if agent["id"] == agent_id:
                return agent
        raise KeyError(f"Agent '{agent_id}' was not found.")

    @staticmethod
    def _require_document_in_state(agent: dict[str, Any], document_id: str) -> dict[str, Any]:
        for document in agent["documents"]:
            if document["id"] == document_id:
                return document
        raise KeyError(f"Document '{document_id}' was not found.")

    @staticmethod
    def _agent_summary(agent: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": agent["id"],
            "name": agent["name"],
            "created_at": agent["created_at"],
            "document_count": len(agent["documents"]),
            "conversation_count": len(agent["conversations"]),
        }

    @staticmethod
    def _conversation_summary(conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "message_count": len(conversation["messages"]),
        }

    @staticmethod
    def _conversation_detail(conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": conversation["messages"],
        }
