"""
Permissions Registry
====================
This is the SINGLE source of truth for every page/resource in the application
and the actions each supports.

To add permissions for a new page:
    1. Add one entry to RESOURCE_REGISTRY below.
    2. Restart the server — permissions are auto-seeded via seed_default_permissions().
    3. The Permissions page in the admin UI will show the new resource immediately.

Format of each entry:
    {
        "resource": "<snake_case_name>",   # must match the URL segment / page name
        "label":    "<Human label>",        # shown in the admin UI
        "actions":  ["read", "write", ...], # subset of: read, write, delete, manage
        "descriptions": {                   # optional per-action override
            "read":   "...",
            "manage": "...",
        },
    }

Default descriptions are auto-generated as "View <label>", "Create and edit <label>", etc.
"""

from __future__ import annotations

from typing import TypedDict


# ── Action description templates ──────────────────────────────────────────────

_ACTION_TEMPLATES: dict[str, str] = {
    "read":   "View {label}",
    "write":  "Create and edit {label}",
    "delete": "Delete {label}",
    "manage": "Full control over {label}",
}


class ResourceEntry(TypedDict, total=False):
    resource:     str
    label:        str
    actions:      list[str]
    descriptions: dict[str, str]  # optional per-action override


# Child agent resources are preserved as legacy aliases in route checks,
# but they are no longer first-class assignable permissions.
RESOURCE_PERMISSION_ALIASES: dict[str, str] = {
    "knowledge": "agents",
    "conversations": "agents",
    "analytics": "agents",
    "settings": "agents",
    "documents": "agents",
}


def normalize_permission_resource(resource: str) -> str:
    return RESOURCE_PERMISSION_ALIASES.get(resource, resource)


def normalize_permission_name(permission_name: str) -> str:
    if permission_name == "*" or ":" not in permission_name:
        return permission_name
    resource, action = permission_name.split(":", 1)
    return f"{normalize_permission_resource(resource)}:{action}"


def normalize_permission_names(permission_names: list[str]) -> list[str]:
    normalized = [normalize_permission_name(name) for name in permission_names]
    if "*" in normalized:
        return ["*"]
    return list(dict.fromkeys(normalized))


def is_visible_permission_resource(resource: str) -> bool:
    return resource not in RESOURCE_PERMISSION_ALIASES


# ── Registry ──────────────────────────────────────────────────────────────────
# Add a new dict here to register a new page.
# Actions are applied in order: read → write → delete → manage.

RESOURCE_REGISTRY: list[ResourceEntry] = [
    {
        "resource": "agents",
        "label":    "agents",
        "actions":  ["read", "write", "delete", "manage"],
    },
    {
        "resource": "users",
        "label":    "users",
        "actions":  ["read", "write", "delete", "manage"],
    },
    {
        "resource": "roles",
        "label":    "roles and permissions",
        "actions":  ["read", "manage"],
        "descriptions": {
            "read":   "View roles and permissions",
            "manage": "Full control over roles and permissions",
        },
    },
    {
        "resource": "billing",
        "label":    "billing",
        "actions":  ["read", "manage"],
        "descriptions": {
            "read":   "View billing and plans",
            "manage": "Full control over billing",
        },
    },
    {
        "resource": "plans",
        "label":    "plans",
        "actions":  ["read", "write", "delete", "manage"],
    },
    {
        "resource": "dashboard",
        "label":    "dashboard",
        "actions":  ["read"],
    },
    {
        "resource": "chats",
        "label":    "chats",
        "actions":  ["read"],
        "descriptions": {
            "read": "View chats",
        },
    },

    # ── Add new pages below this line ─────────────────────────────────────────
    # Example:
    # {
    #     "resource": "reports",
    #     "label":    "reports",
    #     "actions":  ["read", "write", "manage"],
    # },
]


# ── Builder ───────────────────────────────────────────────────────────────────

def build_permission_defaults() -> list[dict]:
    """
    Expand RESOURCE_REGISTRY into the flat list of permission dicts expected
    by seed_default_permissions().

    Called at startup — pure Python, no DB access.
    """
    permissions: list[dict] = []
    for entry in RESOURCE_REGISTRY:
        resource     = entry["resource"]
        label        = entry.get("label", resource)
        actions      = entry.get("actions", ["read", "manage"])
        descriptions = entry.get("descriptions", {})

        for action in actions:
            desc = descriptions.get(
                action,
                _ACTION_TEMPLATES.get(action, f"{action.capitalize()} {label}").format(label=label),
            )
            permissions.append({
                "name":        f"{resource}:{action}",
                "resource":    resource,
                "action":      action,
                "description": desc,
            })
    return permissions
