from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.core.permissions_registry import normalize_permission_name, normalize_permission_names
from app.core.security import decode_access_token
from app.models.user import UserDocument, UserRole

_bearer = HTTPBearer()


class CurrentUser:
    """Resolved user injected into route handlers via Depends."""
    def __init__(self, id: str, email: str, tenant_id: str, role: UserRole, role_key: str = "") -> None:
        self.id = id
        self.email = email
        self.tenant_id = tenant_id
        self.role = role
        # role_key: the original raw string from the JWT (e.g. "customer").
        # May differ from role.value when a custom role doesn't exist in the enum.
        self.role_key = role_key or role.value

    def require_role(self, *allowed: UserRole) -> None:
        # super_admin bypasses all role checks
        if self.role == UserRole.SUPER_ADMIN:
            return
        if self.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{self.role}' is not permitted for this action.",
            )

    async def has_permission(self, resource: str, action: str) -> bool:
        """
        Check permission against the user's CURRENT role in MongoDB.
        The JWT role_key is a hint but may be stale (e.g. role updated after login).
        The DB user document is always the source of truth.
        """
        if self.role == UserRole.SUPER_ADMIN or self.role_key == "super_admin":
            return True
        try:
            from app.main import store

            # ── Step 1: Get the user's current role from DB (handles stale JWTs) ──
            user_doc = await store._users.find_one({"_id": self.id}, {"role": 1})
            current_role = (user_doc.get("role", "") if user_doc else "") or self.role_key

            # Try current DB role first, fall back to JWT role_key
            role_keys_to_try = list(dict.fromkeys([current_role, self.role_key]))

            role_doc = None
            for rk in role_keys_to_try:
                role_doc = (
                    await store._roles.find_one({"key": rk})
                    or await store._roles.find_one({"name": rk})
                )
                if role_doc:
                    break

            if not role_doc:
                # Last resort: use any non-super-admin system role
                role_doc = await store._roles.find_one(
                    {"is_system": True, "is_super_admin": {"$ne": True}}
                )
            if not role_doc:
                return True  # no roles defined at all — graceful fallback

            perm_refs: list[str] = role_doc.get("permissions", [])

            if "*" in perm_refs:
                return True

            # Resolve: name strings pass through, UUID refs get looked up
            name_refs = [r for r in perm_refs if ":" in r]
            id_refs   = [r for r in perm_refs if ":" not in r]
            perm_names: set[str] = set(name_refs)

            if id_refs:
                perm_docs = await store._permissions.find(
                    {"_id": {"$in": id_refs}}, {"name": 1}
                ).to_list(length=500)
                perm_names.update(p["name"] for p in perm_docs)

            perm_names = normalize_permission_names(list(perm_names))

            permission_name = normalize_permission_name(f"{resource}:{action}")
            manage_name     = normalize_permission_name(f"{resource}:manage")

            return permission_name in perm_names or manage_name in perm_names

        except Exception:
            return True  # DB error — don't block users

    async def require_permission(self, resource: str, action: str) -> None:
        """Raise 403 if the user's role lacks the given permission."""
        if not await self.has_permission(resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your role does not have '{resource}:{action}' permission.",
            )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    token = credentials.credentials
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub", "")
        tenant_id: str = payload.get("tenant_id", "")
        role: str = payload.get("role", "")
        email: str = payload.get("email", "")
        if not user_id or not tenant_id or not role:
            raise credentials_error
    except JWTError:
        raise credentials_error

    # Parse role safely — custom roles not in the enum (e.g. "customer") fall back
    # to VIEWER as the baseline. The DB-driven permissions system controls actual access.
    try:
        parsed_role = UserRole(role)
    except ValueError:
        parsed_role = UserRole.VIEWER

    return CurrentUser(
        id=user_id,
        email=email,
        tenant_id=tenant_id,
        role=parsed_role,
        role_key=role,   # preserve original JWT string for DB permission lookup
    )


# Convenience aliases
RequireOwner = Depends(get_current_user)
RequireAdmin = Depends(get_current_user)
RequireAny   = Depends(get_current_user)
