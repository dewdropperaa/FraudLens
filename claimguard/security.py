"""
API authentication: X-API-Key, JWT (HS256), and user login tokens.
Password hashing uses Python built-in hashlib (PBKDF2-SHA256) — no external dependencies.
"""
from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request, status

from claimguard.config import is_production_environment, load_environment

try:
    import jwt
except ImportError:  # pragma: no cover
    jwt = None

_DEV_JWT_SECRET = "claimguard-dev-jwt-secret-2024"
_PBKDF2_ITERATIONS = 260_000


# ── Password helpers (no external lib) ───────────────────────────

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return base64.b64encode(salt + key).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        raw = base64.b64decode(stored_hash)
        salt, key = raw[:16], raw[16:]
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
        return check == key
    except Exception:
        return False


# ── Auth context ──────────────────────────────────────────────────

@dataclass
class AuthContext:
    role: str = "api_key"        # "admin" | "patient" | "api_key"
    user_id: Optional[str] = None
    email: Optional[str] = None
    patient_id: Optional[str] = None


# ── JWT helpers ───────────────────────────────────────────────────

def _api_key_set() -> set[str]:
    load_environment()
    raw = os.getenv("CLAIMAGUARD_API_KEYS", "")
    if not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _jwt_secret() -> str:
    load_environment()
    s = os.getenv("CLAIMAGUARD_JWT_SECRET", "").strip()
    return s if s else _DEV_JWT_SECRET


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    patient_id: Optional[str] = None,
    expires_minutes: int = 60 * 8,
) -> str:
    if jwt is None:
        raise RuntimeError("PyJWT is required")
    now = datetime.now(timezone.utc)
    payload: dict = {
        "sub": user_id,
        "email": email,
        "role": role,
        "aud": "claimguard-api",
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    if patient_id:
        payload["patient_id"] = patient_id
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _decode_user_token(token: str) -> Optional[AuthContext]:
    if jwt is None:
        return None
    try:
        claims = jwt.decode(token, _jwt_secret(), algorithms=["HS256"], audience="claimguard-api")
        return AuthContext(
            role=claims.get("role", "patient"),
            user_id=claims.get("sub"),
            email=claims.get("email"),
            patient_id=claims.get("patient_id"),
        )
    except jwt.PyJWTError:
        return None


def verify_request_auth(request: Request) -> AuthContext:
    """
    Validates auth and returns AuthContext with role, user_id, email, patient_id.
    Accepts X-API-Key (treated as admin) or Authorization: Bearer <JWT>.
    """
    load_environment()
    keys = _api_key_set()

    if not keys:
        if not is_production_environment():
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                ctx = _decode_user_token(auth[7:].strip())
                if ctx:
                    return ctx
            return AuthContext(role="admin")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    api_key = request.headers.get("x-api-key")
    if api_key and api_key in keys:
        return AuthContext(role="admin")

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        ctx = _decode_user_token(auth[7:].strip())
        if ctx:
            return ctx

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
