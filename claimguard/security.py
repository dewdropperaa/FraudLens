"""
API authentication: X-API-Key and optional JWT (HS256).
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request, status

from claimguard.config import is_production_environment, load_environment

try:
    import jwt
except ImportError:  # pragma: no cover
    jwt = None


def _api_key_set() -> set[str]:
    load_environment()
    raw = os.getenv("CLAIMAGUARD_API_KEYS", "")
    if not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _jwt_secret() -> Optional[str]:
    load_environment()
    s = os.getenv("CLAIMAGUARD_JWT_SECRET", "").strip()
    return s if s else None


def verify_request_auth(request: Request) -> None:
    """
    Require a valid X-API-Key or Authorization: Bearer <JWT> when auth is configured.
    In production, configuration validation ensures at least one method exists.
    """
    load_environment()
    keys = _api_key_set()
    secret = _jwt_secret()

    if not keys and not secret:
        # Production startup requires keys; in dev/staging with no secrets, allow local UI + API use.
        if not is_production_environment():
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    api_key = request.headers.get("x-api-key")
    if api_key and api_key in keys:
        return

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer ") and secret:
        token = auth[7:].strip()
        if jwt is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="JWT support requires PyJWT",
            )
        try:
            jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="claimguard-api",
            )
            return
        except jwt.PyJWTError:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
