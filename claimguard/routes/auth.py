"""
User authentication endpoints: /auth/login, /auth/me.
Users are stored in Firestore collection 'users'.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from claimguard.security import (
    AuthContext,
    create_access_token,
    verify_password,
    verify_request_auth,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Pydantic schemas ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    full_name: str | None = None
    patient_id: str | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    full_name: str | None = None
    patient_id: str | None = None


# ── Firestore helpers ─────────────────────────────────────────────

def _get_users_col():
    from claimguard.firebase_config import get_firestore_client
    return get_firestore_client().collection("users")


def _find_user_by_email(email: str) -> tuple[str, dict] | None:
    """Return (doc_id, data) or None."""
    docs = _get_users_col().where("email", "==", email).limit(1).stream()
    for doc in docs:
        return doc.id, doc.to_dict()
    return None


def _get_user_by_id(user_id: str) -> dict | None:
    doc = _get_users_col().document(user_id).get()
    return doc.to_dict() if doc.exists else None


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    result = _find_user_by_email(body.email.lower().strip())
    if not result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    doc_id, user = result
    if not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(
        user_id=doc_id,
        email=user["email"],
        role=user.get("role", "patient"),
        patient_id=user.get("patient_id"),
    )
    return TokenResponse(
        access_token=token,
        role=user.get("role", "patient"),
        email=user["email"],
        full_name=user.get("full_name"),
        patient_id=user.get("patient_id"),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(auth: AuthContext = Depends(verify_request_auth)) -> UserResponse:
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="Not authenticated as a named user")

    user = _get_user_by_id(auth.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=auth.user_id,
        email=user["email"],
        role=user.get("role", "patient"),
        full_name=user.get("full_name"),
        patient_id=user.get("patient_id"),
    )
