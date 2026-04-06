"""
Firebase Admin SDK — Firestore is the sole claim store outside automated tests.

Tests set ``ENVIRONMENT=test`` and use an in-memory store; they must not initialize
this client.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from claimguard.config import load_environment

_firestore_client: Any = None
_initialized = False


def is_test_environment() -> bool:
    load_environment()
    return os.getenv("ENVIRONMENT", "").strip().lower() == "test"


def get_firestore_client():
    """
    Lazy-init Firestore client. Requires ``FIREBASE_PROJECT_ID`` and credentials
    (``GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials).
    """
    global _firestore_client, _initialized
    if _initialized:
        return _firestore_client

    load_environment()
    if is_test_environment():
        raise RuntimeError(
            "Firestore must not be initialized when ENVIRONMENT=test; use MemoryClaimStore."
        )

    try:
        import firebase_admin
        from firebase_admin import credentials, initialize_app
    except ImportError as e:
        raise RuntimeError(
            "firebase_admin is not installed. Add firebase-admin to requirements."
        ) from e

    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is required for Firestore claim storage")

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if cred_path:
        cred = credentials.Certificate(cred_path)
    else:
        cred = credentials.ApplicationDefault()

    initialize_app(cred, {"projectId": project_id})
    from firebase_admin import firestore

    _firestore_client = firestore.client()
    _initialized = True
    return _firestore_client


def get_firebase_web_config() -> dict[str, Optional[str]]:
    """
    Client-side Firebase config (non-secret identifiers only). Do not log or expose in error messages.
    """
    load_environment()
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    }
