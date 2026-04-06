"""
Application configuration: load environment, validate required variables, fail fast in production.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

_ENV_LOADED = False


def _env_file_path() -> Path:
    """`.env` next to this file (project root), not dependent on process cwd."""
    return Path(__file__).resolve().parent / ".env"


def load_environment() -> None:
    """Load .env from project root; safe to call multiple times."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(_env_file_path())
        _ENV_LOADED = True


def _is_production() -> bool:
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def is_production_environment() -> bool:
    """True when ``ENVIRONMENT=production`` (startup validation applies)."""
    load_environment()
    return _is_production()


def _is_test() -> bool:
    return os.getenv("ENVIRONMENT", "").strip().lower() == "test"


def _parse_csv_urls(raw: str | None) -> List[str]:
    if not raw or not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


_DEFAULT_SEPOLIA_RPC = "https://rpc.sepolia.org"
_PLACEHOLDER_PRIVATE_KEYS = frozenset({"your_private_key_here", "your_private_key"})


def get_sepolia_rpc_url() -> str:
    """
    Sepolia JSON-RPC HTTP(S) URL. Prefer Alchemy (ALCHEMY_URL), then SEPOLIA_RPC_URL, then a public fallback.
    """
    load_environment()
    for key in ("ALCHEMY_URL", "SEPOLIA_RPC_URL"):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return _DEFAULT_SEPOLIA_RPC


def get_sepolia_private_key() -> str:
    """
    Hex private key for signing (deploy / validateClaim). SEPOLIA_PRIVATE_KEY or PRIVATE_KEY alias.
    Returns empty string if unset or a known placeholder.
    """
    load_environment()
    for key in ("SEPOLIA_PRIVATE_KEY", "PRIVATE_KEY"):
        v = os.getenv(key, "").strip()
        if v.lower().startswith("0x"):
            v = v[2:]
        if not v or v in _PLACEHOLDER_PRIVATE_KEYS:
            continue
        return v
    return ""


def validate_required_settings() -> None:
    """
    Fail fast in production if required secrets or URLs are missing.
    No silent failures: missing values raise SystemExit with a clear message.
    """
    load_environment()
    if not _is_production():
        return

    missing: list[str] = []

    if not os.getenv("DOCUMENT_ENCRYPTION_KEY", "").strip():
        missing.append("DOCUMENT_ENCRYPTION_KEY")

    api_keys = _parse_csv_urls(os.getenv("CLAIMAGUARD_API_KEYS"))
    jwt_secret = os.getenv("CLAIMAGUARD_JWT_SECRET", "").strip()
    if not api_keys and not jwt_secret:
        missing.append("CLAIMAGUARD_API_KEYS or CLAIMAGUARD_JWT_SECRET")

    cors = _parse_csv_urls(os.getenv("CORS_ORIGINS"))
    if not cors:
        missing.append("CORS_ORIGINS (comma-separated; cannot be empty in production)")

    if not os.getenv("FIREBASE_PROJECT_ID", "").strip():
        missing.append("FIREBASE_PROJECT_ID")

    if missing:
        msg = (
            "Production configuration incomplete. Set the following environment variables:\n  - "
            + "\n  - ".join(missing)
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)


def get_cors_origins() -> List[str]:
    load_environment()
    raw = os.getenv("CORS_ORIGINS", "").strip()
    origins = _parse_csv_urls(raw)
    if _is_production() and not origins:
        raise RuntimeError("CORS_ORIGINS must be set in production")
    # Development / test: explicit localhost list (no "*")
    if not origins and (_is_test() or not _is_production()):
        return ["http://127.0.0.1", "http://localhost", "http://127.0.0.1:8000", "http://localhost:8000"]
    return origins


def parse_document_encryption_key(raw: str) -> bytes:
    """
    Resolve DOCUMENT_ENCRYPTION_KEY to exactly 32 bytes for AES-256-GCM.
    Accepts: 64-char hex, URL-safe base64 of 32 bytes, or UTF-8 string of exactly 32 characters.
    """
    s = raw.strip()
    if not s:
        raise ValueError("DOCUMENT_ENCRYPTION_KEY is empty")
    if len(s) == 64:
        try:
            out = bytes.fromhex(s)
            if len(out) == 32:
                return out
        except ValueError:
            pass
    try:
        import base64

        pad = "=" * (-len(s) % 4)
        decoded = base64.urlsafe_b64decode(s + pad)
        if len(decoded) == 32:
            return decoded
    except Exception:
        pass
    b = s.encode("utf-8")
    if len(b) == 32:
        return b
    raise ValueError(
        "DOCUMENT_ENCRYPTION_KEY must be 32 bytes: use 64 hex chars, base64, or a 32-char UTF-8 string"
    )


def cors_allow_credentials() -> bool:
    """
    Only enable credentials when origins are explicit (never with '*').
    """
    origins = get_cors_origins()
    return bool(origins) and "*" not in origins
