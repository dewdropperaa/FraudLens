"""
Local document store for Human Review claims.

Documents uploaded with a claim are saved here when the pipeline routes
the claim to HUMAN_REVIEW.  They are deleted automatically once the admin
approves or rejects the claim, keeping no sensitive files on disk any
longer than necessary.

This module is intentionally self-contained: it does NOT touch the
orchestrator, agents, or scoring pipeline.
"""
from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("claimguard.doc_store")

_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "review_docs"
_lock = threading.Lock()


def _claim_dir(claim_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in claim_id)
    return _STORE_DIR / safe


def save_document(claim_id: str, filename: str, content_base64: str) -> Optional[str]:
    """
    Decode a base64 string and write it to disk under a per-claim directory.

    Returns the absolute file path on success, None on any failure.
    """
    if not claim_id or not content_base64:
        return None
    try:
        claim_dir = _claim_dir(claim_id)
        claim_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (
            "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in "-_. ")
            .strip()
        ) or "document.pdf"
        file_path = claim_dir / safe_name
        # Pad to a valid base64 length before decoding
        padded = content_base64 + "=" * ((4 - len(content_base64) % 4) % 4)
        data = base64.b64decode(padded)
        with _lock:
            file_path.write_bytes(data)
        LOGGER.info(
            "doc_store.saved claim_id=%s file=%s bytes=%d",
            claim_id, safe_name, len(data),
        )
        return str(file_path)
    except Exception as exc:
        LOGGER.warning("doc_store.save_failed claim_id=%s error=%s", claim_id, exc)
        return None


def get_document_path(claim_id: str) -> Optional[str]:
    """Return path to the first stored file for a claim, or None."""
    claim_dir = _claim_dir(claim_id)
    if not claim_dir.exists():
        return None
    for f in sorted(claim_dir.iterdir()):
        if f.is_file():
            return str(f)
    return None


def get_document_name(claim_id: str) -> Optional[str]:
    """Return the filename of the first stored document, or None."""
    path = get_document_path(claim_id)
    return Path(path).name if path else None


def delete_documents(claim_id: str) -> None:
    """Delete all stored documents for a claim (call after admin decision)."""
    claim_dir = _claim_dir(claim_id)
    if not claim_dir.exists():
        return
    with _lock:
        for f in list(claim_dir.iterdir()):
            try:
                f.unlink()
                LOGGER.debug("doc_store.unlinked file=%s", f)
            except Exception as exc:
                LOGGER.warning("doc_store.unlink_failed file=%s error=%s", f, exc)
        try:
            claim_dir.rmdir()
        except Exception:
            pass
    LOGGER.info("doc_store.deleted claim_id=%s", claim_id)
