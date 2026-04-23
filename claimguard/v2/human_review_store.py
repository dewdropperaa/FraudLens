from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict

from claimguard.firebase_config import get_firestore_client, is_test_environment


@dataclass
class HumanReviewDocumentRef:
    claim_id: str
    file_name: str
    file_path: str
    token: str
    token_expires_at: int


class HumanReviewStore:
    def __init__(self) -> None:
        self._base_dir = Path(os.getenv("HUMAN_REVIEW_TMP_DIR", "claimguard_temp_docs")).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._token_secret = (os.getenv("HUMAN_REVIEW_TOKEN_SECRET", "") or "claimguard-human-review-secret").encode(
            "utf-8"
        )

    def save_temp_document(self, *, claim_id: str, document_part: Dict[str, Any]) -> HumanReviewDocumentRef | None:
        name = str(document_part.get("name") or "").strip()
        content_b64 = str(document_part.get("content_base64") or "").strip()
        if not name or not content_b64:
            return None
        try:
            content = base64.b64decode(content_b64, validate=True)
        except Exception:
            return None
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", " ")).strip() or "document.bin"
        claim_dir = self._base_dir / claim_id
        claim_dir.mkdir(parents=True, exist_ok=True)
        file_path = claim_dir / safe_name
        file_path.write_bytes(content)

        expires_at = int(time()) + 60 * 60
        token_payload = f"{claim_id}:{safe_name}:{expires_at}".encode("utf-8")
        token = hmac.new(self._token_secret, token_payload, hashlib.sha256).hexdigest()
        return HumanReviewDocumentRef(
            claim_id=claim_id,
            file_name=safe_name,
            file_path=str(file_path),
            token=token,
            token_expires_at=expires_at,
        )

    def verify_token(self, *, claim_id: str, file_name: str, token: str, expires_at: int) -> bool:
        if expires_at < int(time()):
            return False
        payload = f"{claim_id}:{file_name}:{expires_at}".encode("utf-8")
        expected = hmac.new(self._token_secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, str(token or ""))


class PendingHumanReviewRepository:
    def __init__(self) -> None:
        self._collection_name = os.getenv("HUMAN_REVIEW_COLLECTION", "pending_human_reviews").strip() or "pending_human_reviews"
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._firestore = None
        if not is_test_environment():
            try:
                self._firestore = get_firestore_client()
            except Exception:
                self._firestore = None

    def save(self, claim_id: str, payload: Dict[str, Any]) -> None:
        self._memory[claim_id] = dict(payload)
        if self._firestore is None:
            return
        self._firestore.collection(self._collection_name).document(claim_id).set(dict(payload), merge=True)

    def get(self, claim_id: str) -> Dict[str, Any] | None:
        in_mem = self._memory.get(claim_id)
        if in_mem is not None:
            return dict(in_mem)
        if self._firestore is None:
            return None
        snap = self._firestore.collection(self._collection_name).document(claim_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        if not isinstance(data, dict):
            return None
        self._memory[claim_id] = dict(data)
        return dict(data)

    def delete(self, claim_id: str) -> None:
        self._memory.pop(claim_id, None)
        if self._firestore is None:
            return
        self._firestore.collection(self._collection_name).document(claim_id).delete()
