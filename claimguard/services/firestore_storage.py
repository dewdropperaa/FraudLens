"""Cloud Firestore — sole persistent claim store outside ``ENVIRONMENT=test``."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4

from google.cloud.firestore import Query, SERVER_TIMESTAMP  # type: ignore[import]

from claimguard.firebase_config import get_firestore_client
from claimguard.models import ClaimResult


def _payload_to_claim(raw: dict) -> ClaimResult:
    """Normalize Firestore payload (e.g. timestamps) for Pydantic."""
    if isinstance(raw.get("timestamp"), datetime):
        d = dict(raw)
        ts = d["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        d["timestamp"] = ts
        return ClaimResult.model_validate(d)
    return ClaimResult.model_validate(raw)


class FirestoreClaimStore:
    """Claim storage in Cloud Firestore (collection configurable via FIRESTORE_COLLECTION)."""

    def __init__(self) -> None:
        self._db = get_firestore_client()
        self._coll_name = os.getenv("FIRESTORE_COLLECTION", "claims").strip() or "claims"

    def _col(self):
        return self._db.collection(self._coll_name)

    def put(self, claim: ClaimResult) -> str:
        payload = claim.model_dump(mode="json")
        ref = self._col().document(claim.claim_id)
        snap = ref.get()
        if snap.exists:
            ref.set({"decision": claim.decision, "payload": payload}, merge=True)
        else:
            ref.set(
                {
                    "decision": claim.decision,
                    "payload": payload,
                    "created_at": SERVER_TIMESTAMP,
                }
            )
        return claim.claim_id

    def get(self, claim_id: str) -> Optional[ClaimResult]:
        snap = self._col().document(claim_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        raw = data.get("payload")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            return None
        return _payload_to_claim(raw)

    def new_id(self) -> str:
        return str(uuid4())

    def _count_matching(self, decision: str | None) -> int:
        col = self._col()
        if decision is None:
            q = col
        else:
            q = col.where("decision", "==", decision.upper())
        try:
            results = q.count().get()
            if results:
                return int(results[0].value)
        except Exception:
            pass
        return sum(1 for _ in q.stream())

    def update_decision(self, claim_id: str, decision: str, reviewer_id: str) -> bool:
        ref = self._col().document(claim_id)
        snap = ref.get()
        if not snap.exists:
            return False
        dn = decision.upper()
        d = snap.to_dict() or {}
        payload = d.get("payload")
        if isinstance(payload, dict):
            payload["decision"] = dn
        ref.set(
            {
                "decision": dn,
                "payload": payload,
                "reviewed_by": reviewer_id,
                "reviewed_at": SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return True

    def list_page(
        self,
        decision: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[ClaimResult], int]:
        offset = max(0, offset)
        limit = min(max(1, limit), 500)

        col = self._col()

        # Fetch all docs, then filter and sort in Python.
        # This avoids any composite index requirements in Firestore.
        all_docs = list(col.stream())

        if decision is not None:
            dn = decision.upper()
            all_docs = [d for d in all_docs if (d.to_dict() or {}).get("decision", "").upper() == dn]

        all_docs.sort(
            key=lambda d: (d.to_dict() or {}).get("created_at") or datetime.min,
            reverse=True,
        )

        total = len(all_docs)
        docs = all_docs

        paged = docs[offset: offset + limit]
        items: List[ClaimResult] = []
        for doc in paged:
            d = doc.to_dict() or {}
            raw = d.get("payload")
            if not isinstance(raw, dict):
                continue
            items.append(_payload_to_claim(raw))
        return items, total
