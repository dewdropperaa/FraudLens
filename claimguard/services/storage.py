from __future__ import annotations

import os
from typing import List, Optional, Protocol, Tuple
from uuid import uuid4

from claimguard.firebase_config import is_test_environment
from claimguard.models import ClaimResult


class ClaimStoreProtocol(Protocol):
    def put(self, claim: ClaimResult) -> str: ...

    def get(self, claim_id: str) -> Optional[ClaimResult]: ...

    def new_id(self) -> str: ...

    def list_page(
        self,
        decision: str | None,
        offset: int,
        limit: int,
    ) -> Tuple[List[ClaimResult], int]: ...


class MemoryClaimStore:
    """In-process claim storage for ``ENVIRONMENT=test`` (pytest) only."""

    def __init__(self) -> None:
        self._claims: dict[str, ClaimResult] = {}

    def put(self, claim: ClaimResult) -> str:
        self._claims[claim.claim_id] = claim
        return claim.claim_id

    def get(self, claim_id: str) -> Optional[ClaimResult]:
        return self._claims.get(claim_id)

    def new_id(self) -> str:
        return str(uuid4())

    def list_page(
        self,
        decision: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[ClaimResult], int]:
        offset = max(0, offset)
        limit = min(max(1, limit), 500)
        items = list(self._claims.values())
        if decision is not None:
            dn = decision.upper()
            items = [c for c in items if c.decision.upper() == dn]
        items.sort(key=lambda c: c.timestamp, reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        return page, total


_store: ClaimStoreProtocol | None = None


def get_claim_store() -> ClaimStoreProtocol:
    """
    Claim persistence: **Cloud Firestore** in all non-test environments.

    Tests use :class:`MemoryClaimStore` so CI does not require Firebase credentials.
    """
    global _store
    if _store is None:
        if is_test_environment():
            _store = MemoryClaimStore()
        else:
            from claimguard.services.firestore_storage import FirestoreClaimStore

            _store = FirestoreClaimStore()
    return _store


def reset_claim_store_for_tests() -> None:
    """Clear singleton (tests only)."""
    global _store
    _store = None
