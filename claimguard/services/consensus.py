from __future__ import annotations

from typing import List, Optional

from claimguard.models import ClaimResult
from claimguard.services.storage import get_claim_store

_LEGACY_DISABLED_MESSAGE = "LEGACY CONSENSUS DISABLED — USE claimguard.v2.orchestrator.run_pipeline_v2"


class ConsensusSystem:
    """Deprecated compatibility shim for removed legacy consensus pipeline."""

    def __init__(self) -> None:
        self._store = get_claim_store()

    async def process_claim(self, claim_data: dict) -> ClaimResult:
        raise Exception(_LEGACY_DISABLED_MESSAGE)

    def evaluate_consensus(self, claim_data: dict, agent_results: list) -> dict:
        raise Exception(_LEGACY_DISABLED_MESSAGE)

    def get_claim(self, claim_id: str) -> Optional[ClaimResult]:
        return self._store.get(claim_id)

    def list_claims(
        self,
        decision: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[ClaimResult], int]:
        return self._store.list_page(decision, offset=offset, limit=limit)


_consensus_singleton: Optional[ConsensusSystem] = None


def get_consensus_system() -> ConsensusSystem:
    global _consensus_singleton
    if _consensus_singleton is None:
        _consensus_singleton = ConsensusSystem()
    return _consensus_singleton
