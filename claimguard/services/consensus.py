from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from claimguard.config import get_sepolia_private_key
from claimguard.models import AgentResult, ClaimResult
from claimguard.crew.consensus import build_final_consensus_payload
from claimguard.services.ai_crew import run_claim_agents_async
from claimguard.services.storage import get_claim_store
from claimguard.services.blockchain import get_blockchain_service
from claimguard.services.ipfs import get_ipfs_service


# Higher weight reduces dilution from low-risk agents (policy/docs) on borderline cases.
DEFAULT_AGENT_WEIGHTS: Dict[str, float] = {
    "Anomaly Agent": 2.0,
    "Pattern Agent": 1.25,
    "Identity Agent": 1.0,
    "Document Agent": 1.0,
    "Policy Agent": 1.0,
    "Graph Agent": 2.0,
}


def _agent_weights() -> Dict[str, float]:
    raw = os.getenv("CONSENSUS_AGENT_WEIGHTS")
    if not raw:
        return dict(DEFAULT_AGENT_WEIGHTS)
    out: Dict[str, float] = dict(DEFAULT_AGENT_WEIGHTS)
    for part in raw.split(","):
        if "=" not in part:
            continue
        name, w = part.split("=", 1)
        name, w = name.strip(), w.strip()
        try:
            out[name] = float(w)
        except ValueError:
            continue
    return out


class ConsensusSystem:
    """
    Weighted consensus with critical-agent overrides:
    - Weighted average score (Graph / Anomaly weighted higher by default).
    - Reject if ANY agent returns decision=False.
    - Critical override: material graph fraud probability or strong anomaly signal
      can force REJECT even when the simple average would look acceptable.
    - Approve only if ALL approve AND weighted score >= threshold (default 75).
    - Extreme fraud veto (amount + graph FP) remains as defense in depth.
    """

    def __init__(self) -> None:
        self._store = get_claim_store()
        self._weights = _agent_weights()
        self._threshold = float(os.getenv("CONSENSUS_APPROVAL_THRESHOLD", "75"))
        self._graph_fp_force = float(os.getenv("GRAPH_FRAUD_FORCE_REJECT_FP", "0.12"))
        self._anomaly_score_force = float(os.getenv("ANOMALY_FORCE_REJECT_SCORE", "40"))
        self._blockchain_enabled = bool(
            os.getenv("CONTRACT_ADDRESS", "").strip() and get_sepolia_private_key()
        )
        self._ipfs_enabled = True

    @property
    def approval_threshold(self) -> float:
        """Minimum weighted consensus score (0–100) required for approval."""
        return self._threshold

    def _weighted_score(self, agent_results: List[AgentResult]) -> float:
        total_w = 0.0
        acc = 0.0
        for r in agent_results:
            w = float(self._weights.get(r.agent_name, 1.0))
            acc += w * float(r.score)
            total_w += w
        return round(acc / total_w, 2) if total_w else 0.0

    def _critical_agent_force_reject(self, agent_results: List[AgentResult]) -> bool:
        """
        Graph: material fraud probability forces reject (overrides weak averaging).
        Anomaly: explicit fail or very low score forces reject.
        """
        by_name = {r.agent_name: r for r in agent_results}
        graph = by_name.get("Graph Agent")
        if graph:
            fp = float(graph.details.get("fraud_probability", 0))
            if fp >= self._graph_fp_force:
                return True
        anomaly = by_name.get("Anomaly Agent")
        if anomaly:
            if not anomaly.decision:
                return True
            if float(anomaly.score) < self._anomaly_score_force:
                return True
        return False

    def _finalize(self, agent_results: List[AgentResult]) -> Tuple[str, float]:
        consensus_score = self._weighted_score(agent_results)

        if any(not r.decision for r in agent_results):
            return "REJECTED", consensus_score

        if self._critical_agent_force_reject(agent_results):
            return "REJECTED", consensus_score

        if consensus_score >= self._threshold:
            return "APPROVED", consensus_score

        return "REJECTED", consensus_score

    @staticmethod
    def _extreme_fraud_veto(
        claim_data: Dict[str, Any], agent_results: List[AgentResult]
    ) -> bool:
        amount = float(claim_data.get("amount") or 0)
        graph = next((r for r in agent_results if r.agent_name == "Graph Agent"), None)
        if not graph:
            return False
        fp = float(graph.details.get("fraud_probability", 0))
        insurance = claim_data.get("insurance", "")

        if amount >= 45_000 and fp >= 0.08:
            return True
        if insurance == "CNSS" and amount > 30_000 and fp >= 0.12:
            return True
        if insurance == "CNOPS" and amount > 50_000 and fp >= 0.12:
            return True
        return False

    def evaluate_consensus(
        self, claim_data: Dict[str, Any], agent_results: List[AgentResult]
    ) -> Tuple[str, float, bool]:
        """
        Return ``(decision, weighted_consensus_score, veto_applied)``.

        Applies the same rules as before: weighted score, unanimous approval,
        critical-agent overrides, then optional extreme-fraud veto.
        """
        decision, consensus_score = self._finalize(agent_results)
        veto_applied = False
        if decision == "APPROVED" and self._extreme_fraud_veto(claim_data, agent_results):
            decision = "REJECTED"
            veto_applied = True
        return decision, consensus_score, veto_applied

    async def process_claim(self, claim_data: Dict[str, Any]) -> ClaimResult:
        raw_in = claim_data.get("claim_id")
        if isinstance(raw_in, str):
            stripped = raw_in.strip()
            resolved_from_client = stripped if stripped else None
        else:
            resolved_from_client = raw_in
        claim_id = resolved_from_client if resolved_from_client else self._store.new_id()
        enriched = {**claim_data, "claim_id": claim_id}

        raw_agent_results = await run_claim_agents_async(enriched)
        agent_results = [AgentResult(**r) for r in raw_agent_results]

        decision, consensus_score, veto_applied = self.evaluate_consensus(
            enriched, agent_results
        )

        audit = build_final_consensus_payload(
            enriched,
            agent_results,
            final_decision=decision,
            weighted_score=consensus_score,
            veto_applied=veto_applied,
            consensus_threshold=self.approval_threshold,
        )
        logging.getLogger("claimguard.consensus").info(
            "consensus_audit %s",
            audit.model_dump(),
        )

        tx_hash: str | None = None
        ipfs_hash: str | None = None
        ipfs_hashes: List[str] = []
        claim_hash: str | None = None
        zk_proof_hash: str | None = None

        if decision == "APPROVED":
            ipfs_hashes = await self._upload_to_ipfs_async(claim_id, enriched)
            ipfs_hash = ipfs_hashes[0] if ipfs_hashes else None

            if self._blockchain_enabled:
                chain_result = await asyncio.to_thread(
                    lambda: get_blockchain_service().validate_claim_on_chain(
                        claim_id=claim_id,
                        score=consensus_score,
                        decision=decision,
                        ipfs_hashes=ipfs_hashes,
                        patient_id=enriched.get("patient_id", ""),
                    )
                )
                tx_hash = chain_result["tx_hash"]
                claim_hash = chain_result.get("claim_id_hash")
                zk_proof_hash = chain_result.get("zk_proof_hash")

        claim_result = ClaimResult(
            claim_id=claim_id,
            decision=decision,
            score=consensus_score,
            agent_results=agent_results,
            timestamp=datetime.now(timezone.utc),
            tx_hash=tx_hash,
            ipfs_hash=ipfs_hash,
            ipfs_hashes=ipfs_hashes,
            claim_hash=claim_hash,
            zk_proof_hash=zk_proof_hash,
        )
        self._store.put(claim_result)
        return claim_result

    @staticmethod
    async def _upload_to_ipfs_async(claim_id: str, claim_data: Dict[str, Any]) -> List[str]:
        extractions = claim_data.get("document_extractions") or []
        documents = claim_data.get("documents", [])
        payload_docs: List[Dict[str, Any]] = []

        if extractions:
            for idx, ex in enumerate(extractions):
                fn = ex.get("file_name") or f"doc_{idx}"
                text = ex.get("extracted_text") or ""
                if len(text) > 20_000:
                    text = text[:20_000] + "\n...[truncated for IPFS]"
                meta = {
                    "claim_id": claim_id,
                    "original_name": fn,
                    "extraction_method": ex.get("extraction_method"),
                    "char_count": ex.get("char_count"),
                    "error": ex.get("error"),
                }
                body = {
                    "extracted_text": text,
                    "meta": meta,
                }

                payload_docs.append(
                    {
                        "name": f"{idx}_{fn}.json",
                        "content": json.dumps(body, ensure_ascii=False),
                        "type": "extracted_document",
                    }
                )
        else:
            payload_docs = [
                {"name": f"doc_{idx}.json", "content": f"reference:{doc}", "type": "claim_document"}
                for idx, doc in enumerate(documents)
            ]

        if not payload_docs:
            payload_docs = [
                {"name": "claim_metadata.json", "content": f"claim:{claim_id}", "type": "metadata"}
            ]
        ipfs_hashes, _ = await get_ipfs_service().upload_claim_documents(claim_id, payload_docs)
        return ipfs_hashes

    def get_claim(self, claim_id: str) -> Optional[ClaimResult]:
        return self._store.get(claim_id)

    def list_claims(
        self,
        decision: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[ClaimResult], int]:
        if decision is None:
            return self._store.list_page(None, offset=offset, limit=limit)
        return self._store.list_page(decision, offset=offset, limit=limit)


_consensus_singleton: Optional[ConsensusSystem] = None


def get_consensus_system() -> ConsensusSystem:
    global _consensus_singleton
    if _consensus_singleton is None:
        _consensus_singleton = ConsensusSystem()
    return _consensus_singleton
