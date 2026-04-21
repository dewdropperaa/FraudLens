from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from claimguard.config import get_sepolia_private_key
from claimguard.models import AgentResult, ClaimResult
from claimguard.crew.consensus import build_final_consensus_payload
from claimguard.services.ai_crew import run_claim_agents_async
from claimguard.services.storage import get_claim_store
from claimguard.services.blockchain import get_blockchain_service
from claimguard.services.ipfs import get_ipfs_service


from claimguard.v2.flow_tracker import get_tracker

NEXUS_AGENT_WEIGHTS: Dict[str, float] = {
    "Identity Agent": 0.1,
    "Document Agent": 0.15,
    "Policy Agent": 0.2,
    "Anomaly Agent": 0.2,
    "Pattern Agent": 0.15,
    "Graph Agent": 0.2,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class ConsensusSystem:
    def __init__(self) -> None:
        self._store = get_claim_store()
        self._weights = dict(NEXUS_AGENT_WEIGHTS)
        self._max_reflexive_retries = 3
        self._blockchain_enabled = bool(
            os.getenv("CONTRACT_ADDRESS", "").strip() and get_sepolia_private_key()
        )
        self._ipfs_enabled = True

    def _weighted_sum(self, entries: Dict[str, Dict[str, Any]]) -> float:
        return sum(
            self._weights.get(agent, 0.0) * float(payload.get("confidence", 0.0))
            for agent, payload in entries.items()
        )

    @staticmethod
    def _detect_contradictions(entries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        contradictions: List[Dict[str, Any]] = []
        identity = entries.get("Identity Agent", {})
        anomaly = entries.get("Anomaly Agent", {})
        policy = entries.get("Policy Agent", {})
        document = entries.get("Document Agent", {})
        graph = entries.get("Graph Agent", {})

        if bool(identity.get("decision", False)) and (not bool(anomaly.get("decision", True))):
            contradictions.append(
                {
                    "agents": ["Identity Agent", "Anomaly Agent"],
                    "H_penalty": 0.22,
                    "reason": "Identity validated while anomaly engine raised high risk.",
                }
            )
        if bool(policy.get("decision", False)) and (not bool(document.get("decision", True))):
            contradictions.append(
                {
                    "agents": ["Policy Agent", "Document Agent"],
                    "H_penalty": 0.18,
                    "reason": "Policy appears compliant while documents fail consistency checks.",
                }
            )
        if float(anomaly.get("score", 100.0)) >= 75.0 and float(graph.get("score", 100.0)) <= 35.0:
            contradictions.append(
                {
                    "agents": ["Anomaly Agent", "Graph Agent"],
                    "H_penalty": 0.14,
                    "reason": "Behavioral anomaly appears low while graph risk remains high.",
                }
            )

        doc_score = float(document.get("score", 50.0))
        identity_score = float(identity.get("score", 50.0))
        if identity_score >= 70.0 and doc_score < 40.0:
            contradictions.append(
                {
                    "agents": ["Identity Agent", "Document Agent"],
                    "H_penalty": 0.15,
                    "reason": "Identity appears verified but documentation is critically insufficient.",
                }
            )

        policy_score = float(policy.get("score", 50.0))
        if policy_score >= 80.0 and doc_score < 50.0:
            contradictions.append(
                {
                    "agents": ["Policy Agent", "Document Agent"],
                    "H_penalty": 0.12,
                    "reason": "Policy score is high but document evidence does not support the claim.",
                }
            )

        return contradictions

    @staticmethod
    def _compute_ts(weighted_sum: float, contradictions: List[Dict[str, Any]]) -> float:
        penalty_product = 1.0
        for contradiction in contradictions:
            penalty_product *= 1.0 - _clamp01(float(contradiction.get("H_penalty", 0.0)))
        return round(_clamp01(weighted_sum * penalty_product) * 100.0, 2)

    @staticmethod
    def _decision_for_ts(ts: float) -> str:
        if ts > 75:
            return "AUTO_APPROVE"
        if 60 <= ts <= 75:
            return "HUMAN_REVIEW"
        return "REFLEXIVE_TRIGGER"

    @staticmethod
    def _to_entries(agent_results: List[AgentResult]) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for row in agent_results:
            entries[row.agent_name] = {
                "score": float(row.score),
                "confidence": _clamp01(float(row.score) / 100.0),
                "decision": bool(row.decision),
                "reasoning": row.reasoning,
                "details": dict(row.details or {}),
            }
        return entries

    def _mahic_breakdown(self, claim_data: Dict[str, Any], entries: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        amount = float(claim_data.get("amount") or 0.0)
        billing_raw = _clamp01(1.0 - (float(entries.get("Document Agent", {}).get("score", 50.0)) / 100.0))
        clinical_raw = _clamp01(1.0 - (float(entries.get("Policy Agent", {}).get("score", 50.0)) / 100.0))
        anomaly_risk = _clamp01(1.0 - (float(entries.get("Anomaly Agent", {}).get("score", 50.0)) / 100.0))
        graph_risk = _clamp01(1.0 - (float(entries.get("Graph Agent", {}).get("score", 50.0)) / 100.0))
        temporal_raw = _clamp01((0.7 * anomaly_risk) + (0.3 if amount > 30_000 else 0.05))
        geo_raw = _clamp01((0.75 * graph_risk) + 0.05)
        return {
            "billing": round(0.35 * billing_raw * 100.0, 2),
            "clinical": round(0.30 * clinical_raw * 100.0, 2),
            "temporal": round(0.20 * temporal_raw * 100.0, 2),
            "geo": round(0.15 * geo_raw * 100.0, 2),
        }

    def _apply_self_correction(
        self, claim_data: Dict[str, Any], entries: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        adjusted = {k: dict(v) for k, v in entries.items()}
        docs = claim_data.get("documents") or []
        history = claim_data.get("history") or []
        for name, entry in adjusted.items():
            score = float(entry.get("score", 50.0))
            confidence = float(entry.get("confidence", 0.5))
            if name in {"Document Agent", "Policy Agent"} and len(docs) < 2:
                score -= 8.0
            if name in {"Anomaly Agent", "Graph Agent"} and len(history) > 3:
                score -= 5.0
            if "limit" in str(entry.get("reasoning", "")).lower() and name == "Policy Agent":
                confidence = min(1.0, confidence + 0.05)
            entry["score"] = round(max(0.0, min(100.0, score)), 2)
            entry["confidence"] = round(_clamp01(confidence), 4)
            entry["decision"] = bool(entry["score"] >= 50.0)
        return adjusted

    def evaluate_consensus(self, claim_data: Dict[str, Any], agent_results: List[AgentResult]) -> Dict[str, Any]:
        entries = self._to_entries(agent_results)
        retry_count = 0
        retry_logs: List[Dict[str, Any]] = []
        score_evolution: List[float] = []

        while True:
            contradictions = self._detect_contradictions(entries)
            weighted_sum = self._weighted_sum(entries)
            ts = self._compute_ts(weighted_sum, contradictions)
            score_evolution.append(ts)
            consensus_decision = self._decision_for_ts(ts)
            logging.getLogger("claimguard.consensus").info(
                "nexus_eval retry=%s Ts=%.2f decision=%s contradictions=%s",
                retry_count,
                ts,
                consensus_decision,
                contradictions,
            )

            if consensus_decision != "REFLEXIVE_TRIGGER":
                break
            if retry_count >= self._max_reflexive_retries:
                consensus_decision = "REJECTED"
                break
            retry_count += 1
            entries = self._apply_self_correction(claim_data, entries)
            retry_log = {
                "retry": retry_count,
                "reason": "Ts < 60 triggered reflexive verifier.",
                "updated_entries": entries,
            }
            retry_logs.append(retry_log)
            logging.getLogger("claimguard.consensus").info("reflexive_retry=%s details=%s", retry_count, retry_log)

        if consensus_decision == "AUTO_APPROVE":
            final_decision = "APPROVED"
        elif consensus_decision == "HUMAN_REVIEW":
            final_decision = "PENDING"
        else:
            final_decision = "REJECTED"
        return {
            "decision": final_decision,
            "consensus_decision": consensus_decision,
            "score": ts,
            "veto_applied": False,
            "Ts": ts,
            "retry_count": retry_count,
            "mahic_breakdown": self._mahic_breakdown(claim_data, entries),
            "contradictions": contradictions,
            "score_evolution": score_evolution,
            "retry_logs": retry_logs,
        }

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

        tracker = get_tracker(claim_id)
        tracker.update("Consensus", "RUNNING")

        consensus = self.evaluate_consensus(enriched, agent_results)
        decision = consensus["decision"]
        consensus_score = float(consensus["score"])
        veto_applied = bool(consensus["veto_applied"])
        
        tracker.update("Consensus", "COMPLETED")
        if consensus["consensus_decision"] == "HUMAN_REVIEW":
            tracker.update("HumanReview", "RUNNING")
        else:
            tracker.update("HumanReview", "SKIPPED")

        audit = build_final_consensus_payload(
            enriched,
            agent_results,
            final_decision=decision,
            weighted_score=consensus_score,
            veto_applied=veto_applied,
            consensus_threshold=90.0,
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
            try:
                ipfs_hashes = await self._upload_to_ipfs_async(claim_id, enriched)
                ipfs_hash = ipfs_hashes[0] if ipfs_hashes else None
            except Exception as exc:
                logging.getLogger("claimguard.consensus").warning(
                    "ipfs_upload_failed claim_id=%s error=%s",
                    claim_id,
                    exc,
                )
                ipfs_hashes = []
                ipfs_hash = None

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
            consensus_decision=consensus["consensus_decision"],
            Ts=consensus["Ts"],
            retry_count=consensus["retry_count"],
            mahic_breakdown=consensus["mahic_breakdown"],
            contradictions=consensus["contradictions"],
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
