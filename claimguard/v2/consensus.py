from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger("claimguard.v2.consensus")

AGENT_WEIGHTS: Dict[str, float] = {
    "IdentityAgent": 0.1,
    "DocumentAgent": 0.15,
    "PolicyAgent": 0.2,
    "AnomalyAgent": 0.2,
    "PatternAgent": 0.15,
    "GraphRiskAgent": 0.2,
}


@dataclass
class Contradiction:
    agents: Tuple[str, str]
    penalty: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents),
            "H_penalty": round(self.penalty, 4),
            "reason": self.reason,
        }


class ConsensusEngine:
    def __init__(self, *, max_reflexive_retries: int = 3) -> None:
        self._max_reflexive_retries = max_reflexive_retries

    @staticmethod
    def _weighted_sum(entries: Dict[str, Dict[str, Any]]) -> float:
        return sum(
            AGENT_WEIGHTS.get(agent, 0.0) * float(payload.get("confidence", 0.0))
            for agent, payload in entries.items()
        )

    @staticmethod
    def _detect_contradictions(entries: Dict[str, Dict[str, Any]]) -> List[Contradiction]:
        contradictions: List[Contradiction] = []
        identity = entries.get("IdentityAgent", {})
        anomaly = entries.get("AnomalyAgent", {})
        policy = entries.get("PolicyAgent", {})
        document = entries.get("DocumentAgent", {})
        graph = entries.get("GraphRiskAgent", {})

        identity_ok = float(identity.get("score", 0.0)) >= 0.7
        anomaly_high = float(anomaly.get("score", 0.0)) >= 0.7
        if identity_ok and anomaly_high:
            contradictions.append(
                Contradiction(
                    agents=("IdentityAgent", "AnomalyAgent"),
                    penalty=0.22,
                    reason="Identity appears valid while anomaly risk is high.",
                )
            )

        policy_ok = float(policy.get("score", 0.0)) >= 0.65
        document_weak = float(document.get("score", 0.0)) <= 0.35
        if policy_ok and document_weak:
            contradictions.append(
                Contradiction(
                    agents=("PolicyAgent", "DocumentAgent"),
                    penalty=0.18,
                    reason="Policy checks pass but document evidence quality is weak.",
                )
            )

        anomaly_very_low = float(anomaly.get("score", 0.0)) <= 0.25
        graph_high = float(graph.get("score", 0.0)) >= 0.75
        if anomaly_very_low and graph_high:
            contradictions.append(
                Contradiction(
                    agents=("AnomalyAgent", "GraphRiskAgent"),
                    penalty=0.14,
                    reason="Behavioral anomaly low but relationship graph risk is high.",
                )
            )

        identity_score = float(identity.get("score", 0.0))
        document_score = float(document.get("score", 0.0))
        if identity_score >= 0.7 and document_score < 0.4:
            contradictions.append(
                Contradiction(
                    agents=("IdentityAgent", "DocumentAgent"),
                    penalty=0.15,
                    reason="Identity appears verified but documentation is critically insufficient.",
                )
            )

        policy_score = float(policy.get("score", 0.0))
        if policy_score >= 0.8 and document_score < 0.5:
            contradictions.append(
                Contradiction(
                    agents=("PolicyAgent", "DocumentAgent"),
                    penalty=0.12,
                    reason="Policy score is high but document evidence does not support the claim.",
                )
            )

        return contradictions

    @staticmethod
    def _nexus_truth_score(
        weighted_sum: float,
        contradictions: List[Contradiction],
    ) -> float:
        penalty_product = 1.0
        for contradiction in contradictions:
            penalty = max(0.0, min(1.0, contradiction.penalty))
            penalty_product *= 1.0 - penalty
        return max(0.0, min(100.0, weighted_sum * penalty_product * 100.0))

    @staticmethod
    def _decision_for_ts(ts: float) -> str:
        if ts > 90:
            return "AUTO_APPROVE"
        if 60 <= ts <= 90:
            return "HUMAN_REVIEW"
        return "REFLEXIVE_TRIGGER"

    @staticmethod
    def _extract_amount(claim_request: Dict[str, Any]) -> float:
        for bucket in ("metadata", "policy", "identity"):
            value = claim_request.get(bucket, {}).get("amount")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _compute_mahic_breakdown(
        self,
        claim_request: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        metadata = claim_request.get("metadata", {})
        policy = claim_request.get("policy", {})
        identity = claim_request.get("identity", {})
        amount = self._extract_amount(claim_request)
        doc_score = float(entries.get("DocumentAgent", {}).get("score", 0.0))
        anomaly_score = float(entries.get("AnomalyAgent", {}).get("score", 0.0))
        pattern_score = float(entries.get("PatternAgent", {}).get("score", 0.0))
        graph_score = float(entries.get("GraphRiskAgent", {}).get("score", 0.0))

        billing_risk = min(1.0, 0.5 * doc_score + 0.3 * anomaly_score + (0.2 if amount > 30000 else 0.05))
        clinical_risk = min(1.0, 0.45 * pattern_score + 0.35 * anomaly_score + 0.2 * float(policy.get("clinical_flags", 0) > 0))

        service_date = metadata.get("service_date") or policy.get("service_date")
        submitted_at = metadata.get("submitted_at")
        temporal_mismatch = 0.0
        if service_date and submitted_at:
            try:
                d1 = datetime.fromisoformat(str(service_date).replace("Z", "+00:00"))
                d2 = datetime.fromisoformat(str(submitted_at).replace("Z", "+00:00"))
                temporal_mismatch = 0.25 if (d2 - d1).days > 180 else 0.05
            except ValueError:
                temporal_mismatch = 0.1
        temporal_risk = min(1.0, 0.7 * anomaly_score + 0.3 * temporal_mismatch)

        claimant_geo = str(identity.get("country") or identity.get("region") or "").lower()
        policy_geo = str(policy.get("country") or policy.get("region") or "").lower()
        geo_mismatch = 0.25 if claimant_geo and policy_geo and claimant_geo != policy_geo else 0.05
        geo_risk = min(1.0, 0.75 * graph_score + 0.25 * geo_mismatch)

        return {
            "billing": round(0.35 * billing_risk * 100.0, 2),
            "clinical": round(0.30 * clinical_risk * 100.0, 2),
            "temporal": round(0.20 * temporal_risk * 100.0, 2),
            "geo": round(0.15 * geo_risk * 100.0, 2),
        }

    @staticmethod
    def _apply_self_correction(
        claim_request: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        corrected = {k: dict(v) for k, v in entries.items()}
        metadata = claim_request.get("metadata", {})
        manual_review = bool(metadata.get("manual_review"))
        has_docs = bool(claim_request.get("documents"))

        for agent_name, entry in corrected.items():
            confidence = float(entry.get("confidence", 0.0))
            score = float(entry.get("score", 0.0))
            if not has_docs and agent_name in {"DocumentAgent", "PolicyAgent"}:
                confidence = max(0.0, confidence - 0.1)
            if manual_review and agent_name in {"AnomalyAgent", "GraphRiskAgent"}:
                score = min(1.0, score + 0.05)
            if "rule" in str(entry.get("explanation", "")).lower() and agent_name == "PolicyAgent":
                confidence = min(1.0, confidence + 0.05)
            entry["confidence"] = round(max(0.0, min(1.0, confidence)), 4)
            entry["score"] = round(max(0.0, min(1.0, score)), 4)
        return corrected

    def evaluate(
        self,
        *,
        claim_request: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_entries = {k: dict(v) for k, v in entries.items()}
        score_evolution: List[float] = []
        retry_logs: List[Dict[str, Any]] = []
        retry_count = 0

        while True:
            weighted_sum = self._weighted_sum(current_entries)
            contradictions = self._detect_contradictions(current_entries)
            ts = round(self._nexus_truth_score(weighted_sum, contradictions), 2)
            score_evolution.append(ts)
            decision = self._decision_for_ts(ts)
            contradiction_payload = [c.to_dict() for c in contradictions]
            LOGGER.info(
                "consensus_eval retry=%s Ts=%.2f decision=%s contradictions=%s",
                retry_count,
                ts,
                decision,
                contradiction_payload,
            )

            if decision != "REFLEXIVE_TRIGGER":
                return {
                    "Ts": ts,
                    "decision": decision,
                    "retry_count": retry_count,
                    "mahic_breakdown": self._compute_mahic_breakdown(claim_request, current_entries),
                    "contradictions": contradiction_payload,
                    "score_evolution": score_evolution,
                    "retry_logs": retry_logs,
                    "entries": current_entries,
                }

            if retry_count >= self._max_reflexive_retries:
                return {
                    "Ts": ts,
                    "decision": "REJECTED",
                    "retry_count": retry_count,
                    "mahic_breakdown": self._compute_mahic_breakdown(claim_request, current_entries),
                    "contradictions": contradiction_payload,
                    "score_evolution": score_evolution,
                    "retry_logs": retry_logs,
                    "entries": current_entries,
                }

            retry_count += 1
            current_entries = self._apply_self_correction(claim_request, current_entries)
            retry_log = {
                "retry": retry_count,
                "reason": "Ts below threshold, running VerifierAgent self-correction.",
                "updated_entries": current_entries,
            }
            retry_logs.append(retry_log)
            LOGGER.info("reflexive_retry=%s details=%s", retry_count, retry_log)
