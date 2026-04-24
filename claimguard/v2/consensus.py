from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

from claimguard.v2.schemas import AgentOutput

LOGGER = logging.getLogger("claimguard.v2.consensus")

AGENT_WEIGHTS: Dict[str, float] = {
    "IdentityAgent": 0.30,  # SCORE-FIX
    "DocumentAgent": 0.20,  # SCORE-FIX
    "PolicyAgent": 0.20,  # SCORE-FIX
    "AnomalyAgent": 0.15,  # SCORE-FIX
    "PatternAgent": 0.10,  # SCORE-FIX
    "GraphRiskAgent": 0.05,  # SCORE-FIX
}


def calculate_weighted_score(agent_results: Dict[str, Dict[str, Any]]) -> float:
    total_weight = 0.0  # SCORE-FIX
    weighted_sum = 0.0  # SCORE-FIX
    used_agents: List[str] = []  # SCORE-FIX
    for agent_name, result in (agent_results or {}).items():
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "ERROR")).upper()
        if status in {"ERROR", "TIMEOUT"}:
            continue
        if result.get("score", None) is None:
            raise ValueError(f"{agent_name} missing score")
        score_value = float(result["score"])
        score_0_100 = score_value * 100.0 if score_value <= 1.0 else score_value
        weight = AGENT_WEIGHTS.get(agent_name, 0.10)
        weighted_sum += score_0_100 * weight
        total_weight += weight
        used_agents.append(agent_name)
    if total_weight == 0:
        return 50.0
    raw_score = weighted_sum / total_weight
    LOGGER.info(
        "[CONSENSUS] weighted_score=%s total_weight=%s agents_used=%s",  # SCORE-FIX
        round(raw_score, 2),
        round(total_weight, 2),
        used_agents,
    )
    return round(raw_score, 2)


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


@dataclass(frozen=True)
class ConsensusConfig:
    # CALIBRATION-FIX: relaxed calibration to avoid false-positive human review routing.
    approved_threshold: float = 65.0
    human_review_min: float = 45.0
    human_review_max: float = 64.0
    rejected_threshold: float = 44.0
    stability_required_delta: float = 8.0
    hallucination_confidence_floor: float = 0.7
    max_contradiction_threshold: float = 0.3
    auto_approve_threshold: float = 75.0
    degraded_memory_penalty: float = 0.15
    unavailable_memory_penalty: float = 0.25


class FlagTier(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warnings"
    INFORMATIONAL = "informational"


@dataclass(frozen=True)
class FlagEvent:
    flag: str
    tier: FlagTier
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "flag": self.flag,
            "tier": self.tier.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class FlagRegistry:
    # PROD-FIX: central flag registry with tiered auditability.
    def __init__(self) -> None:
        self._by_tier: Dict[FlagTier, List[str]] = {
            FlagTier.BLOCKING: [],
            FlagTier.WARNING: [],
            FlagTier.INFORMATIONAL: [],
        }
        self._events: List[FlagEvent] = []

    def emit(self, flag: str, tier: FlagTier, reason: str) -> None:
        clean_flag = str(flag or "").strip().upper()
        if not clean_flag:
            return
        if clean_flag not in self._by_tier[tier]:
            self._by_tier[tier].append(clean_flag)
        self._events.append(
            FlagEvent(
                flag=clean_flag,
                tier=tier,
                reason=str(reason or "").strip() or "No reason provided",
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
        )

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "blocking": list(self._by_tier[FlagTier.BLOCKING]),
            "warnings": list(self._by_tier[FlagTier.WARNING]),
            "informational": list(self._by_tier[FlagTier.INFORMATIONAL]),
        }

    def audit_trail(self) -> List[Dict[str, str]]:
        return [event.to_dict() for event in self._events]


class ScoreCalibrator:
    # CALIBRATION-FIX: ensure infra degradation does not masquerade as fraud.
    def calibrate(
        self,
        *,
        base_ts: float,
        flags: Dict[str, List[str]],
        error_agents: List[str],
        all_agents_done: bool,
        fraud_signals: int,
        all_critical_fields_verified: bool,
        doc_type_correct: bool,
        unverified_critical_fields: int,
        unverified_non_critical: int,
        no_tier1_blocking_flags: bool,
        tool_failures: int,
        cin_found: bool,
        ipp_found: bool,
        amount_found: bool,
        injection_detected: bool,
        cin_format_match: bool,
    ) -> float:
        score = float(base_ts)
        score -= 25.0 * len(flags.get("blocking", []))
        warning_penalty = 0.0
        if "DECISION_STABILITY_FAIL" in flags.get("warnings", []):
            # CALIBRATION-FIX: stability warning can contribute at most -5 and never force outcome.
            warning_penalty += 5.0
        score -= min(warning_penalty, 5.0)
        score -= 8.0 * max(0, int(tool_failures))
        if all_agents_done and len(error_agents) == 0:
            score += 6.0  # CALIBRATION-FIX
        if fraud_signals == 0:
            score += 5.0  # CALIBRATION-FIX
        if cin_found and ipp_found and amount_found:
            score += 4.0  # CALIBRATION-FIX
        if cin_format_match:
            score += 2.0  # CALIBRATION-FIX: Moroccan CIN pattern bonus.
        if doc_type_correct:
            score += 3.0  # CALIBRATION-FIX
        if no_tier1_blocking_flags:
            score += 4.0  # CALIBRATION-FIX
        if not injection_detected:
            score += 3.0  # CALIBRATION-FIX
        if all_critical_fields_verified:
            score += 2.0
        if unverified_non_critical > 1:
            score -= 1.0
        if unverified_critical_fields > 0:
            score -= min(5.0, float(unverified_critical_fields) * 3.0)
        if not (cin_found and ipp_found and amount_found):
            # CALIBRATION-FIX: penalize unverified fields only when critical identity signals are missing.
            if not cin_found or not ipp_found:
                score -= 3.0
        if (
            all_agents_done
            and fraud_signals == 0
            and int(tool_failures) == 0
            and no_tier1_blocking_flags
        ):
            score = max(score, 68.0)  # CALIBRATION-FIX: clean signal floor.
        return max(0.0, min(100.0, round(score, 2)))


def should_force_human_review(
    agent_outputs: List[AgentOutput],
    blackboard: Dict[str, Any],
    config: ConsensusConfig,
) -> tuple[bool, str]:
    hallucination_agents: List[str] = []
    low_confidence_ungrounded_agents: List[str] = []
    for output in agent_outputs:
        hallucination_detected = bool(output.hallucination_flags)
        if hallucination_detected:
            hallucination_agents.append(output.agent)

        evidence_grounded = any(bool(claim.verified) for claim in output.claims)
        if output.confidence < config.hallucination_confidence_floor and not evidence_grounded:
            low_confidence_ungrounded_agents.append(output.agent)

    contradiction_penalty = 0.0
    contradictions = blackboard.get("contradictions", [])
    if isinstance(contradictions, list):
        for row in contradictions:
            if isinstance(row, dict):
                contradiction_penalty += float(row.get("H_penalty", 0.0))

    if len(hallucination_agents) >= 2:
        return True, (
            "Hallucination guard: multiple agents produced hallucination flags "
            f"({', '.join(sorted(hallucination_agents))})."
        )
    if low_confidence_ungrounded_agents:
        return True, (
            "Hallucination guard: low-confidence ungrounded output from "
            f"{', '.join(sorted(low_confidence_ungrounded_agents))}."
        )
    if contradiction_penalty > config.max_contradiction_threshold:
        return True, (
            "Hallucination guard: contradiction penalty exceeded threshold "
            f"({contradiction_penalty:.3f}>{config.max_contradiction_threshold:.3f})."
        )
    return False, (
        "Hallucination guard: no force condition met "
        f"(hallucination_agents={len(hallucination_agents)}, "
        f"low_confidence_ungrounded={len(low_confidence_ungrounded_agents)}, "
        f"contradiction_penalty={contradiction_penalty:.3f})."
    )


class ConsensusEngine:
    def __init__(self, *, max_reflexive_retries: int = 3) -> None:
        self._max_reflexive_retries = max_reflexive_retries

    @staticmethod
    def _weighted_sum(entries: Dict[str, Dict[str, Any]]) -> float:
        weighted = 0.0
        for agent, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status", "DONE")).upper()
            if status != "DONE":
                continue
            if payload.get("score", None) is None:
                raise ValueError(f"{agent} missing score")
            raw_score = float(payload["score"])
            score_0_100 = raw_score * 100.0 if raw_score <= 1.0 else raw_score
            weighted += AGENT_WEIGHTS.get(agent, 0.0) * max(0.0, min(100.0, score_0_100))
        return weighted

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
        normalizer = sum(AGENT_WEIGHTS.values()) or 1.0
        normalized = weighted_sum / normalizer
        return max(0.0, min(100.0, normalized * penalty_product))

    @staticmethod
    def _detect_conflicts(entries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        identity = entries.get("IdentityAgent", {})
        document = entries.get("DocumentAgent", {})
        anomaly = entries.get("AnomalyAgent", {})

        identity_valid = bool(identity.get("is_valid") or identity.get("decision") is True)
        document_mismatch = bool(
            document.get("document_mismatch")
            or document.get("mismatch")
            or "mismatch" in str(document.get("explanation", "")).lower()
        )
        if identity_valid and document_mismatch:
            conflicts.append(
                {
                    "type": "IDENTITY_DOCUMENT_MISMATCH",
                    "reason": "Identity valid while document agent reports mismatch.",
                    "penalty": 8.0,
                }
            )

        anomaly_score = float(anomaly.get("score", 0.0))
        identity_score = float(identity.get("score", 0.0))
        anomaly_score = anomaly_score * 100.0 if anomaly_score <= 1.0 else anomaly_score
        identity_score = identity_score * 100.0 if identity_score <= 1.0 else identity_score
        if anomaly_score >= 80.0 and identity_score >= 80.0:
            conflicts.append(
                {
                    "type": "HIGH_FRAUD_HIGH_IDENTITY_TRUST",
                    "reason": "High anomaly/fraud score conflicts with high identity trust.",
                    "penalty": 6.0,
                }
            )
        return conflicts

    @staticmethod
    def _decision_for_ts(ts: float, config: ConsensusConfig) -> str:
        # CALIBRATION-FIX: threshold windows align to revised risk policy.
        if ts >= config.approved_threshold:
            return "APPROVED"
        if config.human_review_min <= ts <= config.human_review_max:
            return "HUMAN_REVIEW"
        if ts <= config.rejected_threshold:
            return "REJECTED"
        return "HUMAN_REVIEW"

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
        blackboard: Dict[str, Any] | None = None,
        config: ConsensusConfig | None = None,
    ) -> Dict[str, Any]:
        config = config or ConsensusConfig()
        current_entries = {k: dict(v) for k, v in entries.items()}
        board = dict(blackboard or {})
        memory_degraded = bool(board.get("memory_degraded"))
        memory_status = str(board.get("memory_status", "")).upper()
        if memory_degraded:
            # CALIBRATION-FIX: degraded memory is informational only (no score/confidence penalty).
            LOGGER.warning("memory_degraded informational mode: memory_status=%s", memory_status)
        score_evolution: List[float] = []
        retry_logs: List[Dict[str, Any]] = []
        retry_count = 0
        error_agents = [
            name for name, payload in current_entries.items()
            if isinstance(payload, dict) and str(payload.get("status", "DONE")).upper() != "DONE"
        ]
        success_agents = [
            name for name, payload in current_entries.items()
            if isinstance(payload, dict) and str(payload.get("status", "DONE")).upper() == "DONE"
        ]
        failure_ratio = (len(error_agents) / max(1, len(error_agents) + len(success_agents)))
        too_many_error_agents = failure_ratio >= 0.5

        while True:
            weighted_base_ts = calculate_weighted_score(current_entries)  # SCORE-FIX
            contradictions = self._detect_contradictions(current_entries)
            ts = round(weighted_base_ts, 2)  # SCORE-FIX
            conflicts = self._detect_conflicts(current_entries)
            if conflicts:
                ts = max(0.0, ts - sum(float(item.get("penalty", 0.0)) for item in conflicts))
            score_evolution.append(ts)
            decision = self._decision_for_ts(ts, config)
            contradiction_payload = [c.to_dict() for c in contradictions]
            flag_registry = FlagRegistry()
            _meta = entries.get("_meta", {}) if isinstance(entries.get("_meta", {}), dict) else {}
            fraud_signals = int(_meta.get("fraud_signals", 0) or 0)
            unverified_critical_fields = int(_meta.get("unverified_critical_fields", 0) or 0)
            unverified_non_critical = int(_meta.get("unverified_non_critical", 0) or 0)
            all_critical_fields_verified = bool(_meta.get("all_critical_fields_verified", False))
            doc_type_correct = bool(_meta.get("document_type_correct", True))
            layer2_disabled = bool(_meta.get("layer2_disabled", False))
            layer1_triggered = bool(_meta.get("layer1_triggered", False))
            cin_found = bool(_meta.get("cin_found", False))
            ipp_found = bool(_meta.get("ipp_found", False))
            amount_found = bool(_meta.get("amount_found", False))
            injection_detected = bool(_meta.get("injection_detected", False))
            cin_format_match = bool(_meta.get("cin_format_match", False))
            tier1_blocking_flag_count = int(_meta.get("tier1_blocking_flag_count", 0) or 0)
            all_agents_done = len(success_agents) > 0 and len(error_agents) == 0
            if layer2_disabled and not layer1_triggered:
                flag_registry.emit("DEGRADED_SECURITY_MODE", FlagTier.INFORMATIONAL, "Layer2 disabled without layer1 trigger.")
            if conflicts:
                flag_registry.emit("AGENT_CONFLICT", FlagTier.WARNING, "Agent contradiction conflict detected.")
            if too_many_error_agents:
                flag_registry.emit("TOO_MANY_ERROR_AGENTS", FlagTier.BLOCKING, "More than half of agents errored.")
            if memory_status == "DISABLED":
                flag_registry.emit("MEMORY_DISABLED", FlagTier.INFORMATIONAL, "Memory embedding/index unavailable.")
            if unverified_critical_fields > 0 and (not cin_found or not ipp_found):
                flag_registry.emit("UNVERIFIED_FIELDS_PRESENT", FlagTier.WARNING, "Critical field verification incomplete.")

            ts = ScoreCalibrator().calibrate(
                base_ts=ts,
                flags=flag_registry.as_dict(),
                error_agents=error_agents,
                all_agents_done=all_agents_done,
                fraud_signals=fraud_signals,
                all_critical_fields_verified=all_critical_fields_verified,
                doc_type_correct=doc_type_correct,
                unverified_critical_fields=unverified_critical_fields,
                unverified_non_critical=unverified_non_critical,
                no_tier1_blocking_flags=tier1_blocking_flag_count == 0,
                tool_failures=len(error_agents),
                cin_found=cin_found,
                ipp_found=ipp_found,
                amount_found=amount_found,
                injection_detected=injection_detected,
                cin_format_match=cin_format_match,
            )
            decision = self._decision_for_ts(ts, config)

            if not success_agents:
                decision = "HUMAN_REVIEW"
            elif conflicts or too_many_error_agents:
                decision = "HUMAN_REVIEW"
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
                    "conflicts": conflicts,
                    "error_agents": error_agents,
                    "success_agents": success_agents,
                    "too_many_error_agents": too_many_error_agents,
                    "flag_registry": flag_registry.as_dict(),
                    "audit_trail": flag_registry.audit_trail(),
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
                    "conflicts": conflicts,
                    "error_agents": error_agents,
                    "success_agents": success_agents,
                    "too_many_error_agents": too_many_error_agents,
                    "flag_registry": flag_registry.as_dict(),
                    "audit_trail": flag_registry.audit_trail(),
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
