"""
Post-crew consensus helpers: Pydantic views, audit explainability, legacy API mapping.

Business rules remain in ``ConsensusSystem`` (weighted score, critical overrides, veto).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from claimguard.crew.models import AgentDecisionOutput, FinalConsensusPayload
from claimguard.models import AgentResult

logger = logging.getLogger("claimguard.crew.consensus")

_AGENT_ORDER = (
    "Anomaly Agent",
    "Pattern Agent",
    "Identity Agent",
    "Document Agent",
    "Policy Agent",
    "Graph Agent",
)


def _audit_explainability(legacy: dict[str, Any]) -> str:
    """Human-readable audit string derived from deterministic agent output."""
    name = legacy.get("agent_name", "")
    details = legacy.get("details") or {}
    parts = [
        f"agent={name}",
        "rule_path=deterministic_analyze",
    ]
    if name == "Graph Agent":
        fp = details.get("fraud_probability")
        if fp is not None:
            parts.append(f"graph_fraud_probability={fp}")
        pat = details.get("pattern_detected")
        if pat:
            parts.append(f"pattern={pat}")
    else:
        if details:
            parts.append(f"signals={json.dumps(details, sort_keys=True)}")
    return "; ".join(parts)


def enrich_legacy_with_audit(legacy: dict[str, Any]) -> dict[str, Any]:
    """Attach explainability into details without changing scores or decisions."""
    out = dict(legacy)
    det = dict(out.get("details") or {})
    det.setdefault("explainability", _audit_explainability(out))
    out["details"] = det
    return out


def legacy_to_decision_output(legacy: dict[str, Any]) -> AgentDecisionOutput:
    """Map existing analyze()/AgentResult dict into structured audit output."""
    approved = bool(legacy.get("decision"))
    det = dict(legacy.get("details") or {})
    expl = str(det.pop("explainability", "") or "")
    return AgentDecisionOutput(
        agent=str(legacy.get("agent_name", "")),
        decision="APPROVED" if approved else "REJECTED",
        score=float(legacy.get("score", 0.0)),
        reason=str(legacy.get("reasoning", "")),
        explainability=expl,
        details=det,
    )


def decision_output_to_legacy_for_api(out: AgentDecisionOutput) -> dict[str, Any]:
    """Produce AgentResult-compatible dict (decision bool, reasoning, details)."""
    det = dict(out.details)
    det["explainability"] = out.explainability or _audit_explainability(
        {
            "agent_name": out.agent,
            "details": out.details,
            "decision": out.decision == "APPROVED",
        }
    )
    return {
        "agent_name": out.agent,
        "decision": out.decision == "APPROVED",
        "score": out.score,
        "reasoning": out.reason,
        "details": det,
    }


def sort_agent_dicts(agent_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable ordering for API contract (EXPECTED_AGENT_NAMES)."""
    by_name = {d["agent_name"]: d for d in agent_dicts}
    return [by_name[name] for name in _AGENT_ORDER if name in by_name]


def build_final_consensus_payload(
    _claim_data: dict[str, Any],
    agent_results: list[AgentResult],
    *,
    final_decision: str,
    weighted_score: float,
    veto_applied: bool,
    consensus_threshold: float,
) -> FinalConsensusPayload:
    outputs = [legacy_to_decision_output(r.model_dump()) for r in agent_results]
    n = len(agent_results)
    avg_score = round(sum(r.score for r in agent_results) / n, 2) if n else 0.0
    return FinalConsensusPayload(
        final_decision=final_decision,  # type: ignore[arg-type]
        avg_score=avg_score,
        weighted_score=weighted_score,
        consensus_threshold=consensus_threshold,
        agents=outputs,
        veto_applied=veto_applied,
    )


def log_agent_decisions(agent_dicts: list[dict[str, Any]]) -> None:
    for d in sort_agent_dicts(agent_dicts):
        logger.info(
            "agent_decision name=%s approved=%s score=%s",
            d.get("agent_name"),
            d.get("decision"),
            d.get("score"),
        )
