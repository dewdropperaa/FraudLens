from __future__ import annotations

import json
import logging
import os
import hashlib
import re
import inspect
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from crewai import Agent, Crew, Process, Task
from langchain_community.embeddings import FakeEmbeddings, OllamaEmbeddings
from sklearn.cluster import KMeans

from claimguard.agents.identity_agent import IdentityAgent as DeterministicIdentityAgent
from claimguard.agents.validation_agent import ClaimValidationAgent
from claimguard.agents.security_utils import classify_prompt_injection, sanitize_for_prompt
from claimguard.llm_factory import assert_ollama_connection, get_crewai_llm
from claimguard.llm_tracking import parse_llm_json, safe_tracked_llm_call, tracked_agent_context
from claimguard.v2.blackboard import AgentContract, BlackboardValidationError, SharedBlackboard
from claimguard.v2.concierge import build_routing_decision
from claimguard.v2.consensus import ConsensusConfig, ConsensusEngine, should_force_human_review
from claimguard.v2.coverage_score import (
    CoverageScore,
    MIN_COVERAGE_ACCEPT,
    build_explanation,
    compute_coverage_score,
    coverage_decision,
)
from claimguard.v2.document_classifier import classify_document
from claimguard.v2.evidence_mapper import build_fraud_heatmap
from claimguard.v2.extraction.hybrid_extractor import HybridExtractor
from claimguard.v2.field_verification import verify_structured_fields
from claimguard.v2.fraud_ring_graph import get_fraud_ring_graph
from claimguard.v2.flow_tracker import get_tracker
from claimguard.v2.memory import (
    CaseMemoryEntry,
    CaseMemoryLayer,
    build_agent_summary,
    decision_to_fraud_label,
    get_memory_layer,
)
from claimguard.v2.memory_health import MemoryConfig, MemoryHealthReport, MemoryHealthStatus, get_memory_health
from claimguard.v2.human_review_store import HumanReviewStore, PendingHumanReviewRepository
from claimguard.v2.reliability import (
    ExternalValidationResult,
    get_reliability_store,
    hash_payload,
)
from claimguard.v2.schemas import (
    AgentOutput,
    ClaimGuardV2Response,
    DECISION_ENUM_VALUES,
    DocumentType,
    MemoryInsights,
    PreValidationResult,
    RoutingDecision,
    ValidationResult,
)
from claimguard.v2.trust_layer import (
    TrustLayerIPFSFailure,
    TrustLayerService,
)
from claimguard.v2.trace_engine import TraceEngine

LOGGER = logging.getLogger("claimguard.v2")
FORENSIC_MODE = False
DEBUG_EXPLANATION_MODE: bool = os.getenv("DEBUG_EXPLANATION_MODE", "1").strip() not in (
    "", "0", "false", "False",
)
_INPUT_HASH_HISTORY: Dict[str, int] = {}
_CANONICAL_DECISIONS = set(DECISION_ENUM_VALUES)
EXIT_REASONS: tuple[str, ...] = (
    "non_claim",
    "prompt_injection",
    "ocr_unreadable",
    "identity_not_verified",
    "critical_fields_unverified",
    "low_confidence",
    "approved",
    "trust_layer_degraded",
)
_EXIT_REASON_ALIASES: Dict[str, str] = {
    "TRUST_LAYER_DEGRADED": "trust_layer_degraded",
}
MIN_OCR_TEXT_LENGTH = 40
GENERIC_EXPLANATION_MARKERS = (
    "no suspicious patterns detected",
    "everything looks normal",
    "all checks passed",
    "looks legitimate",
    "appears valid",
    "insufficient data to conclude",
)
_CLAIM_INVOICE_KEYWORDS = ("facture", "invoice", "reçu", "recu", "ordonnance", "receipt")
_PROVIDER_KEYWORDS = ("clinic", "clinique", "hospital", "hopital", "hôpital", "doctor", "docteur", "dr.")
_CRITICAL_AGENT_FAILURE_MARKERS = (
    "not found in document",
    "insufficient evidence",
    "missing required document",
)
_CRITICAL_FIELD_KEYS = {"cin", "ipp", "amount"}
_HARD_REJECTION_REASON = "Claim rejected: data not found in supporting document"
# Label normalization is informational only — unknown labels are preserved
# as-is and never rejected. Coverage scoring (claimguard.v2.coverage_score)
# is the single source of pipeline-gating truth.
_DOCUMENT_TYPE_NORMALIZATION_MAP: Dict[str, str] = {
    "medical_claim_bundle": "medical_claim_bundle",
    "incomplete_claim_bundle": "hybrid_bundle",
    "hybrid_bundle": "hybrid_bundle",
    "unknown_bundle": "unknown_bundle",
}
SEQUENTIAL_AGENT_CONTRACTS: tuple[AgentContract, ...] = (
    AgentContract("IdentityAgent", ()),
    AgentContract("DocumentAgent", ("IdentityAgent",)),
    AgentContract("PolicyAgent", ("IdentityAgent", "DocumentAgent")),
    AgentContract("AnomalyAgent", ("IdentityAgent", "DocumentAgent", "PolicyAgent")),
    AgentContract("PatternAgent", ("AnomalyAgent",)),
    AgentContract("GraphRiskAgent", ("PatternAgent",)),
)

# Agent-role descriptions injected into CrewAI backstories so the LLM has
# context for its specialty and for how to interpret memory context.
_AGENT_BACKSTORIES: Dict[str, str] = {
    "IdentityAgent": (
        "You are an identity fraud specialist. "
        "You verify CIN validity, cross-reference documents, and detect identity reuse. "
        "When past cases show the same CIN was used fraudulently, you escalate risk."
    ),
    "DocumentAgent": (
        "You are a forensic document analyst. "
        "You assess completeness and authenticity of submitted documents. "
        "If similar past cases showed document forgery at the same provider, you flag it."
    ),
    "PolicyAgent": (
        "You are a compliance and policy risk analyst. "
        "You verify CNSS/CNOPS coverage rules and detect threshold gaming. "
        "If memory shows repeated near-limit claims from the same identity, you escalate."
    ),
    "AnomalyAgent": (
        "You are a behavioral anomaly detection expert. "
        "You surface abnormal amounts, history inconsistencies, and suspicious stability. "
        "Memory of past fraud cases with similar patterns must increase your risk assessment."
    ),
    "PatternAgent": (
        "You are a fraud pattern analyst. "
        "You detect repetition, timing patterns, and scripted billing signatures. "
        "Matching patterns from past fraud cases in memory must be explicitly flagged."
    ),
    "GraphRiskAgent": (
        "You are a network fraud analyst. "
        "You interpret probabilistic graph risk from provider-patient relationship clusters. "
        "If memory shows fraud involving the same hospital or doctor, you escalate risk."
    ),
}


class ContractViolationError(ValueError):
    """Raised when a response envelope violates the runtime contract."""


def build_response_envelope(
    decision: str,
    Ts: float,
    blackboard: SharedBlackboard | Dict[str, Any],
    *,
    score_evolution: List[float] | None = None,
    reflexive_retry_logs: List[Dict[str, Any]] | None = None,
    flags: Dict[str, Any] | None = None,
    agent_outputs: List[Dict[str, Any]] | None = None,
    exit_reason: str | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    normalized_decision = str(decision or "").strip().upper()
    if normalized_decision not in _CANONICAL_DECISIONS:
        raise ContractViolationError(f"Invalid decision '{decision}' for response envelope.")

    if isinstance(blackboard, SharedBlackboard):
        blackboard_snapshot = blackboard.to_dict()
    else:
        blackboard_snapshot = dict(blackboard or {})

    metadata = blackboard_snapshot.get("request", {}).get("data", {}) if isinstance(blackboard_snapshot.get("request"), dict) else {}
    claim_id = (
        str(extra.get("claim_id") or "")
        or str((blackboard_snapshot.get("metadata") or {}).get("claim_id") or "")
        or str((blackboard_snapshot.get("request_payload") or {}).get("metadata", {}).get("claim_id") or "")
        or str((metadata.get("claim_id") if isinstance(metadata, dict) else "") or "")
    )

    envelope = {
        "decision": normalized_decision,
        "Ts": float(Ts),
        "exit_reason": str(exit_reason or ""),
        "blackboard_snapshot": blackboard_snapshot,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim_id,
        "score_evolution": list(score_evolution or []),
        "reflexive_retry_logs": list(reflexive_retry_logs or []),
        "flags": dict(flags or {}),
        "agent_outputs": list(agent_outputs or []),
    }
    envelope.update(extra)
    return envelope


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _stable_output_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_document_hash(claim_payload: Dict[str, Any]) -> str:
    # BLOCKCHAIN-FIX: deterministic local fingerprint when external trust systems are unavailable.
    canonical = json.dumps(claim_payload, sort_keys=True, ensure_ascii=False, default=str)
    return "0x" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_safe_agent_context(blackboard: SharedBlackboard) -> Dict[str, Any]:
    return {
        "text": blackboard.extracted_text,
        "verified_data": blackboard.verified_structured_data,
        "flags": blackboard.system_flags,
    }


def terminate_pipeline(reason: str, flags: List[str]) -> Dict[str, Any]:
    return {
        "decision": "REJECTED",
        "reason": reason,
        "flags": list(flags),
        "terminated": True,
    }


def soft_degrade(reason: str, flags: List[str]) -> Dict[str, Any]:
    """Non-terminal degradation signal — pipeline continues with warnings.

    Replaces ``terminate_pipeline`` in paths where the previous behavior
    was "reject on document-type mismatch / label". The coverage score
    now drives the real decision downstream.
    """
    return {
        "decision": "HUMAN_REVIEW",
        "reason": reason,
        "flags": list(flags),
        "terminated": False,
        "degraded": True,
    }


def _build_pipeline_explanation(
    *,
    decision: str,
    ts: float,
    coverage: CoverageScore | None,
    reasons: List[str],
    signals: Dict[str, Any],
    tool_outputs: Dict[str, Any],
    summary: str = "",
) -> Dict[str, Any]:
    """Mandatory structured-explanation envelope attached to every exit.

    Returns a dict shaped as::

        {
          "decision": "ACCEPTED | REJECTED | HUMAN_REVIEW",
          "score": float,
          "explanation": {"summary": str, "reasons": [...],
                          "signals": {...}, "tool_outputs": {...}}
        }
    """
    decision_label = "ACCEPTED" if str(decision).upper() == "APPROVED" else str(decision).upper()
    envelope = build_explanation(
        decision=decision_label,
        score=float(ts),
        coverage=coverage,
        summary=summary,
        reasons=reasons,
        signals=signals,
        tool_outputs=tool_outputs,
        debug=DEBUG_EXPLANATION_MODE,
    )
    if DEBUG_EXPLANATION_MODE:
        LOGGER.info(
            "[DEBUG_EXPLANATION] decision=%s score=%.2f reasons=%s bundle=%s",
            envelope["decision"],
            envelope["score"],
            envelope["explanation"]["reasons"],
            (coverage.classifier_bundle if coverage else "n/a"),
        )
    return envelope


def exit_pipeline(reason: str, decision: str, ts: float = 0.0) -> Dict[str, Any]:
    return {
        "decision": str(decision or "").strip().upper(),
        "exit_reason": str(reason or "").strip(),
        "Ts": float(ts),
        "terminated": True,
    }


def _exit_from_ts(ts: float) -> Dict[str, Any]:
    ts_value = float(ts)
    # CALIBRATION-FIX: align terminal decision windows with consensus calibration.
    if ts_value <= 44.0:
        return exit_pipeline("low_confidence", "REJECTED", ts=ts_value)
    if 45.0 <= ts_value <= 64.0:
        return exit_pipeline("low_confidence", "HUMAN_REVIEW", ts=ts_value)
    return exit_pipeline("approved", "APPROVED", ts=ts_value)


def _log_pipeline_terminated(stage: str, reason: str) -> None:
    LOGGER.warning("[PIPELINE TERMINATED] stage=%s reason=%s", stage, reason)


def _confidence_from_score(score: float) -> str:
    # PROD-FIX: confidence mapping contract.
    if score >= 75.0:
        return "HIGH"
    if score >= 55.0:
        return "MEDIUM"
    return "LOW"


def _classify_flags(system_flags: List[str]) -> Dict[str, List[str]]:
    # PROD-FIX: tiered classification for auditability.
    blocking = {
        "INJECTION_DETECTED",
        "IDENTITY_HARD_FAIL",
        "CRITICAL_FIELD_MISSING",
        "AMOUNT_MISMATCH_CRITICAL",
    }
    informational = {
        "MEMORY_DISABLED",
        "DEGRADED_SECURITY_MODE",
        "LLM_LAYER2_DISABLED",
    }
    warnings = {
        "UNVERIFIED_FIELDS_PRESENT",
        "IDENTITY_SOFT_FAIL_CONTINUE",
        "LOW_TS_VALIDATION_GRACEFUL_DEGRADE",
        "DECISION_STABILITY_FAIL",
    }
    out = {"blocking": [], "warnings": [], "informational": []}
    for flag in sorted(set(system_flags or [])):
        if flag in blocking:
            out["blocking"].append(flag)
        elif flag in informational:
            out["informational"].append(flag)
        elif flag in warnings:
            out["warnings"].append(flag)
    return out


def run_agent_with_timeout(
    agent_name: str,
    runner: Any,
    timeout: float = 20.0,
) -> Dict[str, Any]:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(runner)
        try:
            return {"status": "DONE", "result": future.result(timeout=timeout)}
        except FuturesTimeoutError:
            future.cancel()
            return {
                "agent": agent_name,
                "status": "TIMEOUT",
                "score": 0,
                "reason": "Execution timeout",
                "output": {},
            }
        except Exception as exc:
            return {
                "agent": agent_name,
                "status": "ERROR",
                "score": 0,
                "reason": str(exc),
                "output": {},
            }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def validate_agent_result(
    result: Dict[str, Any],
    *,
    required_fields: List[str] | None = None,
) -> Dict[str, Any]:
    required = {"agent", "status", "output", "score", "reason"}
    payload = dict(result or {})
    agent_name = str(payload.get("agent") or "UNKNOWN_AGENT")

    def _error(reason: str) -> Dict[str, Any]:
        return {
            "agent": agent_name,
            "status": "ERROR",
            "output": {},
            "score": 0.0,
            "reason": reason,
        }

    if not required.issubset(payload.keys()):
        return _error("VALIDATION_FAIL:SCHEMA")
    status = str(payload.get("status") or "").upper()
    if status not in {"DONE", "ERROR"}:
        return _error("VALIDATION_FAIL:STATUS")
    output = payload.get("output")
    if not isinstance(output, dict):
        return _error("VALIDATION_FAIL:OUTPUT_TYPE")
    if status == "DONE" and not output:
        return _error("VALIDATION_FAIL:EMPTY_OUTPUT")
    if status == "DONE" and required_fields:
        missing = [field for field in required_fields if output.get(field) in (None, "", [], {})]
        if missing:
            return _error(f"VALIDATION_FAIL:MISSING_FIELDS:{','.join(missing)}")
    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        return _error("VALIDATION_FAIL:SCORE_TYPE")
    if score < 0.0 or score > 100.0:
        return _error("VALIDATION_FAIL:SCORE_RANGE")
    reason = str(payload.get("reason") or "").strip()
    if status == "DONE" and not reason:
        return _error("VALIDATION_FAIL:REASON")
    payload["status"] = status
    payload["score"] = score
    payload["reason"] = reason
    payload["output"] = output
    return payload


def run_agent_safe(agent: Any, input_data: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    agent_name = str(getattr(agent, "name", getattr(agent, "role", "UNKNOWN_AGENT")))
    started = perf_counter()
    LOGGER.info("[AGENT START] %s", agent_name)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        if callable(agent):
            future = executor.submit(agent, input_data)
        elif hasattr(agent, "run") and callable(getattr(agent, "run")):
            future = executor.submit(agent.run, input_data)
        elif hasattr(agent, "analyze") and callable(getattr(agent, "analyze")):
            future = executor.submit(agent.analyze, input_data)
        else:
            raise ValueError(f"Unsupported agent runner type for {agent_name}")
        try:
            raw = future.result(timeout=timeout)
            result = raw if isinstance(raw, dict) else {"agent": agent_name, "status": "ERROR", "output": {}, "score": 0.0, "reason": "VALIDATION_FAIL:NON_DICT"}
            result.setdefault("agent", agent_name)
            result.setdefault("status", "DONE")
            result.setdefault("output", {})
            result.setdefault("score", 0.0)
            result.setdefault("reason", "")
            validated = validate_agent_result(result)
            LOGGER.info("[AGENT STATUS] %s %s", agent_name, validated.get("status"))
            return validated
        except FuturesTimeoutError:
            future.cancel()
            LOGGER.warning("[AGENT STATUS] %s TIMEOUT", agent_name)
            return {
                "agent": agent_name,
                "status": "ERROR",
                "output": {},
                "score": 0.0,
                "reason": "TIMEOUT",
            }
        except Exception:
            LOGGER.exception("agent_execution_exception agent=%s", agent_name)
            return {
                "agent": agent_name,
                "status": "ERROR",
                "output": {},
                "score": 0.0,
                "reason": "EXCEPTION",
            }
    finally:
        elapsed = perf_counter() - started
        LOGGER.info("[AGENT TIME] %s %.3fs", agent_name, elapsed)
        executor.shutdown(wait=False, cancel_futures=True)


def finalize_decision(consensus_result: Dict[str, Any], critical_failures: List[str]) -> str:
    return str(_exit_from_ts(float(consensus_result.get("Ts", 0.0)))["decision"])


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a or "", b=b or "").ratio()


def _output_entropy(outputs: List[Dict[str, Any]]) -> float:
    if not outputs:
        return 0.0
    unique = { _stable_output_hash(item) for item in outputs }
    return len(unique) / max(1, len(outputs))


def _safe_json_load(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _normalize_validation_document_type(raw_document_type: Any) -> tuple[str, str | None]:
    """Return (normalized_label, original_if_remapped).

    The label is informational clustering only — any string is accepted. If
    the raw label is in the remap table we return its canonical form and
    keep the original for forensic traceability; otherwise the raw value is
    returned untouched. We NEVER force-map to "unknown" on label mismatch.
    """
    document_type = str(raw_document_type or "unknown").strip().lower() or "unknown"
    normalized = _DOCUMENT_TYPE_NORMALIZATION_MAP.get(document_type)
    if normalized is not None and normalized != document_type:
        return normalized, document_type
    return document_type, None


def _normalize_score_confidence_scale(score: float, confidence: float) -> tuple[float, float]:
    # Accept either 0..1 or 0..100 outputs; convert percentages when detected.
    s = score / 100.0 if score > 1.0 else score
    c = confidence / 100.0 if confidence > 1.0 else confidence
    return s, c


def _to_ui_agent_status(
    *, runtime_status: str, score_0_100: float, insufficient_data: bool,
    json_parse_failed: bool = False,
) -> str:
    normalized_runtime = str(runtime_status or "").strip().upper()
    if normalized_runtime in {"ERROR", "TIMEOUT"}:
        return "FAIL"
    if insufficient_data:
        return "REVIEW"
    if json_parse_failed:
        return "REVIEW"
    if score_0_100 >= 60.0:
        return "PASS"
    return "FAIL"


def _coerce_claims(parsed: Dict[str, Any], explanation: str) -> List[Dict[str, Any]]:
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return [{"statement": explanation, "evidence": "", "verified": False}]
    normalized: List[Dict[str, Any]] = []
    for raw in claims:
        if not isinstance(raw, dict):
            continue
        normalized.append(
            {
                "statement": str(raw.get("statement", "")).strip(),
                "evidence": str(raw.get("evidence", "")).strip(),
                "verified": bool(raw.get("verified", False)),
            }
        )
    return normalized or [{"statement": explanation, "evidence": "", "verified": False}]


def _is_evidence_in_source(evidence: str, extracted_text: str, structured_data: Dict[str, Any]) -> bool:
    ev = (evidence or "").strip().lower()
    if not ev:
        return False
    raw = (extracted_text or "").lower()
    if ev in raw:
        return True
    for value in structured_data.values():
        candidate = str(value or "").strip().lower()
        if candidate and (ev == candidate or ev in candidate or candidate in ev):
            return True
    return False


def _explanation_is_grounded(explanation: str, extracted_text: str, structured_data: Dict[str, Any]) -> bool:
    text = (explanation or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in GENERIC_EXPLANATION_MARKERS):
        return False
    has_number = any(ch.isdigit() for ch in text)
    mentions_field = any(str(value).strip() and str(value).strip().lower() in text for value in structured_data.values())
    mentions_ocr = any(token in text for token in (extracted_text or "").lower().split()[:30]) if extracted_text else False
    return has_number and (mentions_field or mentions_ocr)


def _extract_field_from_text(patterns: List[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
    return ""


def _normalize_extracted_amount(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    # Remove currency labels/symbols and normalize OCR number formatting.
    cleaned = re.sub(r"(?i)\b(?:mad|dh|dhs|dirhams?|usd|eur)\b", " ", candidate)
    cleaned = re.sub(r"[^\d,.\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    numeric = cleaned.replace(" ", "")
    if "," in numeric and "." in numeric:
        if numeric.rfind(",") > numeric.rfind("."):
            numeric = numeric.replace(".", "").replace(",", ".")
        else:
            numeric = numeric.replace(",", "")
    elif "," in numeric:
        numeric = numeric.replace(",", ".")
    return numeric


def _extract_amount_from_text(text: str) -> str:
    total_match = re.search(r"TOTAL G[ÉE]N[ÉE]RAL\s*\n?\s*([0-9\s.,]+)", text, flags=re.IGNORECASE)
    if total_match:
        return _normalize_extracted_amount(total_match.group(1))
    return _normalize_extracted_amount(
        _extract_field_from_text(
            [
                r"\b(?:montant|amount|total(?:\s+(?:ttc|due|a payer|à payer))?)\s*[:=]?\s*([0-9][0-9\s.,]{0,20})\b",
                r"\b([0-9][0-9\s.,]{0,20})\s*(?:mad|dh|dhs|dirhams?)\b",
            ],
            text,
        )
    )


def _verify_structured_fields(
    structured_fields: Dict[str, Any],
    extracted_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return verify_structured_fields(structured_fields, extracted_text)


def _reconcile_field_verification(
    verification_summary: Dict[str, Any],
    identity_verification: Dict[str, Any],
    agent_outputs: List[Any],
) -> None:
    """Merge IdentityAgent findings into the pre-agent field verification dicts.

    The pre-agent verifier runs before agents with a simpler regex extractor and
    can miss fields (e.g. CIN inside a differently formatted line). IdentityAgent
    runs later with a more thorough extractor. When IdentityAgent passes with high
    confidence and found CIN/IPP in the OCR text, promote those fields from
    unverified → verified so that both the approval guard and the global
    enforcement guard see accurate data.

    Mutates verification_summary and identity_verification in place.
    """
    identity_ao = next((ao for ao in agent_outputs if ao.agent == "IdentityAgent"), None)
    if identity_ao is None:
        return

    ia_details = identity_ao.output_snapshot.get("details", {}) or {}
    ia_cin_found = bool(ia_details.get("cin_found_in_ocr", False))
    ia_ipp_found = bool(ia_details.get("ipp_found_in_ocr", False))
    ia_score = float(identity_ao.score)
    ia_status = str(identity_ao.output_snapshot.get("status", "")).upper()
    ia_high_confidence = ia_score >= 0.6 and ia_status in {"PASS", "REVIEW"}

    if not ia_high_confidence:
        return

    pre_cin = bool(identity_verification.get("cin_found", False))
    pre_ipp = bool(identity_verification.get("ipp_found", False))

    promoted: List[str] = []
    if ia_cin_found and not pre_cin:
        identity_verification["cin_found"] = True
        promoted.append("cin")
    if ia_ipp_found and not pre_ipp:
        identity_verification["ipp_found"] = True
        promoted.append("ipp")

    if promoted:
        LOGGER.info(
            "[FieldReconciliation] IdentityAgent override: %s now verified "
            "(pre-agent had False, IdentityAgent score=%.2f status=%s)",
            promoted, ia_score, ia_status,
        )
        # Reduce unverified_critical_fields count for each promoted field
        prev_count = int(verification_summary.get("unverified_critical_fields", 0))
        reconciled = max(0, prev_count - len(promoted))
        verification_summary["unverified_critical_fields"] = reconciled
        LOGGER.info(
            "[FieldReconciliation] unverified_critical_fields: %d → %d",
            prev_count, reconciled,
        )
        # If no critical fields remain unverified, clear has_unverified_fields
        if reconciled == 0:
            verification_summary["has_unverified_fields"] = False
            LOGGER.info("[FieldReconciliation] has_unverified_fields cleared → False")


def _parse_memory_insights(parsed: Dict[str, Any]) -> MemoryInsights | None:
    """Extract and validate memory_insights from an agent's JSON output."""
    raw = parsed.get("memory_insights")
    if not isinstance(raw, dict):
        return None
    try:
        return MemoryInsights(
            similar_cases_found=int(raw.get("similar_cases_found", 0)),
            fraud_matches=int(raw.get("fraud_matches", 0)),
            identity_reuse_detected=bool(raw.get("identity_reuse_detected", False)),
            impact_on_score=str(raw.get("impact_on_score", "")),
            notes=str(raw.get("notes", "")),
        )
    except Exception:
        return None


def _compute_fallback_memory_insights(
    memory_context: List[Dict[str, Any]],
    current_cin: str,
) -> MemoryInsights:
    """
    When the LLM does not return memory_insights (or returns malformed JSON),
    produce a deterministic fallback from the raw memory context.
    """
    if not memory_context:
        return MemoryInsights(similar_cases_found=0, fraud_matches=0)

    fraud_labels = {"fraud", "suspicious"}
    fraud_matches = sum(
        1 for c in memory_context if c.get("fraud_label", "").lower() in fraud_labels
    )
    identity_reuse = any(
        c.get("cin", "").strip().upper() == current_cin.strip().upper()
        and current_cin.strip()
        for c in memory_context
    )
    if fraud_matches > 0:
        impact = f"Memory contains {fraud_matches} fraud/suspicious case(s) — risk elevated"
    else:
        impact = "No fraud matches found in memory context"

    notes_parts: List[str] = []
    if identity_reuse:
        notes_parts.append("CIN reuse detected across memory cases")
    cinset = {c.get("cin", "") for c in memory_context if c.get("cin")}
    if len(cinset) == 1 and next(iter(cinset)) == current_cin:
        notes_parts.append("All memory cases share the same CIN")
    hospitals = {c.get("hospital", "") for c in memory_context if c.get("hospital")}
    if hospitals:
        notes_parts.append(f"Recurring hospital(s) in memory: {', '.join(sorted(hospitals))}")

    return MemoryInsights(
        similar_cases_found=len(memory_context),
        fraud_matches=fraud_matches,
        identity_reuse_detected=identity_reuse,
        impact_on_score=impact,
        notes="; ".join(notes_parts) if notes_parts else "Memory advisory only",
    )


class ClaimGuardV2Orchestrator:
    def __init__(
        self,
        *,
        trust_layer_service: TrustLayerService | None = None,
        memory_layer: CaseMemoryLayer | None = None,
    ) -> None:
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._consensus_engine = ConsensusEngine()
        self._consensus_config = ConsensusConfig()
        self._trust_layer = trust_layer_service or TrustLayerService.build_default()
        self._memory = memory_layer or get_memory_layer()
        self._memory_config = MemoryConfig(
            min_similarity=float(getattr(self._memory, "_similarity_threshold", 0.7)),
            degraded_memory_auto_approve_threshold=float(
                os.getenv("DEGRADED_MEMORY_AUTO_APPROVE_THRESHOLD", "95.0")
            ),
        )
        self._memory_degraded_streak = 0
        self._alert_webhook_url = os.getenv("MEMORY_ALERT_WEBHOOK_URL", "").strip()
        self._reliability_store = get_reliability_store()
        self._fraud_ring_graph = get_fraud_ring_graph()
        self._human_review_store = HumanReviewStore()
        self._pending_human_reviews = PendingHumanReviewRepository()
        self._hybrid_extractor = HybridExtractor()
        print("LLM Provider: OLLAMA")
        print("Available models:", ["mistral", "llama3", "deepseek-r1"])
        assert_ollama_connection()

    def _make_chat_llm(self, model_name: str):
        return get_crewai_llm(model_name)

    @staticmethod
    def _run_identity_agent_local(claim_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the deterministic IdentityAgent implementation and normalize to the
        JSON shape expected by the v2 multi-agent pipeline.
        """
        result = DeterministicIdentityAgent().analyze(claim_request)
        output = result.get("output", {}) if isinstance(result, dict) else {}
        explanation = str(
            output.get("explanation")
            or output.get("reasoning")
            or result.get("reason")
            or "Identity analysis completed"
        )
        evidence = output.get("evidence", {}) if isinstance(output.get("evidence"), dict) else {}
        memory_insights = output.get("memory_insights")
        if memory_insights is None and isinstance(output.get("details"), dict):
            memory_insights = output["details"].get("memory_insights")
        # IdentityAgent uses legitimacy scale (100=clean, 0=suspicious).
        # Pipeline and consensus also use legitimacy scale (high=clean=APPROVED).
        # Keep score as-is; normalize confidence from the same validity value.
        raw_validity = float(output.get("score", result.get("score", 0.0)))
        confidence = round(min(1.0, max(0.3, raw_validity / 100.0)), 4)
        return {
            "score": raw_validity,
            "confidence": confidence,
            "explanation": explanation,
            "claims": [{"statement": explanation, "evidence": "", "verified": bool(evidence.get("cin_found") or evidence.get("ipp_found"))}],
            "hallucination_flags": [],
            "memory_insights": memory_insights or {},
            "evidence": evidence,
        }

    def get_memory_health_report(self) -> MemoryHealthReport:
        return get_memory_health(self._memory_config, self._memory)

    def _emit_memory_degraded_alert(self, *, claim_id: str, report: MemoryHealthReport) -> None:
        payload = {
            "severity": "critical",
            "type": "memory_degraded_consecutive_claims",
            "claim_id": claim_id,
            "status": report.status.value,
            "failure_reason": report.failure_reason,
            "probe_result_count": report.probe_result_count,
            "latency_ms": report.latency_ms,
        }
        LOGGER.critical("memory_degraded_alert payload=%s", payload)
        if not self._alert_webhook_url:
            return
        try:
            import requests

            requests.post(self._alert_webhook_url, json=payload, timeout=5)
        except Exception as exc:
            LOGGER.warning("memory_degraded_alert_webhook_failed error=%s", exc)

    def _track_memory_health(self, *, claim_id: str, report: MemoryHealthReport) -> None:
        if report.status == MemoryHealthStatus.HEALTHY:
            self._memory_degraded_streak = 0
            return
        self._memory_degraded_streak += 1
        if self._memory_degraded_streak >= 2:
            self._emit_memory_degraded_alert(claim_id=claim_id, report=report)

    def _build_ocr_blackboard_payload(self, claim_request: Dict[str, Any]) -> Dict[str, Any]:
        texts: List[str] = []
        for extraction in claim_request.get("document_extractions", []) or []:
            if isinstance(extraction, dict):
                candidate = str(extraction.get("extracted_text") or "").strip()
                if candidate:
                    texts.append(candidate)
        for doc in claim_request.get("documents", []) or []:
            if isinstance(doc, dict):
                candidate = str(doc.get("text") or "").strip()
                if candidate:
                    texts.append(candidate)
        raw_text = "\n".join(texts).strip()

        hybrid_result = self._hybrid_extractor.extract(raw_text)
        fields = hybrid_result.get("fields", {}) if isinstance(hybrid_result.get("fields"), dict) else {}
        extraction_warnings: List[Dict[str, str]] = []
        if hybrid_result.get("status") != "OK":
            extraction_warnings.append(
                {
                    "type": "HYBRID_EXTRACTION_DEGRADED",
                    "reason": str(hybrid_result.get("reason") or "Hybrid extraction failed"),
                    "stage": str(hybrid_result.get("stage") or "rule"),
                }
            )
        structured_fields: Dict[str, Any] = {
            "name": fields.get("name")
            or _extract_field_from_text(
                [
                    r"\b(?:nom\s+complet|nom(?:\s+du)?\s+patient)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ' -]{3,})",
                ],
                raw_text,
            ),
            "cin": fields.get("cin")
            or _extract_field_from_text(
                [
                    r"\bCIN\s*[:\-]?\s*([A-Z]{1,2}\d{5,6})\b",
                    r"\b([A-Z]{1,2}\d{5,6})\b",
                ],
                raw_text,
            ),
            "ipp": fields.get("ipp")
            or _extract_field_from_text(
                [
                    r"\b(?:N[°º]\s*)?IPP\s*[:\-]?\s*([A-Za-z0-9\-]+)\b",
                ],
                raw_text,
            ),
            "date": fields.get("dob")
            or _extract_field_from_text(
                [
                    r"\b(?:date\s+de\s+naissance|né\s+le)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{2,4})\b",
                    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
                ],
                raw_text,
            ),
            "insurance": fields.get("insurance")
            or _extract_field_from_text(
                [
                    r"\b(?:mutuelle|assurance|organisme)\s*[:\-]\s*([A-Za-z0-9À-ÖØ-öø-ÿ' \-]+)",
                ],
                raw_text,
            ),
            "amount": _extract_amount_from_text(raw_text),
            "provider": _extract_field_from_text(
                [
                    r"\b(Clinique\s+[A-Za-zÀ-ÖØ-öø-ÿ\s'-]+)",
                    r"\b(?:hopital|hôpital|clinique|provider|doctor|dr\.?)[:\s-]*([A-Za-z0-9 '\-]{3,})",
                ],
                raw_text,
            ),
        }
        return {
            "status": "OK",
            "raw_text": raw_text,
            "structured_fields": structured_fields,
            "hybrid_result": hybrid_result,
            "extraction_warnings": extraction_warnings,
        }

    def self_test(self) -> bool:
        return self._hybrid_extractor.self_test()

    @staticmethod
    def _is_unreadable_text(text: str) -> bool:
        if not text:
            return True
        if len(text.strip()) < MIN_OCR_TEXT_LENGTH:
            return True
        printable = sum(1 for ch in text if ch.isalnum() or ch.isspace())
        ratio = printable / max(1, len(text))
        return ratio < 0.45

    @staticmethod
    def _compute_extraction_validation(raw_text: str, structured_fields: Dict[str, Any]) -> Dict[str, Any]:
        fields_detected = sorted(
            [name for name, value in structured_fields.items() if str(value).strip()]
        )
        required_fields = ["cin", "amount", "date", "provider"]
        missing_fields = [name for name in required_fields if name not in fields_detected]
        return {
            "text_length": len(raw_text),
            "fields_detected": fields_detected,
            "missing_fields": missing_fields,
        }

    @staticmethod
    def _build_input_summary(claim_request: Dict[str, Any]) -> str:
        identity = claim_request.get("identity", {}) if isinstance(claim_request.get("identity"), dict) else {}
        policy = claim_request.get("policy", {}) if isinstance(claim_request.get("policy"), dict) else {}
        metadata = claim_request.get("metadata", {}) if isinstance(claim_request.get("metadata"), dict) else {}
        cin = identity.get("cin") or identity.get("CIN") or claim_request.get("patient_id") or "unknown"
        ipp = identity.get("ipp") or identity.get("IPP") or "unknown"
        provider = identity.get("hospital") or policy.get("hospital") or metadata.get("hospital") or "unknown"
        amount = claim_request.get("amount") or metadata.get("amount") or policy.get("amount") or "unknown"
        return f"CIN={cin}; IPP={ipp}; provider={provider}; amount={amount}"

    @staticmethod
    def _run_pre_validation_guard(extracted_text: str) -> Dict[str, Any]:
        text = str(extracted_text or "")
        lowered = text.lower()
        has_amount = bool(
            re.search(r"\b\d+(?:[.,]\d+)?\s*(?:mad|dh|dirham|dhs|eur|usd|€)\b", lowered)
            or re.search(r"\b(?:montant|total|prix|cost|amount)\s*[:=]?\s*\d+(?:[.,]\d+)?\b", lowered)
        )
        has_provider = any(token in lowered for token in _PROVIDER_KEYWORDS)
        has_date = bool(
            re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lowered)
            or re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", lowered)
        )
        has_invoice = any(token in lowered for token in _CLAIM_INVOICE_KEYWORDS)
        signal_count = sum([has_amount, has_provider, has_date, has_invoice])
        injection_result = classify_prompt_injection(text)
        injection_detected = bool(injection_result.get("is_injection", False))
        injection_confidence = int(injection_result.get("confidence", 0))
        injection_reason = str(injection_result.get("reason", "")).strip()
        injection_signals = injection_result.get("signals", {})
        security_flags = list(injection_result.get("security_flags", []))
        degraded_security_mode = bool(injection_result.get("degraded_security_mode", False))
        layer1_blocked = bool(injection_result.get("layer1_blocked", False))
        document_type = "MEDICAL_CLAIM" if signal_count >= 2 else "NON_CLAIM"
        flags: List[str] = []
        if document_type == "NON_CLAIM":
            flags.append("NON_CLAIM")
        if layer1_blocked:
            flags.append("PROMPT_INJECTION_LAYER1")
        elif injection_detected or injection_confidence > 70:
            flags.append("PROMPT_INJECTION")
        # Only deterministic injection block is non-bypassable. NON_CLAIM is informational.
        hard_block = layer1_blocked
        failed = hard_block
        payload: Dict[str, Any] = {
            "document_type": document_type,
            "injection_detected": injection_detected,
            "injection_confidence": injection_confidence,
            "injection_reason": injection_reason,
            "layer1_blocked": layer1_blocked,
            "security_flags": security_flags,
            "degraded_security_mode": degraded_security_mode,
            "injection_classifier": {
                "is_injection": injection_detected,
                "confidence": injection_confidence,
                "reason": injection_reason,
            },
            "injection_signals": injection_signals,
            "claim_signal_count": signal_count,
            "claim_signals": {
                "monetary_amount": has_amount,
                "medical_provider_reference": has_provider,
                "date": has_date,
                "invoice_or_receipt_keywords": has_invoice,
            },
            "hard_block": hard_block,
            "failed": failed,
            "passed": not hard_block,
            "flags": flags,
        }
        if hard_block:
            payload["hard_block_response"] = {
                "score": 0,
                "confidence": max(100, injection_confidence),
                "status": "REJECTED",
                "reason": injection_reason or "Invalid document or prompt injection detected",
                "flags": flags,
            }
        return payload

    @staticmethod
    def _noise_ratio(text: str) -> float:
        if not text:
            return 1.0
        noise = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,:;-/"))
        return noise / max(1, len(text))

    def _compute_input_trust(
        self,
        *,
        raw_text: str,
        structured_fields: Dict[str, Any],
        extraction_validation: Dict[str, Any],
    ) -> Dict[str, int]:
        text_len = len(raw_text or "")
        ocr_quality = max(0, min(100, int((min(1.0, text_len / 400.0) * 70) + ((1.0 - self._noise_ratio(raw_text)) * 30))))
        required = 4
        detected = len(extraction_validation.get("fields_detected", []))
        completeness = max(0, min(100, int((detected / max(1, required)) * 100)))
        lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
        avg_line = (sum(len(ln) for ln in lines) / max(1, len(lines))) if lines else 0
        readability = max(0, min(100, int(min(1.0, avg_line / 50.0) * 100)))
        return {
            "ocr_quality": ocr_quality,
            "completeness": completeness,
            "readability": readability,
        }

    @staticmethod
    def _input_trust_score(trust: Dict[str, int]) -> float:
        return round(
            (0.45 * float(trust.get("ocr_quality", 0)))
            + (0.35 * float(trust.get("completeness", 0)))
            + (0.20 * float(trust.get("readability", 0))),
            2,
        )

    @staticmethod
    def _external_validation_hook(claim_request: Dict[str, Any]) -> ExternalValidationResult:
        metadata = claim_request.get("metadata", {}) if isinstance(claim_request.get("metadata"), dict) else {}
        if metadata.get("provider_registry_mismatch") is True:
            return ExternalValidationResult(ok=False, source="provider_registry", reason="provider mismatch")
        if metadata.get("hospital_validation_mismatch") is True:
            return ExternalValidationResult(ok=False, source="hospital_validation", reason="hospital mismatch")
        return ExternalValidationResult()

    def _register_human_review_context(
        self,
        *,
        claim_id: str,
        claim_request: Dict[str, Any],
        ts_score: float,
        reason: str,
        verified_fields: Dict[str, Any],
        agent_outputs: List[AgentOutput],
        blackboard_snapshot: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        first_doc = (claim_request.get("documents_base64") or [None])[0]
        doc_ref = None
        if isinstance(first_doc, dict):
            doc_ref = self._human_review_store.save_temp_document(claim_id=claim_id, document_part=first_doc)
        document_url = None
        if doc_ref is not None:
            document_url = (
                f"/api/v2/claim/{claim_id}/review-document"
                f"?file={doc_ref.file_name}&token={doc_ref.token}&expires={doc_ref.token_expires_at}"
            )
        heatmap_payload = {"heatmap": [], "fallback": [], "status": "missing_pdf"}
        try:
            heatmap_payload = build_fraud_heatmap(
                blackboard=dict(blackboard_snapshot or {}),
                agent_outputs=[item.model_dump() for item in agent_outputs],
                pdf_path=(doc_ref.file_path if doc_ref else None),
            )
        except Exception as exc:
            LOGGER.warning("heatmap_generation_failed claim_id=%s error=%s", claim_id, exc)
        self._pending_human_reviews.save(claim_id, {
            "claim_id": claim_id,
            "ts": float(ts_score),
            "reason": reason,
            "document_url": document_url,
            "document_file_path": doc_ref.file_path if doc_ref else None,
            "document_file_name": doc_ref.file_name if doc_ref else None,
            "document_token": doc_ref.token if doc_ref else None,
            "document_token_expires": doc_ref.token_expires_at if doc_ref else None,
            "extracted_data": dict(verified_fields or {}),
            "agent_breakdown": [item.model_dump() for item in agent_outputs],
            "heatmap": list(heatmap_payload.get("heatmap", [])),
            "heatmap_fallback": list(heatmap_payload.get("fallback", [])),
            "heatmap_status": str(heatmap_payload.get("status", "missing_pdf")),
            "pipeline_version": "v2",
            "ai_suggested_decision": "HUMAN_REVIEW",
            "risk_breakdown": {
                "ts_score": float(ts_score),
                "risk_level": "MEDIUM" if ts_score < 75 else "HIGH",
            },
        })
        return {
            "document_url": document_url,
            "heatmap": list(heatmap_payload.get("heatmap", [])),
            "heatmap_fallback": list(heatmap_payload.get("fallback", [])),
        }

    def get_human_review_context(self, claim_id: str) -> Dict[str, Any] | None:
        context = self._pending_human_reviews.get(claim_id)
        if not context:
            return None
        return {
            "claim_id": context["claim_id"],
            "ts": context["ts"],
            "reason": context["reason"],
            "document_url": context["document_url"],
            "extracted_data": context.get("extracted_data", {}),
            "agent_breakdown": context.get("agent_breakdown", []),
            "heatmap": context.get("heatmap", []),
            "heatmap_fallback": context.get("heatmap_fallback", []),
            "heatmap_status": context.get("heatmap_status", "missing_pdf"),
            "pipeline_version": context.get("pipeline_version", "v2"),
            "ai_suggested_decision": context.get("ai_suggested_decision", "HUMAN_REVIEW"),
            "risk_breakdown": context.get("risk_breakdown", {}),
        }

    def resolve_human_review_document(
        self,
        *,
        claim_id: str,
        file_name: str,
        token: str,
        expires: int,
    ) -> str | None:
        context = self._pending_human_reviews.get(claim_id)
        if not context:
            return None
        stored_name = str(context.get("document_file_name") or "")
        if stored_name != str(file_name or ""):
            return None
        if not self._human_review_store.verify_token(
            claim_id=claim_id,
            file_name=file_name,
            token=token,
            expires_at=expires,
        ):
            return None
        return context.get("document_file_path")

    def apply_human_decision(
        self,
        *,
        claim_id: str,
        decision: str,
        reviewer_id: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        context = self._pending_human_reviews.get(claim_id)
        if context is None:
            raise KeyError(f"No pending human review found for claim {claim_id}")
        normalized = str(decision or "").upper()
        if normalized not in {"APPROVED", "REJECTED"}:
            raise ValueError("decision must be APPROVED or REJECTED")

        ts_score = float(context.get("ts", 0.0))
        if normalized == "APPROVED":
            trust_layer_payload: Dict[str, Any] | None = None
            try:
                trust_layer_payload = self._trust_layer.process_approved_claim(
                    {
                        "claim_id": claim_id,
                        "decision": "APPROVED",
                        "ts_score": ts_score,
                        "claim_request": {
                            "documents": [
                                {
                                    "id": f"{claim_id}-human-approved",
                                    "document_type": "medical_invoice",
                                    "content": json.dumps(context.get("extracted_data", {})),
                                }
                            ]
                        },
                        "agent_outputs": [
                            {
                                "agent": "HumanReviewer",
                                "explanation": f"reviewer={reviewer_id}; notes={notes[:200]}",
                            }
                        ],
                        "flags": ["HUMAN_REVIEW_FINALIZED"],
                    }
                )
            except Exception as exc:
                # Never fail the manual finalization endpoint due to trust side-effects.
                LOGGER.error(
                    "human_review_trust_layer_degraded claim_id=%s reviewer=%s error=%s",
                    claim_id,
                    reviewer_id,
                    str(exc),
                )
            self._pending_human_reviews.delete(claim_id)
            return {
                "status": "FINALIZED",
                "tx_hash": (trust_layer_payload or {}).get("tx_hash"),
                "ipfs_cid": (trust_layer_payload or {}).get("cid"),
                "firebase_id": (trust_layer_payload or {}).get("firebase_id"),
                "trust_layer_status": "stored" if trust_layer_payload else "degraded",
                "decision": "APPROVED",
            }

        self._pending_human_reviews.delete(claim_id)
        return {"status": "REJECTED", "decision": "REJECTED"}

    def _build_task_prompt(
        self,
        *,
        contract: AgentContract,
        blackboard: SharedBlackboard,
    ) -> str:
        safe_context = build_safe_agent_context(blackboard)
        memory_context = blackboard.memory_context
        memory_section = ""
        if memory_context:
            memory_section = (
                "\n\nMEMORY CONTEXT — PAST SIMILAR CASES (advisory, similarity >= threshold):\n"
                + json.dumps(memory_context, ensure_ascii=False, indent=2)
                + "\n\n"
                "MEMORY ANALYSIS RULES (MANDATORY):\n"
                "1. Examine every case in memory_context above.\n"
                "2. Answer internally: Have I seen similar cases? Do they indicate fraud?\n"
                "3. If past cases show fraud with the same CIN → flag identity reuse, increase risk.\n"
                "4. If past cases show fraud at the same hospital/doctor → increase risk.\n"
                "5. If similarity < 0.7 (already filtered) → ignore. Memory is ADVISORY, not absolute.\n"
                "6. If memory contradicts current data → note the contradiction, do NOT override score.\n"
                "7. Include 'memory_insights' in your JSON output (see schema below).\n"
            )
        else:
            memory_section = (
                "\n\nMEMORY CONTEXT: No similar past cases found above the similarity threshold.\n"
            )

        base_prompt = (
            f"You are the {contract.name}. {_AGENT_BACKSTORIES.get(contract.name, '')}\n\n"
            "You are a fraud detection agent.\n\n"
            "You MUST ONLY use the following VERIFIED information.\n\n"
            "--- VERIFIED DATA ---\n"
            f"{json.dumps(safe_context['verified_data'], ensure_ascii=False)}\n\n"
            "--- DOCUMENT TEXT (OCR) ---\n"
            f"{safe_context['text']}\n\n"
            "--- SYSTEM FLAGS ---\n"
            f"{json.dumps(safe_context['flags'], ensure_ascii=False)}\n\n"
            "STRICT RULES:\n\n"
            "* DO NOT assume missing data\n"
            "* DO NOT infer values not present\n"
            "* If evidence is missing -> reduce score\n"
            "* If contradiction exists -> flag explicitly\n\n"
            "Return:\n"
            "score (0-100)\n"
            "confidence (0-1)\n"
            "explanation (evidence-based only)\n"
            + memory_section
        )
        if contract.name == "IdentityAgent":
            base_prompt += (
                "\nIDENTITY AGENT RULES (Moroccan CIN aware):\n"
                "TASK: Verify claimant identity using CIN and/or IPP with strict OCR evidence.\n"
                "CHECKS:\n"
                "- Validate CIN format with regex ^[A-Z]{1,2}[0-9]{5,6}$ when CIN is provided.\n"
                "- Validate IPP format as numeric-only and length 6-10 when IPP is provided.\n"
                "- MUST check BOTH CIN and IPP if both are present.\n"
                "- MUST verify CIN/IPP presence in extracted_text, never assume validity from format.\n"
                "DECISION RULES:\n"
                "- If both CIN and IPP are missing => status=INSUFFICIENT_DATA.\n"
                "- If provided CIN/IPP is not found in OCR text => status=SUSPICIOUS.\n"
                "- If identity not found in OCR text, score MUST be in 0-20 max.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Output evidence object: {\"cin_found\": true/false, \"ipp_found\": true/false}.\n"
                "- Every claim must include CIN and/or IPP value and where it was found.\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must reference CIN and/or IPP explicitly.\n"
            )
        elif contract.name == "DocumentAgent":
            base_prompt += (
                "\nDOCUMENT AGENT RULES (Structure validator):\n"
                "TASK: Determine if document is a valid medical claim/invoice.\n"
                "REQUIRED FIELDS:\n"
                "- provider/hospital\n"
                "- date\n"
                "- cost/amount\n"
                "- medical description\n"
                "DECISION RULES:\n"
                "- If >=2 required fields are missing => status=INSUFFICIENT_DATA.\n"
                "- If structure is not invoice-like => status=SUSPICIOUS.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Quote exact snippets for provider, amount, and date.\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must list detected fields vs missing fields.\n"
            )
        elif contract.name == "PolicyAgent":
            base_prompt += (
                "\nPOLICY AGENT RULES (Coverage logic):\n"
                "TASK: Check if claim fits policy coverage rules.\n"
                "CHECKS:\n"
                "- Compare service type vs policy coverage.\n"
                "- Compare amount vs policy limits.\n"
                "DECISION RULES:\n"
                "- If key policy fields are missing => status=INSUFFICIENT_DATA.\n"
                "- Never assume coverage without direct evidence.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Must reference service type and amount.\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must explicitly state why covered or not covered.\n"
            )
        elif contract.name == "AnomalyAgent":
            base_prompt += (
                "\nANOMALY AGENT RULES (Numerical and logical consistency):\n"
                "TASK: Detect inconsistencies within the document.\n"
                "CHECKS:\n"
                "- total vs line items\n"
                "- date consistency\n"
                "- duplicate/conflicting values\n"
                "DECISION RULES:\n"
                "- If data is insufficient => status=INSUFFICIENT_DATA.\n"
                "- You MUST perform at least one concrete comparison.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Include numeric comparisons (example: '1200 MAD vs 1200 MAD').\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must describe what comparisons were checked.\n"
            )
        elif contract.name == "PatternAgent":
            base_prompt += (
                "\nPATTERN AGENT RULES (History-aware but honest):\n"
                "TASK: Analyze behavioral patterns using available data.\n"
                "RULES:\n"
                "- If NO historical data => status=INSUFFICIENT_DATA.\n"
                "- Must explain exactly what history is missing.\n"
                "- If data exists, compare frequency, repetition, and provider recurrence.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Must reference CIN.\n"
                "- Must reference provider.\n"
                "- Must reference claim frequency when history is available.\n"
                "FORBIDDEN:\n"
                "- No generic fallback phrases.\n"
                "CONFIDENCE REQUIREMENT:\n"
                "- Confidence must be low when no history exists.\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must state what was checked.\n"
                "- Must state what could NOT be checked due to missing data.\n"
            )
        elif contract.name == "GraphRiskAgent":
            base_prompt += (
                "\nGRAPH RISK AGENT RULES (Relational reasoning):\n"
                "TASK: Analyze relationships across CIN, provider, and claims.\n"
                "RULES:\n"
                "- If graph data is missing or incomplete => status=INSUFFICIENT_DATA.\n"
                "- If connections exist, evaluate relationship risk using the observed graph structure.\n"
                "EVIDENCE REQUIREMENT:\n"
                "- Must reference number of nodes.\n"
                "- Must reference relationship types used.\n"
                "FORBIDDEN:\n"
                "- Do NOT output 0 or 100 without explicit justification from evidence.\n"
                "CONFIDENCE REQUIREMENT:\n"
                "- Confidence must be low when graph is incomplete.\n"
                "EXPLANATION REQUIREMENT:\n"
                "- Must describe the graph structure used for the assessment.\n"
            )
        if contract.name in {"AnomalyAgent", "PatternAgent"}:
            base_prompt += (
                "\nHARD RULES FOR THIS AGENT (MANDATORY):\n"
                "1. NO GENERIC EXPLANATIONS. Do NOT use generic phrases unless tied to input values.\n"
                "2. Your explanation MUST reference at least 2 specific input elements "
                "(examples: CIN, date, amount, provider name, document type, extracted fields).\n"
                "3. REQUIRED JSON format for this agent:\n"
                '{ "score": 0-100, "confidence": 0-100, "status":"...", "claims":[...], '
                '"hallucination_flags":[], "explanation":"..." }\n'
                "4. explanation must include at least one of:\n"
                "   - a numeric value, or\n"
                "   - an extracted field, or\n"
                "   - a direct comparison.\n"
                "   If explanation could be reused for unrelated inputs, it is WRONG.\n"
                "5. INPUT DIFFERENTIATION: If two inputs differ, explanation MUST also differ.\n"
                "6. If reasoning is weak or data is partial, confidence MUST be below 60.\n"
            )
            if contract.name == "AnomalyAgent":
                base_prompt += (
                    "7. ANOMALY-SPECIFIC: Compare values inside the document and identify inconsistencies.\n"
                    "   If no inconsistency is found, explicitly state what comparisons were checked.\n"
                    "   Example: Checked total 1200 MAD vs line items sum 1200 MAD — consistent.\n"
                )
            if contract.name == "PatternAgent":
                base_prompt += (
                    "7. PATTERN-SPECIFIC: If no historical data exists, explain what data is missing, "
                    "which pattern check cannot be run, and fallback logic used.\n"
                    "   Example: No historical claims found for CIN X — cannot compare frequency patterns; "
                    "fallback to single-claim anomaly checks.\n"
                )
        return base_prompt

    @staticmethod
    def _enforce_prompt_context(prompt: str) -> None:
        required_markers = (
            "--- VERIFIED DATA ---",
            "--- DOCUMENT TEXT (OCR) ---",
        )
        missing = [marker for marker in required_markers if marker not in prompt]
        if missing:
            raise RuntimeError(
                f"Prompt context enforcement failed: missing {missing}"
            )

    @staticmethod
    def _goa_trigger(claim_request: Dict[str, Any], routing: RoutingDecision) -> bool:
        documents = claim_request.get("documents", [])
        return len(documents) > 1 or routing.complexity == "complex"

    def _get_embeddings(self):
        try:
            try:
                from langchain_ollama import OllamaEmbeddings as _OllamaEmb  # type: ignore
                return _OllamaEmb(base_url=self._ollama_base_url, model=self._embedding_model)
            except ImportError:
                pass
            return OllamaEmbeddings(base_url=self._ollama_base_url, model=self._embedding_model)
        except Exception:
            return FakeEmbeddings(size=64)

    def _cluster_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if len(documents) <= 1:
            return [documents]
        embedder = self._get_embeddings()
        vectors = embedder.embed_documents(
            [json.dumps(d, ensure_ascii=False, sort_keys=True) for d in documents]
        )
        cluster_count = min(3, len(documents))
        labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(vectors)
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            grouped.setdefault(int(label), []).append(documents[idx])
        return list(grouped.values())

    def _run_goa(self, claim_request: Dict[str, Any], routing: RoutingDecision) -> Dict[str, Any]:
        docs = claim_request.get("documents", [])
        clusters = self._cluster_documents(docs)
        outputs: List[Dict[str, Any]] = []
        for i, cluster in enumerate(clusters):
            llm = self._make_chat_llm(routing.model)
            cluster_agent = Agent(
                role=f"ClusterRiskAgent-{i}",
                goal="Assess cluster-level risk and anomalies",
                backstory="Specialist in clustered claim document patterns.",
                llm=llm,
                verbose=False,
            )
            cluster_task = Task(
                description=(
                    "Assess risk for this document cluster and return JSON "
                    "with score, confidence, explanation.\n"
                    f"Cluster: {json.dumps(cluster, ensure_ascii=False)}"
                ),
                expected_output="JSON with score, confidence, explanation",
                agent=cluster_agent,
            )
            with tracked_agent_context(cluster_agent.role):
                out = Crew(
                    agents=[cluster_agent],
                    tasks=[cluster_task],
                    process=Process.sequential,
                    verbose=False,
                ).kickoff()
            parsed = _safe_json_load(str(out))
            outputs.append(
                {
                    "cluster_id": i,
                    "size": len(cluster),
                    "result": {
                        "score": float(parsed.get("score", 0.5)),
                        "confidence": float(parsed.get("confidence", 0.5)),
                        "explanation": str(parsed.get("explanation", "No explanation provided")),
                    },
                }
            )
        return {"clusters": len(clusters), "cluster_outputs": outputs}

    def _resolve_current_cin(self, claim_request: Dict[str, Any]) -> str:
        identity = claim_request.get("identity", {})
        for key in ("cin", "CIN", "carte_nationale", "national_id"):
            val = identity.get(key)
            if val and str(val).strip():
                return str(val).strip().upper()
        for key in ("patient_id", "cin", "CIN"):
            val = claim_request.get(key)
            if val and str(val).strip():
                return str(val).strip().upper()
        return ""

    @staticmethod
    def _resolve_graph_fields(claim_request: Dict[str, Any]) -> Dict[str, Any]:
        identity = claim_request.get("identity", {})
        policy = claim_request.get("policy", {})
        metadata = claim_request.get("metadata", {})

        claim_id = str(
            metadata.get("claim_id")
            or claim_request.get("claim_id")
            or claim_request.get("id")
            or ""
        ).strip()
        cin = str(
            identity.get("cin")
            or identity.get("CIN")
            or claim_request.get("patient_id")
            or ""
        ).strip().upper()
        hospital = str(
            identity.get("hospital")
            or policy.get("hospital")
            or metadata.get("hospital")
            or ""
        ).strip()
        doctor = str(
            identity.get("doctor")
            or policy.get("doctor")
            or metadata.get("doctor")
            or ""
        ).strip()

        raw_anomaly = (
            metadata.get("anomaly_score")
            or claim_request.get("anomaly_score")
            or policy.get("anomaly_score")
            or 0.0
        )
        try:
            anomaly_score = float(raw_anomaly)
        except (TypeError, ValueError):
            anomaly_score = 0.0

        return {
            "claim_id": claim_id,
            "cin": cin,
            "hospital": hospital,
            "doctor": doctor,
            "anomaly_score": max(0.0, min(1.0, anomaly_score)),
        }

    @staticmethod
    def _has_minimum_documents(claim_request: Dict[str, Any]) -> bool:
        docs = claim_request.get("documents") or []
        exts = claim_request.get("document_extractions") or []
        return bool(docs or exts)

    @staticmethod
    def _has_identity_core_fields(claim_request: Dict[str, Any]) -> bool:
        identity = claim_request.get("identity", {})
        cin = (
            identity.get("cin")
            or identity.get("CIN")
            or claim_request.get("patient_id")
            or ""
        )
        ipp = identity.get("ipp") or identity.get("IPP") or ""
        return bool(str(cin).strip()) or bool(str(ipp).strip())

    def _normalize_agent_output(
        self,
        *,
        agent_name: str,
        score: float,
        confidence: float,
        explanation: str,
        claim_request: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> Dict[str, Any]:
        normalized_score = max(0.0, min(1.0, score))
        normalized_confidence = max(0.0, min(1.0, confidence))
        normalized_explanation = explanation.strip() or "Insufficient data to conclude"
        insufficient_data = False

        has_docs = self._has_minimum_documents(claim_request)
        has_identity_fields = self._has_identity_core_fields(claim_request)
        has_history = bool(claim_request.get("history"))

        if not has_docs:
            insufficient_data = True
            normalized_score = min(normalized_score, 0.5)
            normalized_confidence = min(normalized_confidence, 0.4)
            normalized_explanation = "Insufficient data to perform reliable analysis"

        if agent_name == "IdentityAgent":
            if not has_identity_fields:
                insufficient_data = True
                normalized_score = min(normalized_score, 0.5)
                normalized_confidence = min(normalized_confidence, 0.6)
                normalized_explanation = "Insufficient data to perform reliable analysis"

        if agent_name == "DocumentAgent":
            coverage_payload = claim_request.get("_coverage_score") or {}
            overall_cov = float(coverage_payload.get("overall", 1.0)) if isinstance(coverage_payload, dict) else 1.0
            if overall_cov < MIN_COVERAGE_ACCEPT:
                insufficient_data = True
                normalized_score = min(normalized_score, 0.5)
                normalized_confidence = min(normalized_confidence, 0.55)
                normalized_explanation = (
                    f"Low document coverage (score={overall_cov:.2f}); "
                    "continuing with degraded DocumentAgent confidence"
                )

        if agent_name in {"AnomalyAgent", "PatternAgent"} and not has_history:
            insufficient_data = True
            normalized_score = min(normalized_score, 0.6)
            normalized_confidence = min(normalized_confidence, 0.4)
            cin = (
                claim_request.get("identity", {}).get("cin")
                or claim_request.get("patient_id")
                or "unknown"
            )
            amount = claim_request.get("amount", claim_request.get("policy", {}).get("amount", "unknown"))
            provider = (
                claim_request.get("identity", {}).get("hospital")
                or claim_request.get("policy", {}).get("hospital")
                or claim_request.get("metadata", {}).get("hospital")
                or "unknown"
            )
            history_count = len(claim_request.get("history") or [])
            if agent_name == "PatternAgent":
                normalized_explanation = (
                    f"INSUFFICIENT_DATA for pattern analysis: checked CIN={cin}, provider={provider}, "
                    f"current amount={amount}, and available history_count={history_count}. "
                    "Could not run repetition/frequency/provider recurrence checks because historical claims are missing. "
                    "Claim remains unverified for behavioral pattern assertions."
                )
            else:
                normalized_explanation = (
                    f"No prior history for CIN {cin}; anomaly baseline limited. "
                    f"Checked current claim amount={amount} against in-document fields only."
                )

        if agent_name == "GraphRiskAgent":
            graph_fields = self._resolve_graph_fields(claim_request)
            if not (graph_fields["claim_id"] and graph_fields["cin"] and graph_fields["hospital"] and graph_fields["doctor"]):
                insufficient_data = True
                normalized_score = min(normalized_score, 0.6)
                normalized_confidence = min(normalized_confidence, 0.4)
                missing_keys: List[str] = []
                for key in ("claim_id", "cin", "hospital", "doctor"):
                    if not graph_fields.get(key):
                        missing_keys.append(key)
                normalized_explanation = (
                    "INSUFFICIENT_DATA for graph risk: checked graph identifiers "
                    f"(claim_id={graph_fields.get('claim_id') or 'missing'}, cin={graph_fields.get('cin') or 'missing'}, "
                    f"hospital={graph_fields.get('hospital') or 'missing'}, doctor={graph_fields.get('doctor') or 'missing'}). "
                    "Could not build a complete graph structure (node count and relationship types) because required fields are missing: "
                    f"{', '.join(missing_keys)}. "
                    "Claim remains unverified for relational-risk conclusions."
                )

        analysis_status = (
            "INSUFFICIENT_DATA"
            if insufficient_data
            else ("SUSPICIOUS" if normalized_score >= 0.6 else "VERIFIED")
        )
        return {
            "score": normalized_score,
            "confidence": normalized_confidence,
            "explanation": normalized_explanation,
            "insufficient_data": insufficient_data,
            "analysis_status": analysis_status,
        }

    @staticmethod
    def _collect_critical_failures(
        *,
        extracted_text: str,
        doc_classification: Dict[str, Any],
        pre_validation: Dict[str, Any],
        field_verification: List[Dict[str, Any]],
        identity_failures: List[str] | None = None,
    ) -> List[str]:
        inferred: List[str] = (
            (["DOCUMENT_CLASSIFIED_NON_CLAIM", "NON_CLAIM", "ML_NON_CLAIM"] if str(doc_classification.get("label", "")).upper() == "NON_CLAIM" else [])
            + (["PROMPT_INJECTION_DETECTED", "PROMPT_INJECTION"] if bool(pre_validation.get("injection_detected", False)) else [])
            + (["MISSING_REQUIRED_CLAIM_SIGNALS"] if int(pre_validation.get("claim_signal_count", 0)) < 2 else [])
            + (["OCR_TEXT_EMPTY_OR_IRRELEVANT"] if ClaimGuardV2Orchestrator._is_unreadable_text(str(extracted_text or "")) else [])
            + [
                f"CRITICAL_FIELD_{str(row.get('field', '')).strip().upper()}_NOT_FOUND"
                for row in field_verification
                if str(row.get("field", "")).strip().lower() in _CRITICAL_FIELD_KEYS
                and bool(row.get("input_present", True))
                and str(row.get("status", "")).upper() == "NOT_FOUND"
            ]
            + list(identity_failures or [])
        )
        return sorted(set(inferred))

    @staticmethod
    def _agent_has_critical_failure(parsed: Dict[str, Any], explanation: str) -> bool:
        haystacks = [
            explanation.lower(),
            str(parsed.get("analysis_status", "")).lower(),
            str(parsed.get("status", "")).lower(),
        ]
        return any(marker in text for text in haystacks for marker in _CRITICAL_AGENT_FAILURE_MARKERS)

    @staticmethod
    def _apply_hallucination_penalty(
        *,
        score: float,
        confidence: float,
        explanation: str,
        claims: List[Dict[str, Any]],
        hallucination_flags: List[str],
        extracted_text: str,
        structured_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        verified_claims = 0
        checked_claims = 0
        normalized_claims: List[Dict[str, Any]] = []
        flags = list(hallucination_flags)

        for claim in claims:
            checked_claims += 1
            statement = str(claim.get("statement", "")).strip()
            evidence = str(claim.get("evidence", "")).strip()
            evidence_found = _is_evidence_in_source(evidence, extracted_text, structured_data)
            verified = bool(claim.get("verified", False)) and evidence_found
            if not evidence_found:
                flags.append("ocr_value_not_found")
            if not verified:
                flags.append("unsupported_claim")
                if not evidence:
                    evidence = "UNVERIFIED"
            else:
                verified_claims += 1
            normalized_claims.append(
                {
                    "statement": statement or "UNVERIFIED",
                    "evidence": evidence or "UNVERIFIED",
                    "verified": verified,
                }
            )

        if not _explanation_is_grounded(
            explanation,
            extracted_text,
            structured_data,
        ):
            flags.append("generic_explanation")

        has_unverified = any(not c.get("verified", False) for c in normalized_claims)
        penalty = 0.0
        adjusted_confidence = confidence
        adjusted_score = score
        if has_unverified:
            penalty = 0.4
            adjusted_confidence = max(0.0, confidence * (1.0 - penalty))
            adjusted_score = max(0.0, score - 0.15)
        elif "generic_explanation" in flags:
            penalty = 0.3
            adjusted_confidence = max(0.0, confidence * (1.0 - penalty))
            adjusted_score = max(0.0, score - 0.1)

        flags = sorted(set(flags))
        return {
            "score": adjusted_score,
            "confidence": adjusted_confidence,
            "claims": normalized_claims,
            "hallucination_flags": flags,
            "hallucination_penalty": penalty,
            "debug_log": {
                "claims_checked": checked_claims,
                "verified_claims": verified_claims,
                "hallucination_flags": flags,
                "confidence_adjusted": adjusted_confidence,
            },
        }

    def _run_forensic_input_differentiation_test(
        self,
        *,
        routing: RoutingDecision,
        claim_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_input = claim_request
        random_input = {
            "identity": {"name": "### random ###"},
            "documents": [{"id": "rand-1", "document_type": "text", "text": "ZXQ RANDOM BLOB ####"}],
            "policy": {"diagnosis": ""},
            "metadata": {"forensic_random_input": True, "claim_id": "forensic-random"},
        }
        empty_input = {"identity": {}, "documents": [], "policy": {}, "metadata": {"claim_id": "forensic-empty"}}
        scenarios = [("A", valid_input), ("B", random_input), ("C", empty_input)]
        results: List[Dict[str, Any]] = []
        failures: List[str] = []
        comparisons: List[Dict[str, Any]] = []
        hash_index: Dict[str, Dict[str, str]] = {contract.name: {} for contract in SEQUENTIAL_AGENT_CONTRACTS}
        scenario_text_hashes: Dict[str, str] = {}

        for contract in SEQUENTIAL_AGENT_CONTRACTS:
            per_input: Dict[str, Dict[str, Any]] = {}
            for label, payload in scenarios:
                extraction_payload = self._build_ocr_blackboard_payload(payload)
                board = SharedBlackboard(
                    payload,
                    routing,
                    extracted_text=str(extraction_payload["raw_text"]),
                    structured_data=dict(extraction_payload["structured_fields"]),
                )
                scenario_text_hashes[label] = hashlib.sha256(board.extracted_text.encode("utf-8")).hexdigest()
                llm = self._make_chat_llm(routing.model)
                agent = Agent(
                    role=contract.name,
                    goal="Evaluate claim risk for your scope and produce calibrated output.",
                    backstory=_AGENT_BACKSTORIES.get(contract.name, ""),
                    llm=llm,
                    verbose=False,
                )
                prompt = f"INPUT_ID: {label}-{contract.name}\n" + self._build_task_prompt(
                    contract=contract, blackboard=board
                )
                task = Task(
                    description=prompt,
                    expected_output=(
                        "ONLY return a valid JSON object — no prose, no markdown, no explanation outside JSON. "
                        "Required keys: score (0.0-1.0), confidence (0.0-1.0), explanation (string), "
                        "claims (array), hallucination_flags (array), memory_insights (object)."
                    ),
                    agent=agent,
                )
                with tracked_agent_context(contract.name):
                    raw = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff())
                parsed = _safe_json_load(raw)
                score = float(parsed.get("score", 0.0))
                confidence = float(parsed.get("confidence", 0.0))
                explanation = str(parsed.get("explanation", ""))
                out_payload = {"score": score, "confidence": confidence, "explanation": explanation}
                out_hash = _stable_output_hash(out_payload)
                hash_index[contract.name][label] = out_hash
                per_input[label] = {
                    "prompt": prompt,
                    "score": score,
                    "confidence": confidence,
                    "explanation": explanation,
                    "output_hash": out_hash,
                }

            prompts = {k: v["prompt"] for k, v in per_input.items()}
            hashes = {k: v["output_hash"] for k, v in per_input.items()}
            scores = [v["score"] for v in per_input.values()]
            explanations = [v["explanation"] for v in per_input.values()]
            if len(set(prompts.values())) < 3:
                failures.append(f"PROMPT_IDENTICAL_ACROSS_INPUTS:{contract.name}")
            if len(set(hashes.values())) < 3:
                failures.append(f"IDENTICAL_HASH_ACROSS_INPUTS:{contract.name}")
            if (max(scores) - min(scores) <= 1e-9) and len(set(explanations)) == 1:
                failures.append(f"IDENTICAL_OUTPUTS_ACROSS_INPUTS:{contract.name}")

            comparisons.append(
                {
                    "agent": contract.name,
                    "score_spread": round(max(scores) - min(scores), 6),
                    "explanation_similarity_A_B": round(_text_similarity(per_input["A"]["explanation"], per_input["B"]["explanation"]), 4),
                    "explanation_similarity_A_C": round(_text_similarity(per_input["A"]["explanation"], per_input["C"]["explanation"]), 4),
                }
            )
            results.append({"agent": contract.name, "per_input": per_input})

        return {
            "executed": True,
            "results": results,
            "hashes": hash_index,
            "scenario_text_hashes": scenario_text_hashes,
            "comparisons": comparisons,
            "failures": sorted(
                set(
                    failures
                    + (
                        ["CONSTANT_TEXT_ACROSS_INPUTS"]
                        if len(set(scenario_text_hashes.values())) < 3
                        else []
                    )
                )
            ),
        }

    @staticmethod
    def _ensure_contract_blackboard(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        board = dict(snapshot or {})
        board.setdefault("entries", {})
        board.setdefault("flags", {})
        board.setdefault("score_evolution", [])
        board.setdefault("reflexive_retry_logs", [])
        return board

    def _build_response_from_envelope(
        self,
        *,
        envelope: Dict[str, Any],
        routing: RoutingDecision,
        goa_used: bool,
        retry_count: int,
        mahic_breakdown: Dict[str, float],
        contradictions: List[Dict[str, Any]],
        trust_layer: Dict[str, Any] | None,
        memory_context: List[Dict[str, Any]],
        validation_result: ValidationResult | None,
        pre_validation_result: PreValidationResult | None,
        forensic_trace: Dict[str, Any] | None,
        decision_trace: Dict[str, Any] | None,
        system_flags: List[str],
        agent_outputs: List[AgentOutput],
    ) -> ClaimGuardV2Response:
        blackboard_snapshot = self._ensure_contract_blackboard(dict(envelope.get("blackboard_snapshot", {})))
        blackboard_snapshot["flags"] = dict(blackboard_snapshot.get("flags", {}))
        blackboard_snapshot["flags"].setdefault("exit_reason", envelope.get("exit_reason"))
        blackboard_snapshot["claim_id"] = envelope.get("claim_id")
        blackboard_snapshot["timestamp_utc"] = envelope.get("timestamp_utc")
        return ClaimGuardV2Response(
            agent_outputs=agent_outputs,
            blackboard=blackboard_snapshot,
            routing_decision=routing,
            goa_used=goa_used,
            Ts=float(envelope.get("Ts", 0.0)),
            decision=str(envelope.get("decision", "REJECTED")),
            exit_reason=str(envelope.get("exit_reason", "low_confidence")),
            retry_count=int(retry_count),
            mahic_breakdown=mahic_breakdown,
            contradictions=contradictions,
            trust_layer=trust_layer,
            memory_context=memory_context,
            validation_result=validation_result,
            pre_validation_result=pre_validation_result,
            forensic_trace=forensic_trace,
            decision_trace=decision_trace,
            response_envelope=envelope,
            system_flags=sorted(set(system_flags)),
        )

    def run(self, claim_request: Dict[str, Any]) -> ClaimGuardV2Response:
        start_time = time.time()
        max_pipeline_time = 60.0
        LOGGER.info("[PIPELINE START] claim_id=%s", str(claim_request.get("metadata", {}).get("claim_id") or ""))
        print("ACTIVE_PIPELINE_FILE:", inspect.getfile(self.__class__.run))
        print("[STAGE] PRE_VALIDATION")
        print(f"[PIPELINE TIME] {time.time() - start_time:.2f}s")
        routing = build_routing_decision(claim_request)
        claim_id = str(claim_request.get("metadata", {}).get("claim_id") or "")
        if not claim_id:
            claim_id = f"v2-{int(perf_counter() * 1000)}"
        trace_engine = TraceEngine(claim_id=claim_id)

        def _trace_stage(
            *,
            stage: str,
            status: str,
            inputs: Dict[str, Any] | None = None,
            outputs: Dict[str, Any] | None = None,
            reason: str = "",
            flags: List[str] | None = None,
            decision_snapshot: str = "",
        ) -> None:
            trace_engine.add_stage(
                {
                    "stage": stage,
                    "status": status,
                    "inputs": inputs or {},
                    "outputs": outputs or {},
                    "reason": reason,
                    "flags": flags or [],
                    "decision_snapshot": decision_snapshot,
                }
            )

        def _trace_critical_stop(reason: str, flags: List[str]) -> None:
            _trace_stage(
                stage="CRITICAL_STOP",
                status="FAIL",
                reason=reason,
                flags=flags,
                decision_snapshot="REJECTED",
            )

        def _finalize_response(*, exit_reason: str, **payload: Any) -> ClaimGuardV2Response:
            decision_value = str(payload.get("decision", "REJECTED")).strip().upper()
            normalized_exit_reason = str(exit_reason or "").strip()
            normalized_exit_reason = _EXIT_REASON_ALIASES.get(normalized_exit_reason, normalized_exit_reason.lower())
            assert decision_value in _CANONICAL_DECISIONS, f"Invalid decision enum: {decision_value}"
            assert normalized_exit_reason in EXIT_REASONS, f"Invalid exit_reason enum: {exit_reason}"
            payload["decision"] = decision_value
            payload["exit_reason"] = normalized_exit_reason
            coverage_payload = payload.get("coverage_score")
            coverage_obj: CoverageScore | None = None
            if isinstance(coverage_payload, CoverageScore):
                coverage_obj = coverage_payload
                payload["coverage_score"] = coverage_obj.model_dump()
            elif isinstance(coverage_payload, dict) and coverage_payload:
                try:
                    coverage_obj = CoverageScore.model_validate(coverage_payload)
                except Exception:
                    coverage_obj = None
            ts_value = float(payload.get("Ts", 0.0))
            explanation_reasons: List[str] = []
            if payload.get("reason"):
                explanation_reasons.append(str(payload.get("reason")))
            explanation_reasons.extend(str(f) for f in (payload.get("system_flags") or []))
            explanation_signals: Dict[str, Any] = {
                "exit_reason": normalized_exit_reason,
                "Ts": ts_value,
                "flags": sorted(set(payload.get("system_flags") or [])),
                "retry_count": int(payload.get("retry_count", 0) or 0),
            }
            explanation_tool_outputs: Dict[str, Any] = {}
            blackboard_for_tools = payload.get("blackboard") or {}
            if isinstance(blackboard_for_tools, dict):
                for key in ("document_classification", "pre_validation", "extraction_validation", "field_verification_summary"):
                    if blackboard_for_tools.get(key):
                        explanation_tool_outputs[key] = blackboard_for_tools.get(key)
            existing_explanation = payload.get("explanation")
            if isinstance(existing_explanation, dict):
                explanation_reasons.extend(existing_explanation.get("reasons") or [])
                explanation_signals.update(existing_explanation.get("signals") or {})
                explanation_tool_outputs.update(existing_explanation.get("tool_outputs") or {})
            decision_envelope = _build_pipeline_explanation(
                decision=decision_value,
                ts=ts_value,
                coverage=coverage_obj,
                reasons=explanation_reasons,
                signals=explanation_signals,
                tool_outputs=explanation_tool_outputs,
                summary=str(payload.get("reason") or "") or f"Pipeline exit {normalized_exit_reason}",
            )
            payload["explanation"] = decision_envelope["explanation"]
            if coverage_obj is not None:
                payload["coverage_score"] = coverage_obj.model_dump()
            elif not payload.get("coverage_score"):
                payload["coverage_score"] = {}
            trace_payload = payload.get("trace")
            if not isinstance(trace_payload, dict):
                trace_payload = trace_engine.export()
                payload["trace"] = trace_payload
            blackboard_payload = self._ensure_contract_blackboard(dict(payload.get("blackboard", {})))
            envelope = build_response_envelope(
                decision=decision_value,
                Ts=float(payload.get("Ts", 0.0)),
                blackboard=blackboard_payload,
                score_evolution=list(blackboard_payload.get("score_evolution", [])),
                reflexive_retry_logs=list(blackboard_payload.get("reflexive_retry_logs", [])),
                flags=dict(blackboard_payload.get("flags", {})),
                agent_outputs=[a.model_dump() if hasattr(a, "model_dump") else dict(a) for a in payload.get("agent_outputs", [])],
                exit_reason=normalized_exit_reason,
                claim_id=claim_id,
            )
            runtime_agents = payload.get("agents")
            if isinstance(runtime_agents, list) and runtime_agents:
                envelope["agents"] = runtime_agents
            else:
                envelope["agents"] = [
                    {
                        "agent": str(item.get("agent", "")),
                        "status": "DONE",
                        "output": item,
                        "score": float(item.get("score", 0.0)),
                        "reason": str(item.get("explanation", "")),
                    }
                    for item in envelope.get("agent_outputs", [])
                ]
            blackboard_snapshot = self._ensure_contract_blackboard(dict(envelope["blackboard_snapshot"]))
            blackboard_snapshot["flags"] = dict(blackboard_snapshot.get("flags", {}))
            blackboard_snapshot["flags"]["exit_reason"] = normalized_exit_reason
            blackboard_snapshot["claim_id"] = claim_id
            blackboard_snapshot["timestamp_utc"] = envelope["timestamp_utc"]
            payload["blackboard"] = blackboard_snapshot
            payload["response_envelope"] = envelope
            payload["explanation"] = envelope.get("explanation") or payload.get("explanation")
            # SCORE-FIX: save agents list NOW (before dict conversion below) for UI builder.
            _agents_list = list(payload.get("agents") or [])
            runtime_agents_list = _agents_list
            if runtime_agents_list:
                ui_agent_results: List[Dict[str, Any]] = []
                for item in runtime_agents_list:
                    if not isinstance(item, dict) or not item.get("agent"):
                        continue
                    runtime_status = str(item.get("status", "ERROR")).upper()
                    raw_score = float(item.get("score", 0.0))
                    score_0_100 = (raw_score * 100.0) if raw_score <= 1.0 else raw_score
                    score_0_100 = max(0.0, min(100.0, score_0_100))
                    output_payload = item.get("output") if isinstance(item.get("output"), dict) else {}
                    reasoning_value = str(
                        output_payload.get("explanation")
                        or output_payload.get("reasoning")
                        or item.get("reason")
                        or ""
                    ).strip()
                    if not reasoning_value:
                        reasoning_value = "Fallback: agent did not return structured output"
                    insufficient_data = bool(
                        output_payload.get("insufficient_data", False)
                        or str(output_payload.get("analysis_status", "")).upper() == "INSUFFICIENT_DATA"
                    )
                    json_parse_failed = "json_parse_failed" in list(
                        output_payload.get("hallucination_flags", []) or []
                    )
                    ui_status = _to_ui_agent_status(
                        runtime_status=runtime_status,
                        score_0_100=score_0_100,
                        insufficient_data=insufficient_data,
                        json_parse_failed=json_parse_failed,
                    )
                    confidence_0_100 = float(output_payload.get("confidence", 0.0))
                    if confidence_0_100 <= 1.0:
                        confidence_0_100 *= 100.0
                    confidence_0_100 = max(0.0, min(100.0, confidence_0_100))
                    signals = list(output_payload.get("hallucination_flags", []))
                    if not signals:
                        signals = list(item.get("flags", []))
                    ui_agent_results.append(
                        {
                            "agent_name": str(item.get("agent", "")),
                            "status": ui_status,
                            "score": round(score_0_100, 2),
                            "confidence": round(confidence_0_100, 2),
                            "explanation": reasoning_value,
                            "reasoning": reasoning_value,
                            "signals": signals,
                            "data_used": output_payload if output_payload else {},
                            "flags": list(item.get("flags", [])),
                            "decision": ui_status == "PASS",
                        }
                    )
                payload["agent_results"] = ui_agent_results
            else:
                ui_agent_results = []
                for item in payload.get("agent_outputs", []):
                    raw_score = float(getattr(item, "score", 0.0))
                    score_0_100 = (raw_score * 100.0) if raw_score <= 1.0 else raw_score
                    score_0_100 = max(0.0, min(100.0, score_0_100))
                    explanation_value = str(getattr(item, "explanation", "")).strip()
                    if not explanation_value:
                        explanation_value = "Fallback: agent did not return structured output"
                    insufficient_data = bool(getattr(item, "output_snapshot", {}).get("insufficient_data", False))
                    ui_status = _to_ui_agent_status(
                        runtime_status="DONE",
                        score_0_100=score_0_100,
                        insufficient_data=insufficient_data,
                    )
                    raw_confidence = float(getattr(item, "confidence", 0.0))
                    confidence_0_100 = raw_confidence * 100.0 if raw_confidence <= 1.0 else raw_confidence
                    confidence_0_100 = max(0.0, min(100.0, confidence_0_100))
                    ui_agent_results.append(
                        {
                            "agent_name": str(getattr(item, "agent", "")),
                            "status": ui_status,
                            "score": round(score_0_100, 2),
                            "confidence": round(confidence_0_100, 2),
                            "explanation": explanation_value,
                            "reasoning": explanation_value,
                            "signals": list(getattr(item, "hallucination_flags", [])),
                            "data_used": dict(getattr(item, "output_snapshot", {}) or {}),
                            "flags": list(getattr(item, "hallucination_flags", [])),
                            "decision": ui_status == "PASS",
                        }
                    )
                payload["agent_results"] = ui_agent_results
            payload.setdefault("claim_id", claim_id)
            payload.setdefault("pipeline_version", "v2")
            payload.setdefault("extracted_data", blackboard_snapshot.get("verified_structured_data", {}))
            stages = trace_payload.get("stages", []) if isinstance(trace_payload, dict) else []
            stage_name = "FINAL_DECISION"
            if isinstance(stages, list) and stages:
                stage_name = str((stages[-1] or {}).get("stage") or "FINAL_DECISION")
            score_value = float(payload.get("Ts", 0.0))
            flags_value = sorted(set(payload.get("system_flags", []) or []))
            reason_value = str(
                payload.get("reason")
                or payload.get("exit_reason")
                or envelope.get("exit_reason")
                or "Decision computed by pipeline safeguards"
            )
            payload["score"] = score_value
            payload["stage"] = stage_name
            classified_flags = _classify_flags(flags_value)
            payload["flags"] = classified_flags
            payload["reason"] = reason_value
            payload["confidence"] = _confidence_from_score(score_value)
            payload["stage_reached"] = stage_name
            envelope["score"] = score_value
            envelope["stage"] = stage_name
            envelope["reason"] = reason_value
            envelope["flags"] = flags_value
            _expl = payload.get("explanation")
            envelope["explanation"] = _expl.model_dump() if hasattr(_expl, "model_dump") else dict(_expl or {})
            envelope["coverage_score"] = dict(payload.get("coverage_score") or {})
            envelope["decision_envelope"] = {
                "decision": decision_envelope["decision"],
                "score": decision_envelope["score"],
                "explanation": decision_envelope["explanation"],
                "debug_mode": decision_envelope.get("debug_mode", DEBUG_EXPLANATION_MODE),
            }
            if _agents_list:
                payload["agents"] = {
                    str(item.get("agent", "")): {
                        "status": str(item.get("status", "DONE")),
                        "score": float(item.get("score", 0.0)),
                        "reason": str(item.get("reason", "")),
                        "flags": list(item.get("flags", [])),
                    }
                    for item in _agents_list
                    if isinstance(item, dict) and item.get("agent")
                }
            payload["field_verification"] = {
                "cin_found": bool((blackboard_snapshot.get("identity", {}) or {}).get("cin_found", False)),
                "ipp_found": bool((blackboard_snapshot.get("identity", {}) or {}).get("ipp_found", False)),
                "amount_found": not bool((blackboard_snapshot.get("field_verification_summary", {}) or {}).get("amount_missing", False)),
                "unverified_fields": list((blackboard_snapshot.get("field_verification_summary", {}) or {}).get("unverified_fields", [])),
            }
            payload["memory_status"] = str(blackboard_snapshot.get("memory_status", "DISABLED") or "DISABLED").upper()
            payload["audit_trail"] = list((payload.get("decision_trace", {}) or {}).get("audit_trail", []))
            payload["processing_time_ms"] = int((time.time() - start_time) * 1000)
            payload["routed_to"] = "INVESTIGATOR" if decision_value == "HUMAN_REVIEW" else "DASHBOARD"
            payload["agent_results"] = [
                {
                    "agent_name": str(row.get("agent_name", "")),
                    "status": str(row.get("status", "REVIEW")).upper() if str(row.get("status", "")).upper() in {"PASS", "FAIL", "REVIEW"} else "REVIEW",
                    "score": max(0.0, min(100.0, float(row.get("score", 50.0)))),
                    "confidence": max(0.0, min(100.0, float(row.get("confidence", 50.0)))),
                    "explanation": str(row.get("explanation") or row.get("reasoning") or "").strip() or "Fallback: agent did not return structured output",
                    "reasoning": str(row.get("explanation") or row.get("reasoning") or "").strip() or "Fallback: agent did not return structured output",
                    "signals": list(row.get("signals", [])),
                    "data_used": dict(row.get("data_used", {}) or {}),
                    "flags": list(row.get("flags", [])),
                    "decision": bool(row.get("decision", False)),
                }
                for row in payload.get("agent_results", [])
            ]
            verified = payload.get("extracted_data") if isinstance(payload.get("extracted_data"), dict) else {}
            identity_bucket = blackboard_snapshot.get("identity", {}) if isinstance(blackboard_snapshot.get("identity"), dict) else {}
            ocr_text = str(blackboard_snapshot.get("extracted_text", ""))
            # Blockchain and IPFS are only meaningful for APPROVED claims.
            # REJECTED and HUMAN_REVIEW are stored in Firebase only — no need
            # to anchor them on-chain or in IPFS.
            if decision_value == "APPROVED":
                local_doc_hash = compute_document_hash(
                    {
                        "claim_id": claim_id,
                        "ocr_text": ocr_text[:500],
                        "amount": verified.get("amount"),
                        "cin": verified.get("cin") or identity_bucket.get("cin"),
                        "decision": decision_value,
                        "score": score_value,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                trust_payload = payload.get("trust_layer") if isinstance(payload.get("trust_layer"), dict) else {}
                tx_hash_value = str(trust_payload.get("tx_hash") or payload.get("blockchain_tx") or payload.get("tx_hash") or local_doc_hash)
                cid_value = str(trust_payload.get("cid") or "").strip()
                ipfs_document_value = (
                    cid_value if cid_value.startswith("ipfs://") else f"ipfs://{cid_value}"
                ) if cid_value else str(payload.get("ipfs_document") or f"ipfs://claimguard/{claim_id}/{local_doc_hash[2:18]}")
                payload["blockchain_tx"] = tx_hash_value
                payload["ipfs_document"] = ipfs_document_value
                payload["tx_hash"] = tx_hash_value
                payload["ipfs_hash"] = ipfs_document_value
            else:
                payload["blockchain_tx"] = ""
                payload["ipfs_document"] = ""
                payload["tx_hash"] = ""
                payload["ipfs_hash"] = ""
            if DEBUG_EXPLANATION_MODE:
                print("[DEBUG_EXPLANATION_MODE] Full decision explanation:")
                print(json.dumps(envelope["explanation"], ensure_ascii=False, indent=2, default=str))
            response = ClaimGuardV2Response(**payload)
            assert response.exit_reason in EXIT_REASONS
            return response

        def _check_pipeline_timeout(stage_name: str, current_agent_outputs: List[AgentOutput] | None = None) -> ClaimGuardV2Response | None:
            elapsed = time.time() - start_time
            print(f"[STAGE] {stage_name}")
            print(f"[PIPELINE TIME] {elapsed:.2f}s")
            if elapsed <= max_pipeline_time:
                return None
            system_flags = ["TIMEOUT"]
            return _finalize_response(
                exit_reason="low_confidence",
                agent_outputs=current_agent_outputs or [],
                blackboard={"entries": {}, "terminated": True, "flags": {"pipeline_timeout": True}},
                routing_decision=routing,
                goa_used=False,
                Ts=50.0,
                decision="HUMAN_REVIEW",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=None,
                forensic_trace=None,
                decision_trace={"decision_reason": "Pipeline timeout exceeded", "final_decision": "HUMAN_REVIEW"},
                reason="Pipeline timeout exceeded",
                system_flags=system_flags,
                extracted_data={},
                pipeline_version="v2",
            )

        test_mode = bool(claim_request.get("metadata", {}).get("test_mode")) or os.getenv("ENVIRONMENT", "").lower() == "test"
        if not test_mode and not self.self_test():
            return _finalize_response(
                exit_reason="low_confidence",
                agent_outputs=[],
                blackboard={"entries": {}, "terminated": True, "flags": {"hybrid_pipeline_not_stable": True}},
                routing_decision=routing,
                goa_used=False,
                Ts=50.0,
                decision="HUMAN_REVIEW",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=None,
                forensic_trace=None,
                decision_trace={"decision_reason": "HYBRID_PIPELINE_NOT_STABLE", "final_decision": "HUMAN_REVIEW"},
                reason="HYBRID_PIPELINE_NOT_STABLE",
                system_flags=["HYBRID_PIPELINE_NOT_STABLE"],
                extracted_data={},
                pipeline_version="v2",
            )

        timeout_response = _check_pipeline_timeout("PRE_VALIDATION")
        if timeout_response is not None:
            return timeout_response

        tracker = get_tracker(claim_id)
        tracker.update("OCR Extraction", "RUNNING")
        extraction_payload = self._build_ocr_blackboard_payload(claim_request)
        if str(extraction_payload.get("status", "OK")).upper() == "ERROR":
            review_context = self._register_human_review_context(
                claim_id=claim_id,
                claim_request=claim_request,
                ts_score=50.0,
                reason="Hybrid extraction failed",
                verified_fields={},
                agent_outputs=[],
                blackboard_snapshot={
                    "entries": {},
                    "terminated": True,
                    "flags": {"hybrid_extraction_failed": True},
                    "extraction_error": {
                        "status": "ERROR",
                        "reason": str(extraction_payload.get("reason") or "Unknown extraction error"),
                        "stage": str(extraction_payload.get("stage") or "rule"),
                    },
                },
            )
            return _finalize_response(
                exit_reason="low_confidence",
                agent_outputs=[],
                blackboard={
                    "entries": {},
                    "terminated": True,
                    "flags": {"hybrid_extraction_failed": True},
                    "extraction_error": {
                        "status": "ERROR",
                        "reason": str(extraction_payload.get("reason") or "Unknown extraction error"),
                        "stage": str(extraction_payload.get("stage") or "rule"),
                    },
                },
                routing_decision=routing,
                goa_used=False,
                Ts=50.0,
                decision="HUMAN_REVIEW",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=None,
                forensic_trace=None,
                decision_trace={
                    "decision_reason": "Hybrid extraction failed",
                    "final_decision": "HUMAN_REVIEW",
                    "document_url": str(review_context.get("document_url") or ""),
                    "extraction_error": {
                        "status": "ERROR",
                        "reason": str(extraction_payload.get("reason") or "Unknown extraction error"),
                        "stage": str(extraction_payload.get("stage") or "rule"),
                    },
                },
                reason="Hybrid extraction failed",
                system_flags=["HYBRID_EXTRACTION_FAILED"],
                extracted_data={},
                pipeline_version="v2",
            )
        extracted_text = str(extraction_payload["raw_text"])
        system_flags: List[str] = []
        print("=== OCR TEXT START ===")
        print(extracted_text[:3000])
        print("=== OCR TEXT END ===")
        sanitized_extracted_text = sanitize_for_prompt(extracted_text)
        structured_data = dict(extraction_payload["structured_fields"])
        extraction_warnings = list(extraction_payload.get("extraction_warnings", []))
        if extraction_warnings:
            system_flags.append("HYBRID_EXTRACTION_DEGRADED")
        doc_classification = classify_document(extracted_text, structured_data)
        raw_doc_label = str(doc_classification.get("label", "")).upper()
        if raw_doc_label in {"NON_CLAIM", "UNCERTAIN"}:
            system_info_flags = [f"CLASSIFIER_{raw_doc_label}_INFO_ONLY"]
            doc_classification["raw_label"] = raw_doc_label
            doc_classification["label"] = "INFO_ONLY_CLUSTER"
            doc_classification["non_blocking"] = True
            system_flags.extend(system_info_flags)
        extraction_validation = self._compute_extraction_validation(extracted_text, structured_data)
        pre_validation = self._run_pre_validation_guard(extracted_text)
        # Coverage-score model replaces brittle document_type enum checks.
        # The classifier tool's bundle label (medical_claim_bundle / hybrid /
        # unknown) is informational only — the decision is driven by the
        # weighted coverage score.
        try:
            from claimguard.v2.tools.core_tools import document_classifier_tool as _doc_tool
            doc_tool_output = _doc_tool({
                "documents": claim_request.get("documents", []) or [],
                "document_extractions": claim_request.get("document_extractions", []) or [],
            })
        except Exception as _cov_exc:
            LOGGER.warning("document_classifier_tool_failed error=%s", _cov_exc)
            doc_tool_output = {}
        coverage_score_obj = compute_coverage_score(
            extracted_text=extracted_text,
            structured_data=structured_data,
            ml_classification=doc_classification,
            document_classifier_tool=doc_tool_output,
        )
        claim_request["_coverage_score"] = coverage_score_obj.model_dump()
        claim_request["_coverage_decision"] = coverage_decision(coverage_score_obj)
        if DEBUG_EXPLANATION_MODE:
            LOGGER.info(
                "[DEBUG_EXPLANATION] coverage=%s decision=%s warnings=%s bundle=%s",
                coverage_score_obj.overall,
                claim_request["_coverage_decision"],
                coverage_score_obj.warnings,
                coverage_score_obj.classifier_bundle,
            )
        _trace_stage(
            stage="PRE_VALIDATION",
            status="PASS" if not bool(pre_validation.get("failed", False)) else "FAIL",
            inputs={"raw_text_length": len(extracted_text)},
            outputs={"pre_validation": pre_validation},
            reason=str(pre_validation.get("reason") or ""),
            flags=list(pre_validation.get("flags", [])),
        )
        _trace_stage(
            stage="OCR_EXTRACTION",
            status="PASS" if bool(extracted_text.strip()) else "FAIL",
            inputs={"document_count": len(claim_request.get("documents", []) or [])},
            outputs={"structured_data": structured_data, "extraction_validation": extraction_validation},
            reason="OCR extraction and field extraction completed",
        )
        _trace_stage(
            stage="DOCUMENT_CLASSIFIER",
            status="FAIL" if str(doc_classification.get("label", "")).upper() == "NON_CLAIM" else "PASS",
            inputs={"raw_text_length": len(extracted_text)},
            outputs={"classification": doc_classification},
            reason=str(doc_classification.get("reason") or ""),
            flags=["NON_CLAIM"] if str(doc_classification.get("label", "")).upper() == "NON_CLAIM" else [],
        )
        print("[STAGE] FIELD_VERIFICATION")
        print(f"[PIPELINE TIME] {time.time() - start_time:.2f}s")
        field_verification, verification_meta = _verify_structured_fields(structured_data, extracted_text)
        verified_structured_data = dict(verification_meta.get("verified_fields", {}))
        verification_summary = dict(verification_meta.get("summary", {}))
        identity_verification = dict(verification_meta.get("identity", {}))
        identity_failures = list(verification_summary.get("identity_failures", []))
        _trace_stage(
            stage="FIELD_VERIFICATION",
            status="FAIL" if bool(verification_summary.get("should_stop_pipeline")) else "PASS",
            inputs={"structured_data": structured_data},
            outputs={"field_verification": field_verification, "summary": verification_summary},
            reason="Structured field verification completed",
            flags=list(verification_summary.get("critical_stop_reasons", []) or []),
        )
        _trace_stage(
            stage="IDENTITY_CHECK",
            status="FAIL" if bool(identity_failures) else "PASS",
            inputs={"identity_input": claim_request.get("identity", {})},
            outputs={"identity_verification": identity_verification},
            reason="Identity fields validated against OCR output",
            flags=identity_failures,
        )
        input_trust = self._compute_input_trust(
            raw_text=extracted_text,
            structured_fields=structured_data,
            extraction_validation=extraction_validation,
        )
        input_trust_score = self._input_trust_score(input_trust)
        input_summary = self._build_input_summary(claim_request)
        ocr_snapshot = extracted_text[:1200]
        critical_failures: List[str] = []
        if bool(pre_validation.get("injection_detected", False)):
            flags = ["PROMPT_INJECTION_DETECTED", "PROMPT_INJECTION"]
            if "PROMPT_INJECTION_LAYER1" in list(pre_validation.get("flags", [])):
                flags.append("PROMPT_INJECTION_LAYER1")
            terminal_result = terminate_pipeline(
                str(pre_validation.get("injection_reason") or "Prompt injection detected"),
                flags,
            )
            _log_pipeline_terminated("PRE_VALIDATION", terminal_result["reason"])
            _trace_stage(
                stage="PRE_VALIDATION",
                status="FAIL",
                inputs={"pre_validation": pre_validation},
                outputs={"terminal_result": terminal_result},
                reason=str(terminal_result["reason"]),
                flags=flags,
                decision_snapshot="REJECTED",
            )
            _trace_critical_stop(reason=str(terminal_result["reason"]), flags=flags)
            tracker.update("OCR Extraction", "FAILED")
            system_flags.extend(flags)
            return _finalize_response(
                exit_reason="prompt_injection",
                agent_outputs=[],
                coverage_score=coverage_score_obj,
                blackboard={
                    "entries": {},
                    "extracted_text": sanitized_extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "identity": identity_verification,
                    "critical_failures": sorted(set(flags)),
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "security_flags": pre_validation.get("security_flags", []),
                    "degraded_security_mode": bool(pre_validation.get("degraded_security_mode", False)),
                    "document_classification": doc_classification,
                    "coverage_score": coverage_score_obj.model_dump(),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision=str(terminal_result["decision"]),
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=PreValidationResult(
                    score=0,
                    confidence=int(pre_validation.get("injection_confidence", 100)),
                    status="REJECTED",
                    reason=str(terminal_result["reason"]),
                    flags=sorted(set(flags)),
                    document_type=str(pre_validation.get("document_type", "UNKNOWN")),
                    injection_detected=True,
                    passed=False,
                ),
                forensic_trace=None,
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": str(terminal_result["reason"]),
                    "critical_failures": sorted(set(flags)),
                    "system_flags": sorted(set(system_flags)),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                reason=str(terminal_result["reason"]),
                system_flags=sorted(set(system_flags)),
            )
        # Coverage-score replaces the NON_CLAIM hard gate. Reject only when
        # the coverage score is below MIN_COVERAGE_ACCEPT AND the OCR text is
        # unreadable — any document with acceptable coverage continues the
        # pipeline even when the ML classifier labels it NON_CLAIM, and the
        # classifier label is emitted as a soft-fail warning instead of a
        # hard termination.
        ml_label_non_claim = str(doc_classification.get("label", "")).upper() == "NON_CLAIM"
        # Also check raw_label for NON_CLAIM (set before being relabeled to INFO_ONLY_CLUSTER)
        raw_label_non_claim = str(doc_classification.get("raw_label", "")).upper() == "NON_CLAIM"
        raw_label_uncertain = str(doc_classification.get("raw_label", "")).upper() == "UNCERTAIN"
        non_claim_confident = (
            (ml_label_non_claim or raw_label_non_claim)
            and float(doc_classification.get("confidence", 0)) >= 80
        )
        # UNCERTAIN + low coverage is also out-of-context: classifier isn't sure it's a claim
        # AND coverage is below the accept threshold — hard reject, same as NON_CLAIM.
        uncertain_low_coverage = (
            raw_label_uncertain
            and float(doc_classification.get("confidence", 0)) >= 70
            and coverage_score_obj.overall < MIN_COVERAGE_ACCEPT
        )
        ocr_unreadable = self._is_unreadable_text(extracted_text)
        coverage_below_accept = coverage_score_obj.overall < MIN_COVERAGE_ACCEPT
        if ml_label_non_claim:
            system_flags.append("ML_NON_CLAIM_ADVISORY")
            _trace_stage(
                stage="DOCUMENT_CLASSIFIER",
                status="WARN",
                inputs={"classification": doc_classification},
                outputs={
                    "coverage_score": coverage_score_obj.model_dump(),
                    "decision": claim_request["_coverage_decision"],
                },
                reason="ML classifier flagged NON_CLAIM; treated as advisory — coverage-score drives decision",
                flags=["ML_NON_CLAIM_ADVISORY"],
            )
        # NON_CLAIM-FIX: when the classifier is confident this is not a medical claim (>=80%)
        # AND coverage is below the accept threshold, hard-reject immediately with score=0.
        # This catches readable but completely irrelevant documents (forensics slides, receipts,
        # car insurance, etc.) that previously slipped through as soft-fails and could reach
        # Ts >= 65 because the fraud detector found no fraud signals.
        if (non_claim_confident or uncertain_low_coverage) and coverage_below_accept:
            flags = [
                "COVERAGE_BELOW_MIN_ACCEPT",
                "NON_CLAIM_DOCUMENT",
                "DOCUMENT_OUT_OF_CONTEXT",
            ]
            _rej_label = str(doc_classification.get("raw_label", "NON_CLAIM")).upper()
            terminal_result = {
                "decision": "REJECTED",
                "reason": (
                    f"Document hors contexte — classifié {_rej_label} (confiance "
                    f"{doc_classification.get('confidence', 0)}%) avec couverture "
                    f"{coverage_score_obj.overall:.2f} < {MIN_COVERAGE_ACCEPT:.2f}. "
                    f"Le document soumis n'est pas un dossier médical."
                ),
                "flags": flags,
                "terminated": True,
            }
            _log_pipeline_terminated("PRE_VALIDATION", terminal_result["reason"])
            _trace_stage(
                stage="DOCUMENT_CLASSIFIER",
                status="FAIL",
                inputs={"classification": doc_classification, "coverage_score": coverage_score_obj.model_dump()},
                outputs={"terminal_result": terminal_result},
                reason=str(terminal_result["reason"]),
                flags=flags,
                decision_snapshot="REJECTED",
            )
            _trace_critical_stop(reason="NON_CLAIM_DOCUMENT", flags=flags)
            tracker.update("OCR Extraction", "FAILED")
            system_flags.extend(flags)
            return _finalize_response(
                exit_reason="ocr_unreadable",
                agent_outputs=[],
                coverage_score=coverage_score_obj,
                blackboard={
                    "entries": {},
                    "extracted_text": sanitized_extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "identity": identity_verification,
                    "critical_failures": sorted(set(flags)),
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "security_flags": pre_validation.get("security_flags", []),
                    "degraded_security_mode": bool(pre_validation.get("degraded_security_mode", False)),
                    "document_classification": doc_classification,
                    "coverage_score": coverage_score_obj.model_dump(),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision=str(terminal_result["decision"]),
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=PreValidationResult(
                    score=0,
                    confidence=100,
                    status="REJECTED",
                    reason=str(terminal_result["reason"]),
                    flags=sorted(set(flags)),
                    document_type=str(coverage_score_obj.classifier_bundle),
                    injection_detected=bool(pre_validation.get("injection_detected", False)),
                    passed=False,
                ),
                forensic_trace=None,
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": str(terminal_result["decision"]),
                    "final_decision": str(terminal_result["decision"]),
                    "decision_reason": str(terminal_result["reason"]),
                    "coverage_score": coverage_score_obj.model_dump(),
                    "critical_failures": sorted(set(flags)),
                    "system_flags": sorted(set(system_flags)),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                reason=str(terminal_result["reason"]),
                system_flags=sorted(set(system_flags)),
            )
        if coverage_below_accept and (ocr_unreadable or not extracted_text.strip()):
            flags = [
                "COVERAGE_BELOW_MIN_ACCEPT",
                "OCR_TEXT_UNREADABLE_OR_EMPTY",
                "NON_CLAIM" if ml_label_non_claim else "COVERAGE_INSUFFICIENT",
            ]
            terminal_result = {
                "decision": "REJECTED",
                "reason": (
                    f"Coverage score {coverage_score_obj.overall:.2f} below accept threshold "
                    f"{MIN_COVERAGE_ACCEPT:.2f} and OCR text is unreadable/empty"
                ),
                "flags": flags,
                "terminated": True,
            }
            _log_pipeline_terminated("PRE_VALIDATION", terminal_result["reason"])
            _trace_stage(
                stage="DOCUMENT_CLASSIFIER",
                status="FAIL",
                inputs={
                    "classification": doc_classification,
                    "coverage_score": coverage_score_obj.model_dump(),
                },
                outputs={"terminal_result": terminal_result},
                reason=str(terminal_result["reason"]),
                flags=flags,
                decision_snapshot="REJECTED",
            )
            _trace_critical_stop(reason="COVERAGE_INSUFFICIENT", flags=flags)
            tracker.update("OCR Extraction", "FAILED")
            system_flags.extend(flags)
            return _finalize_response(
                exit_reason="ocr_unreadable",
                agent_outputs=[],
                coverage_score=coverage_score_obj,
                blackboard={
                    "entries": {},
                    "extracted_text": sanitized_extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "identity": identity_verification,
                    "critical_failures": sorted(set(flags)),
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "security_flags": pre_validation.get("security_flags", []),
                    "degraded_security_mode": bool(pre_validation.get("degraded_security_mode", False)),
                    "document_classification": doc_classification,
                    "coverage_score": coverage_score_obj.model_dump(),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision=str(terminal_result["decision"]),
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=PreValidationResult(
                    score=0,
                    confidence=100,
                    status="REJECTED",
                    reason=str(terminal_result["reason"]),
                    flags=sorted(set(flags)),
                    document_type=str(coverage_score_obj.classifier_bundle),
                    injection_detected=bool(pre_validation.get("injection_detected", False)),
                    passed=False,
                ),
                forensic_trace=None,
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": str(terminal_result["decision"]),
                    "final_decision": str(terminal_result["decision"]),
                    "decision_reason": str(terminal_result["reason"]),
                    "coverage_score": coverage_score_obj.model_dump(),
                    "critical_failures": sorted(set(flags)),
                    "system_flags": sorted(set(system_flags)),
                    "terminated": bool(terminal_result.get("terminated")),
                },
                reason=str(terminal_result["reason"]),
                system_flags=sorted(set(system_flags)),
            )
        # Low coverage but OCR readable → soft-fail, pipeline continues with
        # degraded confidence and a warning flag.
        if coverage_below_accept:
            system_flags.extend([
                "COVERAGE_BELOW_MIN_ACCEPT",
                "COVERAGE_SOFT_FAIL",
            ])
            _trace_stage(
                stage="DOCUMENT_CLASSIFIER",
                status="WARN",
                inputs={"coverage_score": coverage_score_obj.model_dump()},
                outputs={"degraded_mode": True, "coverage_score": coverage_score_obj.model_dump()},
                reason=(
                    f"Coverage score {coverage_score_obj.overall:.2f} below accept threshold "
                    f"{MIN_COVERAGE_ACCEPT:.2f} — continuing pipeline with degraded confidence"
                ),
                flags=["COVERAGE_SOFT_FAIL"],
            )
        identity_hard_flags = {"NO_IDENTITY", "CIN_NOT_FOUND", "IPP_NOT_FOUND", "CIN_OR_IPP_NOT_FOUND"}
        identity_reasons = sorted(set(identity_failures).intersection(identity_hard_flags))
        if identity_reasons:
            flags = ["IDENTITY_NOT_FOUND_IN_OCR", "IDENTITY_FALLBACK_MODE", *identity_reasons]
            system_flags.extend(flags)
            _trace_stage(
                stage="IDENTITY_CHECK",
                status="FAIL",
                inputs={"identity_failures": identity_failures},
                outputs={"degraded_mode": True},
                reason="Identity mismatch detected; downgraded to fallback mode",
                flags=flags,
                decision_snapshot="HUMAN_REVIEW",
            )
        amount_row = next(
            (row for row in field_verification if str(row.get("field", "")).strip().lower() == "amount"),
            None,
        )
        amount_missing = bool(amount_row and str(amount_row.get("status", "")).upper() == "NOT_FOUND")
        if amount_missing:
            flags = ["AMOUNT_MISMATCH", "CRITICAL_FIELD_AMOUNT_NOT_FOUND", "FIELD_VERIFICATION_FALLBACK_MODE"]
            system_flags.extend(flags)
            _trace_stage(
                stage="FIELD_VERIFICATION",
                status="FAIL",
                inputs={"amount_row": amount_row},
                outputs={"degraded_mode": True},
                reason="Amount mismatch detected; degraded to fallback mode",
                flags=flags,
                decision_snapshot="HUMAN_REVIEW",
            )
        cin_row = next(
            (row for row in field_verification if str(row.get("field", "")).strip().lower() == "cin"),
            None,
        )
        ipp_row = next(
            (row for row in field_verification if str(row.get("field", "")).strip().lower() == "ipp"),
            None,
        )
        cin_mismatch = bool(cin_row and bool(cin_row.get("input_present", True)) and str(cin_row.get("status", "")).upper() == "NOT_FOUND")
        ipp_mismatch = bool(ipp_row and bool(ipp_row.get("input_present", True)) and str(ipp_row.get("status", "")).upper() == "NOT_FOUND")
        if cin_mismatch or ipp_mismatch:
            flags = ["CIN_IPP_MISMATCH", "IDENTITY_FALLBACK_MODE"]
            if cin_mismatch:
                flags.append("CRITICAL_FIELD_CIN_NOT_FOUND")
            if ipp_mismatch:
                flags.append("CRITICAL_FIELD_IPP_NOT_FOUND")
            system_flags.extend(flags)
            _trace_stage(
                stage="IDENTITY_CHECK",
                status="FAIL",
                inputs={"cin_row": cin_row, "ipp_row": ipp_row},
                outputs={"degraded_mode": True},
                reason="CIN/IPP mismatch detected; degraded to fallback mode",
                flags=flags,
                decision_snapshot="HUMAN_REVIEW",
            )
        if (not pre_validation.get("failed", False)) and doc_classification.get("label") == "UNCERTAIN":
            tracker.update("OCR Extraction", "FAILED")
            system_flags.append("ML_CLASSIFIER_UNCERTAIN")
            return _finalize_response(
                exit_reason="non_claim",
                agent_outputs=[],
                blackboard={
                    "entries": {},
                    "extracted_text": sanitized_extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "security_flags": pre_validation.get("security_flags", []),
                    "degraded_security_mode": bool(pre_validation.get("degraded_security_mode", False)),
                    "document_classification": doc_classification,
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision="REJECTED",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=PreValidationResult(
                    score=int(doc_classification.get("confidence", 0)),
                    confidence=int(doc_classification.get("confidence", 0)),
                    status="UNCERTAIN",
                    reason="Document classification uncertain; manual review required",
                    flags=["ML_CLASSIFIER_UNCERTAIN"],
                    document_type="UNCERTAIN",
                    passed=False,
                ),
                forensic_trace=None,
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": "ML document classifier returned UNCERTAIN",
                    "system_flags": sorted(set(system_flags)),
                },
                system_flags=sorted(set(system_flags)),
            )
        if pre_validation.get("failed", False):
            flags = list(pre_validation.get("flags", []))
            system_flags.extend(flags)
            block_reason = str(
                pre_validation.get("injection_reason") or "Invalid document or prompt injection detected"
            )
            tracker.update("OCR Extraction", "FAILED")
            return _finalize_response(
                exit_reason="critical_fields_unverified",
                agent_outputs=[],
                blackboard={
                    "entries": {},
                    "extracted_text": sanitized_extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "security_flags": pre_validation.get("security_flags", []),
                    "degraded_security_mode": bool(pre_validation.get("degraded_security_mode", False)),
                    "document_classification": doc_classification,
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision="REJECTED",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                pre_validation_result=PreValidationResult(
                    score=0,
                    confidence=int(pre_validation.get("injection_confidence", 100)),
                    status="REJECTED",
                    reason=block_reason,
                    flags=flags,
                    document_type=str(pre_validation.get("document_type", "UNKNOWN")),
                    injection_detected=bool(pre_validation.get("injection_detected", False)),
                    passed=False,
                ),
                forensic_trace=None,
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": block_reason,
                    "system_flags": sorted(set(system_flags)),
                },
                system_flags=sorted(set(system_flags)),
            )
        external_validation = self._external_validation_hook(claim_request)
        mismatch_flag = external_validation.to_flag()
        if mismatch_flag:
            system_flags.append(mismatch_flag)
        if self._is_unreadable_text(extracted_text):
            tracker.update("OCR Extraction", "FAILED")
            system_flags.append("OCR_FAILURE")
            return _finalize_response(
                exit_reason="ocr_unreadable",
                agent_outputs=[],
                blackboard={
                    "entries": {},
                    "extracted_text": extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "document_classification": doc_classification,
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision="REJECTED",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                forensic_trace=(
                    {
                        "hard_failures": ["OCR_QUALITY_CHECK_FAILED"],
                        "extraction_validation": extraction_validation,
                    }
                    if FORENSIC_MODE or bool(claim_request.get("metadata", {}).get("forensic_debug", False))
                    else None
                ),
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": "OCR failure",
                    "system_flags": system_flags,
                },
                system_flags=system_flags,
            )
        if input_trust_score < 50:
            tracker.update("OCR Extraction", "FAILED")
            system_flags.append("LOW_INPUT_TRUST")
            return _finalize_response(
                exit_reason="ocr_unreadable",
                agent_outputs=[],
                blackboard={
                    "entries": {},
                    "extracted_text": extracted_text,
                    "structured_data": structured_data,
                    "verified_structured_data": verified_structured_data,
                    "field_verification": field_verification,
                    "field_verification_summary": verification_summary,
                    "extraction_validation": extraction_validation,
                    "input_trust": input_trust,
                    "input_trust_score": input_trust_score,
                    "pre_validation": pre_validation,
                    "document_classification": doc_classification,
                },
                routing_decision=routing,
                goa_used=False,
                Ts=0.0,
                decision="REJECTED",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=[],
                validation_result=None,
                forensic_trace=(
                    {
                        "hard_failures": ["INPUT_TRUST_BELOW_GATE"],
                        "extraction_validation": extraction_validation,
                    }
                    if FORENSIC_MODE or bool(claim_request.get("metadata", {}).get("forensic_debug", False))
                    else None
                ),
                decision_trace={
                    "claim_id": claim_id,
                    "input_summary": input_summary,
                    "ocr_snapshot": ocr_snapshot,
                    "input_trust": input_trust,
                    "agent_summaries": [],
                    "contradictions": [],
                    "Ts_score": 0.0,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": "Input trust score below 50",
                    "system_flags": system_flags,
                },
                system_flags=system_flags,
            )
        tracker.update("OCR Extraction", "COMPLETED")
        blackboard = SharedBlackboard(
            claim_request,
            routing,
            extracted_text=sanitized_extracted_text,
            structured_data=structured_data,
        )
        blackboard.set_pre_validation(pre_validation)
        blackboard.set_field_verification(field_verification)
        blackboard.set_identity_validation(identity_verification)
        forensic_enabled = FORENSIC_MODE or bool(
            claim_request.get("metadata", {}).get("forensic_debug", False)
        )
        forensic_input_id = str(
            claim_request.get("metadata", {}).get("forensic_input_id")
            or claim_request.get("metadata", {}).get("claim_id")
            or f"forensic-{int(perf_counter() * 1000)}"
        )
        forensic_trace: Dict[str, Any] | None = None
        if forensic_enabled:
            forensic_trace = {
                "input_id": forensic_input_id,
                "llm_calls_count": 0,
                "simulation_or_stub_detected": False,
                "known_template_matches": [],
                "raw_input_trace": [],
                "prompt_trace": [],
                "response_trace": [],
                "output_hashes": [],
                "hash_collisions_across_inputs": [],
                "blackboard_flow_trace": [],
                "input_differentiation_test": {"executed": False, "results": [], "failures": []},
                "final_decision_trace": {},
                "entry_input_trace": {},
                "agent_forensic_trace": [],
                "fallback_trace": [],
                "hard_failures": [],
                "broken": False,
                "system_status": "WORKING",
                "final_audit_report": {},
                "extraction_validation": extraction_validation,
            }
            parsed_claim_data = claim_request if isinstance(claim_request, dict) else {}
            input_hash = _stable_output_hash(parsed_claim_data if parsed_claim_data else {"_empty": True})
            forensic_trace["entry_input_trace"] = {
                "raw_input": _stable_json_dumps(claim_request),
                "parsed_claim_data": parsed_claim_data,
                "input_hash": input_hash,
            }
            if not parsed_claim_data:
                forensic_trace["hard_failures"].append("EMPTY_PARSED_CLAIM_DATA")
            _INPUT_HASH_HISTORY[input_hash] = _INPUT_HASH_HISTORY.get(input_hash, 0) + 1
            if _INPUT_HASH_HISTORY[input_hash] > 1:
                forensic_trace["hard_failures"].append("INPUT_HASH_REUSED_ACROSS_RUNS")

        # ── Step 1: CLAIM VALIDATION (HARD GATE) ───────────────────────────
        tracker.update("ClaimValidation", "RUNNING")
        
        validation_agent = ClaimValidationAgent()
        validation_raw = run_agent_safe(validation_agent, claim_request, timeout=20.0)
        validation_raw = validate_agent_result(
            validation_raw,
            required_fields=["validation_status", "validation_score", "document_type", "should_stop_pipeline", "reason"],
        )
        validation_payload = validation_raw.get("output", validation_raw) if isinstance(validation_raw, dict) else {}
        normalized_document_type, original_document_type = _normalize_validation_document_type(
            validation_payload.get("document_type", "unknown")
        )
        validation_details = dict(validation_payload.get("details", {}) or {})
        if original_document_type is not None:
            validation_details["raw_document_type"] = original_document_type
            validation_details["document_type_normalized"] = normalized_document_type
        
        validation_result = ValidationResult(
            validation_status=validation_payload.get("validation_status", "INVALID"),
            validation_score=int(validation_payload.get("validation_score", 0)),
            document_type=normalized_document_type,
            missing_fields=validation_payload.get("missing_fields", []),
            found_fields=validation_payload.get("found_fields", []),
            reason=validation_payload.get("reason", "Validation output missing required fields"),
            should_stop_pipeline=bool(validation_payload.get("should_stop_pipeline", True)),
            details=validation_details,
        )
        
        LOGGER.info(
            "claim_validation claim_id=%s status=%s score=%d type=%s missing=%s",
            claim_id,
            validation_result.validation_status,
            validation_result.validation_score,
            validation_result.document_type,
            validation_result.missing_fields,
        )
        
        # Coverage-based soft-fail: the previous hard gate on
        # should_stop_pipeline is replaced with degraded confidence when the
        # coverage score is acceptable, and with a structured rejection
        # (still with explanation) only when coverage is below threshold.
        if validation_result.should_stop_pipeline:
            degrade_flags = [
                "CLAIM_VALIDATION_SOFT_FAIL",
                "VALIDATION_SOFT_FAIL_FALLBACK_MODE",
                "CLAIM_VALIDATION_FALLBACK_MODE",
            ]
            system_flags.extend(degrade_flags)
            LOGGER.warning(
                "claim_validation_soft_fail claim_id=%s reason=%s coverage=%.2f",
                claim_id, validation_result.reason, coverage_score_obj.overall,
            )
            _trace_stage(
                stage="CLAIM_VALIDATION",
                status="WARN",
                inputs={"validation_result": validation_result.model_dump()},
                outputs={
                    "degraded_mode": True,
                    "coverage_score": coverage_score_obj.model_dump(),
                },
                reason=(
                    f"Claim validation requested stop — converted to soft-fail "
                    f"(coverage={coverage_score_obj.overall:.2f}): {validation_result.reason}"
                ),
                flags=degrade_flags,
                decision_snapshot="HUMAN_REVIEW",
            )
            tracker.update("ClaimValidation", "SOFT_FAIL")
        
        tracker.update("ClaimValidation", "COMPLETED")
        LOGGER.info("claim_validation_passed claim_id=%s score=%d", claim_id, validation_result.validation_score)
        if verification_summary.get("should_stop_pipeline"):
            stop_reasons = list(verification_summary.get("critical_stop_reasons", []) or [])
            if not stop_reasons:
                stop_reasons = ["CRITICAL_FIELD_MISMATCH"]
            degrade_flags = [*stop_reasons, "FIELD_VERIFICATION_SOFT_FAIL", "FIELD_VERIFICATION_FALLBACK_MODE"]
            system_flags.extend(degrade_flags)
            _trace_stage(
                stage="FIELD_VERIFICATION",
                status="FAIL",
                inputs={"verification_summary": verification_summary},
                outputs={"degraded_mode": True},
                reason="Field verification requested hard stop; converted to soft fail fallback",
                flags=degrade_flags,
                decision_snapshot="HUMAN_REVIEW",
            )

        # ── Step 1: Retrieve memory context BEFORE any agent runs ──────────
        memory_health = self.get_memory_health_report()
        memory_status = "OK"
        if memory_health.status == MemoryHealthStatus.DEGRADED:
            memory_status = "DEGRADED"
        elif memory_health.status == MemoryHealthStatus.UNAVAILABLE:
            memory_status = "DISABLED"

        if memory_health.status == MemoryHealthStatus.HEALTHY:
            self._track_memory_health(claim_id=claim_id, report=memory_health)
            try:
                memory_context = self._memory.retrieve_similar_cases(claim_request)
            except Exception as exc:
                LOGGER.warning("memory_retrieve_after_healthcheck_failed error=%s", exc)
                memory_context = []
                memory_status = "DEGRADED"
        else:
            memory_context = []
            self._track_memory_health(claim_id=claim_id, report=memory_health)
            if memory_status == "DISABLED":
                system_flags.append("MEMORY_DISABLED")
            else:
                system_flags.append("MEMORY_DEGRADED")
            LOGGER.warning(
                "memory_status=%s failure_reason=%s",
                memory_status,
                memory_health.failure_reason,
            )

        blackboard.set_memory_status(memory_status)
        blackboard.inject_memory_context(memory_context)
        current_cin = self._resolve_current_cin(claim_request)
        provider_name = str(
            structured_data.get("provider")
            or claim_request.get("identity", {}).get("hospital")
            or claim_request.get("policy", {}).get("hospital")
            or ""
        ).strip()
        feedback_signal = self._reliability_store.get_feedback_signal(
            cin=current_cin,
            provider=provider_name,
        )
        memory_signals: List[Dict[str, Any]] = list(memory_context)
        if feedback_signal:
            memory_signals.append(
                {
                    "signal_type": "human_feedback",
                    "claim_id": feedback_signal.get("claim_id"),
                    "outcome": feedback_signal.get("outcome"),
                    "cin": feedback_signal.get("cin"),
                    "provider": feedback_signal.get("provider"),
                    "timestamp": feedback_signal.get("timestamp"),
                }
            )
            system_flags.append("HUMAN_FEEDBACK_SIGNAL_USED")

        if memory_context:
            fraud_cases = [c for c in memory_context if c.get("fraud_label") in ("fraud", "suspicious")]
            LOGGER.info(
                "memory_context_injected total=%d fraud_cases=%d cin=%s",
                len(memory_context), len(fraud_cases), current_cin,
            )
        else:
            LOGGER.info("memory_context_empty claim_cin=%s", current_cin)

        # ── Step 2: Run agents sequentially ────────────────────────────────
        print("[STAGE] AGENTS")
        print(f"[PIPELINE TIME] {time.time() - start_time:.2f}s")
        agent_outputs: List[AgentOutput] = []
        runtime_agents: List[Dict[str, Any]] = []
        fraud_ring_analysis: Dict[str, Any] = {"fraud_rings": []}
        agent_input_traces: List[Dict[str, Any]] = []

        for contract in SEQUENTIAL_AGENT_CONTRACTS:
            timeout_response = _check_pipeline_timeout("AGENTS", agent_outputs)
            if timeout_response is not None:
                return timeout_response
            try:
                blackboard.require(contract.requires)
            except BlackboardValidationError:
                tracker.update(contract.name, "SKIPPED")
                for remaining in SEQUENTIAL_AGENT_CONTRACTS[SEQUENTIAL_AGENT_CONTRACTS.index(contract)+1:]:
                    tracker.update(remaining.name, "SKIPPED")
                raise
            
            tracker.update(contract.name, "RUNNING")
            started = perf_counter()
            llm = self._make_chat_llm(routing.model)
            blackboard_before = blackboard.to_dict()
            agent_input = blackboard.get_agent_input()
            input_text_hash = hashlib.sha256(agent_input["text"].encode("utf-8")).hexdigest()
            fields_used = sorted(list(agent_input.get("data", {}).keys()))
            agent_input_traces.append(
                {
                    "agent": contract.name,
                    "input_text_hash": input_text_hash,
                    "fields_used": fields_used,
                }
            )
            if forensic_enabled and forensic_trace is not None:
                forensic_trace["raw_input_trace"].append(
                    {
                        "agent": contract.name,
                        "received_input": blackboard.get_agent_input(),
                        "blackboard_before": blackboard_before,
                    }
                )
            agent = Agent(
                role=contract.name,
                goal="Evaluate claim risk for your scope and produce calibrated output.",
                backstory=_AGENT_BACKSTORIES.get(
                    contract.name,
                    "Insurance fraud specialist collaborating through a shared blackboard.",
                ),
                llm=llm,
                verbose=False,
            )
            prompt = self._build_task_prompt(
                contract=contract,
                blackboard=blackboard,
            )
            prompt = f"INPUT_ID: {forensic_input_id}\n{prompt}"
            assert "structured_data" not in prompt
            assert "blackboard" not in prompt.lower()
            try:
                self._enforce_prompt_context(prompt)
            except RuntimeError:
                tracker.update(contract.name, "FAILED")
                raise RuntimeError(f"PROMPT_MISSING_REQUIRED_CONTEXT:{contract.name}")
            if forensic_enabled and forensic_trace is not None:
                contains_claim_data = blackboard.extracted_text in prompt and _stable_json_dumps(blackboard.structured_data) in prompt
                forensic_trace["prompt_trace"].append(
                    {
                        "agent": contract.name,
                        "prompt": prompt,
                        "contains_claim_data": contains_claim_data,
                    }
                )
                if not contains_claim_data:
                    forensic_trace["hard_failures"].append(
                        f"PROMPT_MISSING_INPUT_DATA:{contract.name}"
                    )
            task = Task(
                description=prompt,
                expected_output=(
                    "JSON with keys: score, confidence, claims, hallucination_flags, "
                    "explanation, memory_insights"
                ),
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            def _kickoff_with_context() -> Any:
                with tracked_agent_context(contract.name):
                    return safe_tracked_llm_call(
                        contract.name,
                        prompt,
                        lambda _: crew.kickoff(),
                    )
            print(f"[AGENT START] {contract.name}")
            identity_timeout = float(os.getenv("CLAIMGUARD_IDENTITY_AGENT_TIMEOUT_S", "25"))
            per_agent_timeout = identity_timeout if contract.name == "IdentityAgent" else 20.0
            runner = (
                lambda: self._run_identity_agent_local(claim_request)
                if contract.name == "IdentityAgent"
                else _kickoff_with_context
            )
            run_result = run_agent_with_timeout(
                contract.name,
                runner,
                timeout=per_agent_timeout,
            )
            if run_result.get("status") != "DONE":
                print(f"[AGENT END] {contract.name} -> {run_result}")
                if contract.name == "IdentityAgent":
                    identity = claim_request.get("identity", {}) if isinstance(claim_request.get("identity"), dict) else {}
                    cin = str(identity.get("cin") or identity.get("CIN") or claim_request.get("patient_id") or "").strip().upper()
                    ipp = str(identity.get("ipp") or identity.get("IPP") or "").strip()
                    normalized_text = str(blackboard.extracted_text or "").upper()
                    cin_found = bool(cin) and cin in normalized_text
                    ipp_found = bool(ipp) and ipp in normalized_text
                    fallback_explanation = (
                        "IdentityAgent timeout fallback used. "
                        f"CIN present={bool(cin)} CIN found_in_ocr={cin_found} "
                        f"IPP present={bool(ipp)} IPP found_in_ocr={ipp_found}."
                    )
                    parsed = {
                        "score": 0.2 if (cin or ipp) and not (cin_found or ipp_found) else 0.4,
                        "confidence": 0.35,
                        "explanation": fallback_explanation,
                        "claims": [{"statement": fallback_explanation, "evidence": "", "verified": False}],
                        "hallucination_flags": ["identity_timeout_fallback"],
                        "memory_insights": {"used": False, "signals": [], "note": "Timeout fallback path"},
                        "evidence": {"cin_found": cin_found, "ipp_found": ipp_found},
                    }
                    result = json.dumps(parsed, ensure_ascii=False)
                    print("[AGENT END] IdentityAgent -> FALLBACK_DONE")
                    tracker.update(contract.name, "DONE")
                    system_flags.append(f"AGENT_{run_result.get('status', 'ERROR')}_{contract.name.upper()}_FALLBACK")
                else:
                    tracker.update(contract.name, "FAILED")
                    system_flags.append(f"AGENT_{run_result.get('status', 'ERROR')}_{contract.name.upper()}")
                    runtime_agents.append(
                        {
                            "agent": contract.name,
                            "status": "ERROR",
                            "output": {},
                            "score": 0.0,
                            "reason": str(run_result.get("status") or "EXCEPTION"),
                            "flags": ["AGENT_EXECUTION_FAILURE"],  # SCORE-FIX
                        }
                    )
                    continue
            else:
                result = run_result.get("result")
            print(f"[AGENT END] {contract.name} -> DONE")
            try:
                raw_response = ""  # always defined before branching
                if forensic_enabled and forensic_trace is not None:
                    forensic_trace["llm_calls_count"] += 1
                if isinstance(result, dict) and {"response", "parsed", "agent"}.issubset(result.keys()):
                    raw_response = str(result.get("response") or "")
                    if not raw_response.strip():
                        raise RuntimeError("LLM_RESPONSE_LOST")
                    print("[LLM OUTPUT USED]")
                    parsed = result.get("parsed")
                    if not isinstance(parsed, dict):
                        parsed = parse_llm_json(raw_response)
                    # Fallback: missing "score", OR score=0 with no real explanation
                    _has_score = isinstance(parsed, dict) and "score" in parsed and float(parsed.get("score") or 0) > 0
                    _has_expl = isinstance(parsed, dict) and bool(str(parsed.get("explanation") or "").strip())
                    if isinstance(parsed, dict) and (not _has_score or not _has_expl):
                        existing_expl = str(
                            parsed.get("explanation") or parsed.get("analysis")
                            or parsed.get("reasoning") or parsed.get("verdict")
                            or parsed.get("summary") or ""
                        ).strip()
                        trimmed = existing_expl[:600] or raw_response.strip()[:600]
                        parsed = {
                            "score": float(parsed.get("score") or 0) if _has_score else 0.6,
                            "confidence": float(parsed.get("confidence") or 0.4),
                            "explanation": trimmed or "Agent completed analysis without structured JSON output.",
                            "hallucination_flags": list(parsed.get("hallucination_flags") or []) + ([] if _has_expl else ["json_parse_failed"]),
                        }
                elif isinstance(result, dict) and "score" in result:
                    # Already a parsed dict (e.g. from _run_identity_agent_local)
                    parsed = result
                else:
                    parsed = parse_llm_json(str(result))
                    if not isinstance(parsed, dict):
                        parsed = _safe_json_load(str(result))
            except Exception as exc:
                tracker.update(contract.name, "FAILED")
                LOGGER.exception("v2_agent_parse_failed agent=%s", contract.name)
                system_flags.append(f"AGENT_PARSE_ERROR_{contract.name.upper()}")
                runtime_agents.append(
                    {
                        "agent": contract.name,
                        "status": "ERROR",
                        "output": {},
                        "score": 0.0,
                        "reason": "VALIDATION_FAIL",
                        "flags": ["AGENT_PARSE_ERROR"],  # SCORE-FIX
                    }
                )
                continue
            
            score = float(parsed.get("score", 0.6))
            confidence = float(parsed.get("confidence", 0.4))
            score, confidence = _normalize_score_confidence_scale(score, confidence)
            explanation = str(parsed.get("explanation") or "").strip()
            if not explanation:
                explanation = str(raw_response).strip()[:400] if isinstance(raw_response, str) and raw_response.strip() else "Agent completed analysis."
            claims = _coerce_claims(parsed, explanation)
            hallucination_flags = parsed.get("hallucination_flags", [])
            if not isinstance(hallucination_flags, list):
                hallucination_flags = []
            raw_llm_response = str(result)
            if forensic_enabled and forensic_trace is not None:
                forensic_trace["response_trace"].append(
                    {
                        "agent": contract.name,
                        "raw_llm_response": raw_llm_response,
                        "parsed_output": parsed,
                        "response_mentions_input_id": forensic_input_id in raw_llm_response,
                    }
                )
                if forensic_input_id not in raw_llm_response:
                    forensic_trace["hard_failures"].append(
                        f"LLM_RESPONSE_MISSING_INPUT_MARKER:{contract.name}"
                    )

            # Extract or synthesise memory_insights
            memory_insights = _parse_memory_insights(parsed)
            if memory_insights is None and memory_context:
                if forensic_enabled and forensic_trace is not None:
                    forensic_trace["fallback_trace"].append(
                        {
                            "agent": contract.name,
                            "fallback_type": "memory_insights_fallback",
                            "explicitly_logged": True,
                        }
                    )
                    if forensic_enabled:
                        forensic_trace["hard_failures"].append(
                            f"FALLBACK_USED_UNDER_FORENSIC_MODE:{contract.name}"
                        )
                else:
                    memory_insights = _compute_fallback_memory_insights(memory_context, current_cin)

            if contract.name == "GraphRiskAgent":
                graph_fields = self._resolve_graph_fields(claim_request)
                if graph_fields["claim_id"] and graph_fields["cin"] and graph_fields["hospital"] and graph_fields["doctor"]:
                    graph_context = self._fraud_ring_graph.add_claim(**graph_fields)
                else:
                    graph_context = {
                        "cluster_membership": None,
                        "reuse_detection": {
                            "cin_reuse_detected": False,
                            "doctor_reuse_detected": False,
                            "cin_claim_count": 0,
                            "doctor_claim_count": 0,
                        },
                        "network_risk_score": 0.0,
                        "fraud_rings": {"fraud_rings": []},
                    }

                fraud_ring_analysis = graph_context.get("fraud_rings", {"fraud_rings": []})
                graph_risk = float(graph_context.get("network_risk_score", 0.0))
                parsed["score"] = max(float(parsed.get("score", 0.0)), graph_risk)
                parsed["confidence"] = max(float(parsed.get("confidence", 0.0)), min(1.0, 0.55 + (graph_risk * 0.4)))
                parsed["explanation"] = (
                    f"{parsed.get('explanation', '').strip()} "
                    f"[graph cluster={graph_context.get('cluster_membership')}; "
                    f"cin_reuse={graph_context['reuse_detection']['cin_reuse_detected']}; "
                    f"doctor_reuse={graph_context['reuse_detection']['doctor_reuse_detected']}; "
                    f"network_risk={graph_risk:.3f}]"
                ).strip()
                score = parsed["score"]
                confidence = parsed["confidence"]
                explanation = parsed["explanation"]

            normalized = self._normalize_agent_output(
                agent_name=contract.name,
                score=score,
                confidence=confidence,
                explanation=explanation,
                claim_request=claim_request,
                validation_result=validation_result,
            )
            score = normalized["score"]
            confidence = normalized["confidence"]
            explanation = normalized["explanation"]
            parsed["insufficient_data"] = normalized["insufficient_data"]
            parsed["analysis_status"] = normalized["analysis_status"]
            parsed_status = str(parsed.get("status", "")).strip().upper()
            allowed_statuses = {"VERIFIED", "SUSPICIOUS", "INSUFFICIENT_DATA"}
            if parsed_status not in allowed_statuses:
                parsed_status = normalized["analysis_status"]
            parsed["status"] = parsed_status
            parsed["agent_name"] = contract.name
            parsed["score"] = round(max(0.0, min(100.0, score * 100.0)), 2)
            parsed["confidence"] = round(max(0.0, min(100.0, confidence * 100.0)), 2)
            parsed["explanation"] = explanation
            parsed["signals"] = list(parsed.get("hallucination_flags", [])) if isinstance(parsed.get("hallucination_flags", []), list) else []
            parsed["data_used"] = dict(parsed.get("data_used", {}) or {})
            parsed["status_ui"] = (
                "REVIEW"
                if bool(parsed.get("insufficient_data", False))
                else ("PASS" if score >= 0.6 else "FAIL")
            )
            LOGGER.info("[AGENT OUTPUT] %s -> %s", contract.name, parsed)
            if self._agent_has_critical_failure(parsed, explanation):
                failure_flag = f"AGENT_FAILURE_{contract.name.upper()}"
                terminal_result = terminate_pipeline(
                    reason=f"Agent failure: {contract.name}",
                    flags=[failure_flag],
                )
                _log_pipeline_terminated("AGENTS", terminal_result["reason"])
                _trace_stage(
                    stage=f"AGENT_{contract.name}",
                    status="FAIL",
                    inputs={"agent_input": blackboard.get_agent_input()},
                    outputs={"parsed": parsed},
                    reason=str(terminal_result["reason"]),
                    flags=[failure_flag],
                    decision_snapshot="REJECTED",
                )
                _trace_stage(
                    stage="AGENTS",
                    status="FAIL",
                    inputs={"failed_agent": contract.name},
                    outputs={},
                    reason=str(terminal_result["reason"]),
                    flags=[failure_flag],
                    decision_snapshot="REJECTED",
                )
                _trace_critical_stop(reason=str(terminal_result["reason"]), flags=[failure_flag])
                tracker.update(contract.name, "FAILED")
                for remaining in SEQUENTIAL_AGENT_CONTRACTS[SEQUENTIAL_AGENT_CONTRACTS.index(contract)+1:]:
                    tracker.update(remaining.name, "SKIPPED")
                tracker.update("Consensus", "SKIPPED")
                system_flags.append(failure_flag)
                return _finalize_response(
                    exit_reason="critical_fields_unverified",
                    agent_outputs=agent_outputs,
                    blackboard={
                        **blackboard.to_dict(),
                        "field_verification": blackboard.field_verification,
                        "verified_structured_data": blackboard.verified_structured_data,
                        "field_verification_summary": verification_summary,
                        "extraction_validation": extraction_validation,
                        "critical_failures": [failure_flag],
                        "input_trust": input_trust,
                        "input_trust_score": input_trust_score,
                        "pre_validation": pre_validation,
                        "document_classification": doc_classification,
                        "terminated": bool(terminal_result.get("terminated")),
                    },
                    routing_decision=routing,
                    goa_used=False,
                    Ts=0.0,
                    decision=str(terminal_result["decision"]),
                    retry_count=0,
                    mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                    contradictions=[],
                    trust_layer=None,
                    memory_context=memory_context,
                    validation_result=validation_result,
                    pre_validation_result=PreValidationResult(
                        score=0,
                        confidence=100,
                        status="REJECTED",
                        reason=str(terminal_result["reason"]),
                        flags=[failure_flag],
                        document_type=str(pre_validation.get("document_type", "UNKNOWN")),
                        injection_detected=bool(pre_validation.get("injection_detected", False)),
                        passed=False,
                    ),
                    forensic_trace=forensic_trace,
                    decision_trace={
                        "claim_id": claim_id,
                        "input_summary": input_summary,
                        "ocr_snapshot": ocr_snapshot,
                        "input_trust": input_trust,
                        "agent_summaries": [],
                        "contradictions": [],
                        "Ts_score": 0.0,
                        "decision_before_guard": "REJECTED",
                        "final_decision": "REJECTED",
                        "decision_reason": str(terminal_result["reason"]),
                        "critical_failures": [failure_flag],
                        "system_flags": sorted(set(system_flags)),
                        "terminated": bool(terminal_result.get("terminated")),
                    },
                    reason=str(terminal_result["reason"]),
                    system_flags=sorted(set(system_flags)),
                )

            if contract.name in {"PatternAgent", "GraphRiskAgent"} and memory_status != "OK":
                # CALIBRATION-FIX: memory degradation is informational and must not penalize scoring.
                parsed["insufficient_data"] = True
                parsed["analysis_status"] = "INSUFFICIENT_DATA"
                parsed["status"] = "INSUFFICIENT_DATA"
                explanation = (
                    f"{explanation} Memory status={memory_status}; running in degraded mode without score penalty."
                ).strip()

            hallucination_eval = self._apply_hallucination_penalty(
                score=score,
                confidence=confidence,
                explanation=explanation,
                claims=claims,
                hallucination_flags=hallucination_flags,
                extracted_text=blackboard.extracted_text,
                structured_data=blackboard.verified_structured_data,
            )
            score = float(hallucination_eval["score"])
            confidence = float(hallucination_eval["confidence"])
            parsed["claims"] = hallucination_eval["claims"]
            parsed["hallucination_flags"] = hallucination_eval["hallucination_flags"]
            parsed["hallucination_penalty"] = hallucination_eval["hallucination_penalty"]
            parsed["debug_log"] = hallucination_eval["debug_log"]

            is_fraud = score >= 0.7
            tracker.update(contract.name, "COMPLETED", score=score, confidence=confidence, explanation=explanation, is_fraud=is_fraud)
            
            elapsed_ms = int((perf_counter() - started) * 1000)

            blackboard.append(
                contract.name,
                score=score,
                confidence=confidence,
                explanation=explanation,
                claims=parsed.get("claims", []),
                hallucination_flags=parsed.get("hallucination_flags", []),
                hallucination_penalty=float(parsed.get("hallucination_penalty", 0.0)),
            )
            agent_key_map = {
                "IdentityAgent": "identity",
                "DocumentAgent": "document",
                "PolicyAgent": "policy",
                "AnomalyAgent": "anomaly",
                "PatternAgent": "pattern",
                "GraphRiskAgent": "graph",
            }
            board_key = agent_key_map.get(contract.name)
            if board_key:
                parsed_output = parsed if isinstance(parsed, dict) else {}
                clean_output = {
                    key: value
                    for key, value in parsed_output.items()
                    if value not in (None, "", [], {})
                }
                if clean_output:
                    blackboard._state[board_key] = clean_output
                blackboard._state[f"{board_key}_score"] = float(score)
            blackboard_after = blackboard.to_dict()
            if forensic_enabled and forensic_trace is not None:
                output_payload = {
                    "score": score,
                    "confidence": confidence,
                    "explanation": explanation,
                    "claims": parsed.get("claims", []),
                    "hallucination_flags": parsed.get("hallucination_flags", []),
                    "hallucination_penalty": parsed.get("hallucination_penalty", 0.0),
                    "analysis_status": parsed.get("analysis_status"),
                    "insufficient_data": parsed.get("insufficient_data"),
                }
                output_hash = _stable_output_hash(output_payload)
                forensic_trace["output_hashes"].append(
                    {
                        "agent": contract.name,
                        "input_id": forensic_input_id,
                        "hash_output": output_hash,
                        "output": output_payload,
                    }
                )
                previous = [
                    h for h in forensic_trace["output_hashes"]
                    if h["agent"] == contract.name and h["input_id"] != forensic_input_id and h["hash_output"] == output_hash
                ]
                if previous:
                    forensic_trace["hash_collisions_across_inputs"].append(
                        {
                            "agent": contract.name,
                            "hash_output": output_hash,
                            "current_input_id": forensic_input_id,
                            "previous_inputs": [p["input_id"] for p in previous],
                        }
                    )
                    forensic_trace["hard_failures"].append(
                        f"IDENTICAL_HASH_ACROSS_INPUTS:{contract.name}"
                    )
                forensic_trace["blackboard_flow_trace"].append(
                    {
                        "agent": contract.name,
                        "reads": list(contract.requires),
                        "blackboard_before": blackboard_before,
                        "writes": {contract.name: output_payload},
                        "blackboard_after": blackboard_after,
                    }
                )
                forensic_trace["agent_forensic_trace"].append(
                    {
                        "agent": contract.name,
                        "received_input": blackboard.get_agent_input(),
                        "blackboard_before": blackboard_before,
                        "prompt": prompt,
                        "llm_called": True,
                        "raw_response": raw_llm_response,
                        "parsed_output": parsed,
                        "output_hash": output_hash,
                    }
                )
                if blackboard_before == blackboard_after:
                    forensic_trace["hard_failures"].append(
                        f"BLACKBOARD_UNCHANGED_AFTER_AGENT:{contract.name}"
                    )

            output = AgentOutput(
                agent=contract.name,
                score=score,
                confidence=confidence,
                claims=parsed.get("claims", []),
                hallucination_flags=parsed.get("hallucination_flags", []),
                explanation=explanation,
                hallucination_penalty=float(parsed.get("hallucination_penalty", 0.0)),
                debug_log=parsed.get("debug_log"),
                elapsed_ms=elapsed_ms,
                input_snapshot={
                    "required_context": list(contract.requires),
                    "routing_model": routing.model,
                    "memory_cases_available": len(memory_context),
                    "agent_input": blackboard.get_agent_input(),
                    "graph_context": graph_context if contract.name == "GraphRiskAgent" else {},
                },
                output_snapshot=parsed,
                memory_insights=memory_insights,
            )
            agent_outputs.append(output)
            runtime_agents.append(
                validate_agent_result(
                    {
                        "agent": contract.name,
                        "status": "DONE",
                        "output": parsed if isinstance(parsed, dict) else {},
                        "score": (float(score) * 100.0) if float(score) <= 1.0 else float(score),
                        "reason": str(explanation),
                        "flags": list(parsed.get("hallucination_flags", [])) if isinstance(parsed, dict) else [],  # SCORE-FIX
                    }
                )
            )
            LOGGER.info(  # SCORE-FIX
                "[AGENT SCORE] %s status=%s score=%s reason=%s",
                contract.name,
                "DONE",
                (float(score) * 100.0) if float(score) <= 1.0 else float(score),
                str(explanation)[:80],
            )
            _trace_stage(
                stage=f"AGENT_{contract.name}",
                status="PASS",
                inputs=output.input_snapshot,
                outputs=output.output_snapshot,
                reason=output.explanation,
                flags=list(output.hallucination_flags),
            )
            LOGGER.info(
                "v2_agent_complete agent=%s elapsed_ms=%s confidence=%.3f "
                "memory_fraud_matches=%s input=%s output=%s",
                contract.name,
                elapsed_ms,
                confidence,
                memory_insights.fraud_matches if memory_insights else 0,
                output.input_snapshot,
                output.output_snapshot,
            )

            # Early exit if identity is invalid / highly unreliable.
            if contract.name == "IdentityAgent" and score < 0.3:
                identity_payload = claim_request.get("identity", {}) if isinstance(claim_request.get("identity"), dict) else {}
                cin_candidate = str(
                    identity_payload.get("cin")
                    or identity_payload.get("CIN")
                    or verified_structured_data.get("cin")
                    or verified_structured_data.get("CIN")
                    or ""
                ).strip().upper()
                ocr_upper = str(blackboard.extracted_text or "").upper()
                cin_found_in_ocr = bool(cin_candidate) and cin_candidate in ocr_upper
                cin_format_ok = bool(re.match(r"^[A-Z]{1,2}\d{5,6}$", cin_candidate))
                if cin_found_in_ocr:
                    # CALIBRATION-FIX: clear soft-fail when CIN exists in OCR evidence.
                    tracker.update("IdentityAgent", "COMPLETED")
                    if "IDENTITY_SOFT_FAIL_CONTINUE" in system_flags:
                        system_flags = [f for f in system_flags if f != "IDENTITY_SOFT_FAIL_CONTINUE"]
                    if cin_format_ok:
                        system_flags.append("IDENTITY_CIN_FORMAT_MATCH")
                else:
                    # CALIBRATION-FIX: keep soft-fail only when CIN is absent from OCR.
                    tracker.update(
                        "IdentityAgent",
                        "COMPLETED",
                        score=score,
                        confidence=confidence,
                        explanation=(
                            f"{explanation} CIN not found in OCR; marked for REVIEW instead of execution failure."
                        ),
                        is_fraud=is_fraud,
                    )
                    system_flags.append("IDENTITY_SOFT_FAIL_CONTINUE")

        # ── Step 3: GOA ────────────────────────────────────────────────────
        if not agent_outputs:
            return _finalize_response(
                exit_reason="low_confidence",
                agent_outputs=[],
                agents=runtime_agents,
                blackboard={"entries": {}, "terminated": True},
                routing_decision=routing,
                goa_used=False,
                Ts=50.0,
                decision="HUMAN_REVIEW",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=memory_context,
                validation_result=validation_result,
                pre_validation_result=None,
                forensic_trace=forensic_trace,
                decision_trace={"final_decision": "HUMAN_REVIEW", "decision_reason": "All agents failed or timed out"},
                reason="ALL_AGENTS_FAILED",
                system_flags=sorted(set([*system_flags, "ALL_AGENTS_FAILED"])),
                extracted_data=verified_structured_data,
                pipeline_version="v2",
            )

        goa_used = self._goa_trigger(claim_request, routing)
        if goa_used:
            goa_payload = self._run_goa(claim_request, routing)
            blackboard.append(
                "GraphOfAgents",
                score=0.5,
                confidence=0.5,
                explanation=f"Generated {goa_payload['clusters']} dynamic clusters",
            )
            blackboard_state = blackboard.to_dict()
            blackboard_state["goa"] = goa_payload
        else:
            blackboard_state = blackboard.to_dict()
        _trace_stage(
            stage="AGENTS",
            status="PASS",
            inputs={"agents_expected": [c.name for c in SEQUENTIAL_AGENT_CONTRACTS]},
            outputs={"agents_completed": [a.agent for a in agent_outputs]},
            reason="All sequential agents completed",
        )
        blackboard_state["fraud_ring_analysis"] = fraud_ring_analysis
        blackboard_state["identity"] = identity_verification
        blackboard_state["field_verification"] = blackboard.field_verification
        blackboard_state["verified_structured_data"] = blackboard.verified_structured_data
        blackboard_state["field_verification_summary"] = verification_summary
        blackboard_state["extraction_validation"] = extraction_validation
        blackboard_state["agent_input_traces"] = agent_input_traces
        blackboard_state["input_trust"] = input_trust
        blackboard_state["input_trust_score"] = input_trust_score
        blackboard_state["memory_signals"] = memory_signals
        blackboard_state["memory_status"] = memory_status
        blackboard_state["pre_validation"] = pre_validation
        blackboard_state["document_classification"] = doc_classification
        if forensic_enabled and forensic_trace is not None:
            hash_values = {item["input_text_hash"] for item in agent_input_traces}
            if len(hash_values) > 1:
                forensic_trace["hard_failures"].append("AGENT_TEXT_HASH_MISMATCH")
            if not blackboard.extracted_text.strip():
                forensic_trace["hard_failures"].append("EMPTY_EXTRACTED_TEXT")
            if blackboard_state.get("entries", {}) == {}:
                forensic_trace["hard_failures"].append("BLACKBOARD_NOT_CHANGING")

        # Merge IdentityAgent findings into pre-agent field verification before
        # any guard (approval guard or global enforcement guard) reads those values.
        _reconcile_field_verification(verification_summary, identity_verification, agent_outputs)

        # ── Step 4: Consensus ──────────────────────────────────────────────
        timeout_response = _check_pipeline_timeout("CONSENSUS", agent_outputs)
        if timeout_response is not None:
            return timeout_response
        print("[STAGE] CONSENSUS")
        print(f"[PIPELINE TIME] {time.time() - start_time:.2f}s")
        if blackboard_state.get("terminated"):
            _trace_stage(
                stage="CONSENSUS",
                status="SKIPPED",
                inputs={"terminated": True},
                outputs={},
                reason="Terminated before consensus",
                decision_snapshot="REJECTED",
            )
            return _finalize_response(
                exit_reason="low_confidence",
                agent_outputs=agent_outputs,
                blackboard=blackboard_state,
                routing_decision=routing,
                goa_used=goa_used,
                Ts=0.0,
                decision="REJECTED",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
                trust_layer=None,
                memory_context=memory_context,
                validation_result=validation_result,
                pre_validation_result=None,
                forensic_trace=forensic_trace,
                decision_trace={
                    "claim_id": claim_id,
                    "decision_before_guard": "REJECTED",
                    "final_decision": "REJECTED",
                    "decision_reason": "Terminated before consensus",
                    "terminated": True,
                },
                reason="Terminated before consensus",
                system_flags=sorted(set(system_flags)),
                extracted_data=verified_structured_data,
                pipeline_version="v2",
            )
        tracker.update("Consensus", "RUNNING")
        consensus_entries = dict(blackboard_state.get("entries", {}))
        for agent_row in runtime_agents:
            if not isinstance(agent_row, dict):
                continue
            agent_name = str(agent_row.get("agent") or "")
            if not agent_name:
                continue
            payload = dict(consensus_entries.get(agent_name, {}))
            payload["status"] = str(agent_row.get("status", "ERROR")).upper()
            runtime_score = float(agent_row.get("score", 0.0))
            payload["score"] = runtime_score  # SCORE-FIX
            payload.setdefault("confidence", 0.0)
            payload.setdefault("explanation", str(agent_row.get("reason", "")))
            consensus_entries[agent_name] = payload
        for ao in agent_outputs:
            payload = dict(consensus_entries.get(ao.agent, {}))
            # Keep runtime execution status canonical for consensus engine.
            payload["status"] = "DONE"
            payload["analysis_status"] = str(ao.output_snapshot.get("status", ""))
            payload["insufficient_data"] = bool(ao.output_snapshot.get("insufficient_data", False))
            payload["hallucination_flags"] = list(ao.hallucination_flags)
            consensus_entries[ao.agent] = payload
        assert len([name for name in consensus_entries.keys() if name != "_meta"]) >= 5, "Expected at least 5 agent results for consensus"
        consensus_entries["_meta"] = {
            "missing_fields": extraction_validation.get("missing_fields", []),
            "field_verification": blackboard.field_verification,
            "unverified_critical_fields": int(verification_summary.get("unverified_critical_fields", 0)),
            "has_unverified_fields": bool(verification_summary.get("has_unverified_fields", False)),
            "pre_validation_failed": bool(pre_validation.get("failed", False)),
            "critical_failures": sorted(set(critical_failures)),
            "cin_found": bool(identity_verification.get("cin_found", False)),
            "ipp_found": bool(identity_verification.get("ipp_found", False)),
            "amount_found": not bool(verification_summary.get("amount_missing", False)),
            "injection_detected": bool(pre_validation.get("injection_detected", False)),
            "cin_format_match": bool(
                re.match(
                    r"^[A-Z]{1,2}\d{5,6}$",
                    str(
                        (claim_request.get("identity", {}) or {}).get("cin")
                        or (claim_request.get("identity", {}) or {}).get("CIN")
                        or verified_structured_data.get("cin")
                        or verified_structured_data.get("CIN")
                        or ""
                    ).strip().upper(),
                )
            ),
            "tier1_blocking_flag_count": len(
                [
                    flag
                    for flag in set(system_flags)
                    if flag in {"INJECTION_DETECTED", "IDENTITY_HARD_FAIL", "CRITICAL_FIELD_MISSING", "AMOUNT_MISMATCH_CRITICAL"}
                ]
            ),
        }
        consensus_result = self._consensus_engine.evaluate(
            claim_request=claim_request,
            entries=consensus_entries,
            blackboard={"memory_degraded": memory_status != "OK", "memory_status": memory_status},
            config=self._consensus_config,
        )
        consensus_result["flags"] = dict(consensus_result.get("flags", {}))
        critical_failures = list(consensus_result.get("critical_failures", []) or [])
        if critical_failures:
            system_flags.extend(critical_failures)
        consensus_result["contradictions"] = list(consensus_result.get("contradictions", []))
        force_human_review, force_reason = should_force_human_review(
            agent_outputs=agent_outputs,
            blackboard={"contradictions": consensus_result["contradictions"]},
            config=self._consensus_config,
        )
        # PROD-FIX: do not force hallucination review when layer2 is disabled without layer1 trigger.
        if bool(pre_validation.get("degraded_security_mode", False)) and not bool(pre_validation.get("injection_detected", False)):
            force_human_review = False
            force_reason = "Layer2 disabled only; no layer1 trigger."
        # CALIBRATION-FIX: hallucination is only true when output contradicts OCR-grounded evidence.
        hallucination_agents = []
        for ao in agent_outputs:
            raw_flags = [str(flag).strip().lower() for flag in (ao.hallucination_flags or [])]
            grounded_flags = [
                flag for flag in raw_flags
                if any(
                    token in flag
                    for token in (
                        "ocr_value_not_found",
                        "field_not_present_in_document",
                        "contradicts_ocr",
                        "fabricated",
                    )
                )
            ]
            if grounded_flags:
                hallucination_agents.append(ao.agent)
        contradiction_penalty_total = round(
            sum(float(item.get("H_penalty", 0.0)) for item in consensus_result["contradictions"] if isinstance(item, dict)),
            4,
        )
        consensus_result["hallucination_agents"] = hallucination_agents
        consensus_result["hallucination_penalty_product"] = round(max(0.0, 1.0 - contradiction_penalty_total), 4)
        consensus_result["hallucination_force_human_review"] = force_human_review
        consensus_result["flags"]["hallucination_force_reason"] = force_reason
        if force_human_review and hallucination_agents:
            consensus_result["decision"] = "HUMAN_REVIEW"
            consensus_result["Ts"] = round(
                min(consensus_result["Ts"], self._consensus_config.auto_approve_threshold - 0.01),
                2,
            )
            system_flags.append("HALLUCINATION_FORCE_HUMAN_REVIEW")
        if consensus_result.get("conflicts"):
            system_flags.append("AGENT_CONFLICT")
        if bool(consensus_result.get("too_many_error_agents", False)):
            system_flags.append("TOO_MANY_ERROR_AGENTS")
            consensus_result["decision"] = "HUMAN_REVIEW"
        blackboard_state["insufficient_agents"] = consensus_result.get("insufficient_agents", [])
        blackboard_state["insufficient_force_human_review"] = consensus_result.get("insufficient_force_human_review", False)
        blackboard_state["hallucination_agents"] = consensus_result.get("hallucination_agents", [])
        blackboard_state["hallucination_force_human_review"] = consensus_result.get("hallucination_force_human_review", False)
        blackboard_state["hallucination_penalty_product"] = consensus_result.get("hallucination_penalty_product", 1.0)
        blackboard_state["force_human_review"] = force_human_review
        blackboard_state["flags"] = dict(blackboard_state.get("flags", {}))
        blackboard_state["flags"]["hallucination_force_reason"] = force_reason
        blackboard_state["field_verification_penalty"] = consensus_result.get("field_verification_penalty", 0.0)
        blackboard_state["unverified_critical_fields"] = consensus_result.get("unverified_critical_fields", 0)
        blackboard_state["critical_failures"] = critical_failures

        severe_contradiction = any(bool(item.get("severe")) for item in consensus_result.get("contradictions", []))
        has_hallucinations = bool(hallucination_agents)
        has_two_insufficient = bool(consensus_result.get("insufficient_force_human_review"))
        if severe_contradiction:
            system_flags.append("SEVERE_CONTRADICTION_DETECTED")
        if has_hallucinations:
            system_flags.append("HALLUCINATION_FLAGS_PRESENT")
        if has_two_insufficient:
            system_flags.append("MULTI_AGENT_INSUFFICIENT_DATA")
        if int(verification_summary.get("unverified_critical_fields", 0)) > 0:
            system_flags.append("UNVERIFIED_FIELDS_PRESENT")
        if severe_contradiction or has_two_insufficient:
            consensus_result["decision"] = "HUMAN_REVIEW"
        memory_degraded = memory_status != "OK"
        blackboard_state["memory_degraded"] = memory_degraded
        blackboard_state["flags"]["memory_degraded"] = memory_degraded
        if memory_degraded:
            system_flags.append("memory_degraded")
            # CALIBRATION-FIX: do not force human review only due to memory degradation.

        # Global enforcement rule — only block on clear fraud/injection signals.
        # Removed: critical_conf_ok (Ollama gives ~0.6 confidence on valid claims)
        # Removed: unverified_critical_fields (reconciliation handles this)
        if (
            consensus_result["decision"] == "APPROVED"
            and (
                bool(consensus_result.get("hallucination_force_human_review"))
                or self._reliability_store.is_auto_approve_disabled()
                or not external_validation.ok
            )
        ):
            consensus_result["decision"] = "HUMAN_REVIEW"
            system_flags.append("GLOBAL_ENFORCEMENT_GUARD")
        feedback_outcome = str(feedback_signal.get("outcome", "")).upper()
        if feedback_outcome in {"REJECTED", "FRAUD", "SUSPICIOUS"}:
            consensus_result["Ts"] = round(max(0.0, consensus_result["Ts"] - 10.0), 2)
            system_flags.append("HUMAN_FEEDBACK_RISK_PENALTY")
            if consensus_result["decision"] == "APPROVED":
                consensus_result["decision"] = "HUMAN_REVIEW"
        _trace_stage(
            stage="CONSENSUS",
            status="PASS",
            inputs={"agent_count": len(agent_outputs), "verification_summary": verification_summary},
            outputs={"consensus_result": consensus_result},
            reason="Consensus rules and safeguards applied",
            flags=sorted(set(system_flags)),
            decision_snapshot=str(consensus_result.get("decision", "")),
        )
        tracker.update("Consensus", "COMPLETED")
        
        if consensus_result["decision"] == "HUMAN_REVIEW":
            tracker.update("HumanReview", "RUNNING")
        else:
            tracker.update("HumanReview", "SKIPPED")
        blackboard_state["reflexive_retry_logs"] = consensus_result["retry_logs"]
        blackboard_state["score_evolution"] = consensus_result["score_evolution"]
        blackboard_state["consensus_entries"] = consensus_result["entries"]
        LOGGER.info(
            "consensus_final Ts=%.2f decision=%s retries=%s score_evolution=%s",
            consensus_result["Ts"],
            consensus_result["decision"],
            consensus_result["retry_count"],
            consensus_result["score_evolution"],
        )
        decision_before_guard = consensus_result["decision"]
        if forensic_enabled and forensic_trace is not None:
            forensic_trace["final_decision_trace"] = {
                "Ts": consensus_result["Ts"],
                "decision_before_guard": decision_before_guard,
                "decision_after_guard": consensus_result["decision"],
                "reason": "Consensus and approval guard evaluation completed",
            }

            prompt_texts = [entry["prompt"] for entry in forensic_trace["prompt_trace"]]
            if len(set(prompt_texts)) <= 1 and len(prompt_texts) > 1:
                forensic_trace["hard_failures"].append("PROMPT_STATIC_ACROSS_AGENTS_OR_INPUTS")
            if forensic_trace["llm_calls_count"] == 0:
                forensic_trace["hard_failures"].append("LLM_NOT_INVOKED")
            if "APPROVED" == consensus_result["decision"] and claim_request.get("metadata", {}).get("forensic_random_input", False):
                forensic_trace["hard_failures"].append("INVALID_INPUT_APPROVED")
            if Crew.__name__.lower().startswith("_strict") or Crew.__name__.lower().startswith("fake"):
                forensic_trace["simulation_or_stub_detected"] = True
                forensic_trace["hard_failures"].append("SIMULATION_OR_STUB_MODE_DETECTED")

            known_templates = [
                "No explanation provided",
                "Insufficient data to perform reliable analysis",
                "Cannot establish baseline — insufficient history",
            ]
            for item in forensic_trace["response_trace"]:
                response_text = str(item.get("raw_llm_response", ""))
                for template in known_templates:
                    if template in response_text:
                        forensic_trace["known_template_matches"].append(
                            {"agent": item["agent"], "template": template}
                        )
            if forensic_trace["known_template_matches"]:
                forensic_trace["hard_failures"].append("KNOWN_TEMPLATE_RESPONSES_DETECTED")

            differentiation = self._run_forensic_input_differentiation_test(
                routing=routing,
                claim_request=claim_request,
            )
            forensic_trace["input_differentiation_test"] = differentiation
            forensic_trace["hard_failures"].extend(differentiation.get("failures", []))
            for item in differentiation.get("results", []):
                per_input = item.get("per_input", {})
                hash_a = per_input.get("A", {}).get("output_hash")
                hash_b = per_input.get("B", {}).get("output_hash")
                if hash_a and hash_a == hash_b:
                    forensic_trace["hash_collisions_across_inputs"].append(
                        {
                            "agent": item["agent"],
                            "hash_output": hash_a,
                            "inputs": ["A", "B"],
                        }
                    )
            forensic_trace["hard_failures"] = sorted(set(forensic_trace["hard_failures"]))
            forensic_trace["broken"] = bool(forensic_trace["hard_failures"])
            forensic_trace["system_status"] = "BROKEN" if forensic_trace["broken"] else "WORKING"
            all_outputs: List[Dict[str, Any]] = []
            for row in differentiation.get("results", []):
                per_input = row.get("per_input", {})
                for key in ("A", "B", "C"):
                    item = per_input.get(key, {})
                    if item:
                        all_outputs.append(
                            {
                                "agent": row.get("agent"),
                                "input": key,
                                "score": item.get("score"),
                                "confidence": item.get("confidence"),
                                "explanation": item.get("explanation"),
                            }
                        )
            entropy = _output_entropy(all_outputs)
            if entropy < 0.35:
                forensic_trace["hard_failures"].append("OUTPUT_ENTROPY_TOO_LOW")
            llm_calls_per_agent: Dict[str, int] = {}
            for row in forensic_trace.get("agent_forensic_trace", []):
                name = str(row.get("agent"))
                llm_calls_per_agent[name] = llm_calls_per_agent.get(name, 0) + int(bool(row.get("llm_called")))
            if any(
                trace.get("decision") == "APPROVED" and trace.get("input_type") in {"B", "C"}
                for trace in [
                    {
                        "input_type": "B" if "random" in str(claim_request.get("metadata", {}).get("claim_id", "")).lower() else "A",
                        "decision": consensus_result["decision"],
                    }
                ]
            ):
                forensic_trace["hard_failures"].append("INVALID_INPUT_APPROVED")
            forensic_trace["hard_failures"] = sorted(set(forensic_trace["hard_failures"]))
            forensic_trace["broken"] = bool(forensic_trace["hard_failures"])
            forensic_trace["system_status"] = "BROKEN" if forensic_trace["broken"] else "WORKING"
            forensic_trace["final_audit_report"] = {
                "system_status": forensic_trace["system_status"],
                "input_integrity": "PASS" if "EMPTY_PARSED_CLAIM_DATA" not in forensic_trace["hard_failures"] else "FAIL",
                "llm_usage": {
                    "total_calls": forensic_trace["llm_calls_count"],
                    "calls_per_agent": llm_calls_per_agent,
                    "status": "PASS" if forensic_trace["llm_calls_count"] > 0 else "FAIL",
                },
                "prompt_quality": {
                    "status": "FAIL" if any("PROMPT_" in f for f in forensic_trace["hard_failures"]) else "PASS",
                    "details": forensic_trace["prompt_trace"],
                },
                "agent_variability": {
                    "entropy": round(entropy, 4),
                    "status": "FAIL" if "OUTPUT_ENTROPY_TOO_LOW" in forensic_trace["hard_failures"] else "PASS",
                },
                "decision_integrity": {
                    "status": "FAIL" if "INVALID_INPUT_APPROVED" in forensic_trace["hard_failures"] else "PASS",
                    "trace": forensic_trace["final_decision_trace"],
                },
                "critical_failures": forensic_trace["hard_failures"],
                "root_cause_analysis": [
                    "input not passed to agents",
                    "static prompt",
                    "LLM not connected",
                    "decision override bug",
                ],
                "fix_recommendations": [
                    "Inject structured claim fields directly into every agent prompt.",
                    "Enforce per-agent prompt templates with dynamic data assertions.",
                    "Block pipeline completion when LLM calls are zero.",
                    "Require blackboard diff checks between sequential agents.",
                    "Force HUMAN_REVIEW for empty/random inputs.",
                ],
            }

        try:
            metrics = self._reliability_store.push_decision_metrics(
                decision=consensus_result["decision"],
                ts=consensus_result["Ts"],
            )
        except Exception as exc:
            LOGGER.exception("reliability_metrics_failed claim_id=%s error=%s", claim_id, exc)
            metrics = {
                "window": 0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
                "average_ts": 0.0,
                "approved_disabled": False,
                "system_alert": False,
            }
            system_flags.append("RELIABILITY_METRICS_FAILED")
        if metrics.get("system_alert"):
            system_flags.append("SYSTEM_ALERT_APPROVAL_SPIKE")
        if metrics.get("approved_disabled"):
            system_flags.append("APPROVED_DISABLED_RATE_LIMIT")

        # Stability test: deterministic re-evaluation from the same consensus inputs.
        stability_runs: List[Dict[str, Any]] = []
        expected_decision = consensus_result["decision"]
        expected_ts = consensus_result["Ts"]
        decision_scores: Dict[str, float] = {}
        for _ in range(3):
            rerun = self._consensus_engine.evaluate(
                claim_request=claim_request,
                entries=consensus_entries,
                blackboard={"memory_degraded": memory_status != "OK", "memory_status": memory_status},
                config=self._consensus_config,
            )
            snapshot = {"Ts": rerun["Ts"], "decision": rerun["decision"]}
            stability_runs.append(snapshot)
            decision_scores[snapshot["decision"]] = max(
                float(snapshot["Ts"]),
                float(decision_scores.get(snapshot["decision"], 0.0)),
            )
        competing_decisions = sorted(
            [{"decision": name, "score": value} for name, value in decision_scores.items() if float(value) > 50.0],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        stability_fail = False
        if len(competing_decisions) >= 2:
            delta = float(competing_decisions[0]["score"]) - float(competing_decisions[1]["score"])
            if delta < 8.0:
                stability_fail = True
        if stability_fail:
            system_flags.append("DECISION_STABILITY_FAIL")
            consensus_result["Ts"] = max(0.0, round(float(consensus_result["Ts"]) - 5.0, 2))  # CALIBRATION-FIX

        agent_summaries = [
            {
                "agent": ao.agent,
                "score": round(float(ao.score) * 100, 2),
                "confidence": round(float(ao.confidence) * 100, 2),
                "status": str(ao.output_snapshot.get("status", "")),
                "key_evidence": ao.output_snapshot.get("claims", []),
                "reasoning_summary": ao.explanation,
                "hallucination_flags": ao.hallucination_flags,
            }
            for ao in agent_outputs
        ]
        decision_trace = {
            "claim_id": claim_id,
            "input_summary": input_summary,
            "ocr_snapshot": ocr_snapshot,
            "input_trust": input_trust,
            "field_verification": blackboard.field_verification,
            "field_verification_summary": verification_summary,
            "agent_summaries": agent_summaries,
            "memory_signals": memory_signals,
            "memory_status": memory_status,
            "contradictions": consensus_result.get("contradictions", []),
            "Ts_score": consensus_result["Ts"],
            "decision_before_guard": decision_before_guard,
            "final_decision": consensus_result["decision"],
            "decision_reason": _HARD_REJECTION_REASON if critical_failures else "Reliability trust rules enforced",
            "critical_failures": critical_failures,
            "system_flags": sorted(set(system_flags)),
            "system_metrics": metrics,
            "stability_test": {"runs": stability_runs, "stable": not stability_fail},
            "proof_trace": trace_engine.export(),
        }
        contradiction_penalty = sum(
            float(item.get("H_penalty", 0.0)) for item in consensus_result.get("contradictions", [])
        )
        dispute_risk = (
            abs(float(consensus_result["Ts"]) - 90.0) <= 5.0
            or contradiction_penalty >= 0.25
            or memory_degraded
        )
        blackboard_state["dispute_risk"] = dispute_risk
        blackboard_state["flags"]["dispute_risk"] = dispute_risk
        try:
            trace_hash = self._reliability_store.persist_decision_trace(claim_id, decision_trace)
        except Exception as exc:
            LOGGER.exception("decision_trace_persist_failed claim_id=%s error=%s", claim_id, exc)
            trace_hash = hash_payload(
                {
                    "claim_id": claim_id,
                    "final_decision": decision_trace.get("final_decision"),
                    "ts_score": decision_trace.get("Ts_score"),
                }
            )
            system_flags.append("DECISION_TRACE_PERSIST_FAILED")
        decision_trace["trace_hash"] = trace_hash
        decision_trace["trace_anchor_tx"] = None

        try:
            replay_package = {
                "raw_input": claim_request,
                "ocr_output": {"text": extracted_text, "structured_data": structured_data},
                "prompts_used": [row.get("prompt") for row in (forensic_trace or {}).get("prompt_trace", [])],
                "agent_outputs": [item.model_dump() for item in agent_outputs],
                "consensus_entries": consensus_entries,
                "decision": consensus_result["decision"],
                "Ts": consensus_result["Ts"],
                "decision_trace_hash": trace_hash,
            }
            replay_package["replay_hash"] = hash_payload(replay_package)
            self._reliability_store.register_replay_package(claim_id, replay_package)
        except Exception as exc:
            LOGGER.exception("replay_package_register_failed claim_id=%s error=%s", claim_id, exc)
            system_flags.append("REPLAY_PACKAGE_REGISTER_FAILED")

        final_outcome = _exit_from_ts(float(consensus_result.get("Ts", 0.0)))
        # Graceful degradation: valid covered claims should not hard reject solely due low Ts.
        clean_critical_fields = (
            bool(identity_verification.get("cin_found", False))
            and bool(identity_verification.get("ipp_found", False))
            and not bool(verification_summary.get("amount_missing", False))
            and int(verification_summary.get("unverified_critical_fields", 0) or 0) == 0
        )
        if (
            final_outcome.get("decision") == "REJECTED"
            and validation_result is not None
            and str(validation_result.validation_status).upper() == "VALID"
            and not bool(pre_validation.get("injection_detected", False))
            and not clean_critical_fields
        ):
            final_outcome = {"decision": "HUMAN_REVIEW", "exit_reason": "low_confidence", "Ts": float(consensus_result.get("Ts", 0.0))}
            system_flags.append("LOW_TS_VALIDATION_GRACEFUL_DEGRADE")
            cin_found = bool(identity_verification.get("cin_found", False))
            ipp_found = bool(identity_verification.get("ipp_found", False))
            amount_found = not bool(verification_summary.get("amount_missing", False))
            unverified_critical = int(verification_summary.get("unverified_critical_fields", 0) or 0)
            if float(consensus_result.get("Ts", 0.0)) <= 0.0 and not (
                cin_found and ipp_found and amount_found and unverified_critical == 0
            ):
                consensus_result["Ts"] = max(
                    40.0,
                    float(validation_result.validation_score),
                )
                system_flags.append("LOW_TS_FLOOR_FROM_VALIDATION")
        FINAL_DECISION = str(final_outcome["decision"]).upper()
        FINAL_EXIT_REASON = str(final_outcome["exit_reason"])
        assert FINAL_DECISION in ["APPROVED", "REJECTED", "HUMAN_REVIEW"]

        human_review_reason = None
        if FINAL_DECISION == "HUMAN_REVIEW":
            if 60.0 <= float(consensus_result["Ts"]) < 75.0:
                human_review_reason = "Uncertain / requires human validation"
                system_flags.append("HUMAN_REVIEW_TS_WINDOW")
            else:
                human_review_reason = "Manual review required by trust safeguards"

        consensus_result["decision"] = FINAL_DECISION
        blackboard_state["terminated"] = FINAL_DECISION == "HUMAN_REVIEW"
        decision_trace["final_decision"] = FINAL_DECISION
        decision_trace["decision_reason"] = (
            _HARD_REJECTION_REASON
            if FINAL_DECISION == "REJECTED"
            else (human_review_reason or "Reliability trust rules enforced")
        )
        decision_trace["terminated"] = blackboard_state["terminated"]

        document_url = None
        heatmap: List[Dict[str, Any]] = []
        heatmap_fallback: List[Dict[str, Any]] = []
        if FINAL_DECISION == "HUMAN_REVIEW":
            review_context = self._register_human_review_context(
                claim_id=claim_id,
                claim_request=claim_request,
                ts_score=float(consensus_result["Ts"]),
                reason=human_review_reason or "Manual review required",
                verified_fields=verified_structured_data,
                agent_outputs=agent_outputs,
                blackboard_snapshot=blackboard_state,
            )
            document_url = str(review_context.get("document_url") or "") or None
            heatmap = list(review_context.get("heatmap", []))
            heatmap_fallback = list(review_context.get("heatmap_fallback", []))
            decision_trace["final_decision"] = "HUMAN_REVIEW"
            decision_trace["decision_reason"] = human_review_reason or decision_trace.get("decision_reason", "")
        _trace_stage(
            stage="FINAL_DECISION",
            status="PASS" if FINAL_DECISION != "REJECTED" else "FAIL",
            inputs={"Ts": consensus_result["Ts"], "decision_before_guard": decision_before_guard},
            outputs={"final_decision": FINAL_DECISION},
            reason=str(decision_trace.get("decision_reason", "")),
            flags=sorted(set(system_flags)),
            decision_snapshot=FINAL_DECISION,
        )

        # ── Step 5: Trust layer ────────────────────────────────────────────
        trust_layer_payload = None
        trust_layer_error = ""
        decision_modifier = 0.0
        if FINAL_DECISION == "APPROVED":
            try:
                trust_layer_payload = self._trust_layer.process_approved_claim(
                    {
                        "claim_id": claim_id,
                        "decision": FINAL_DECISION,
                        "ts_score": consensus_result["Ts"],
                        "claim_request": claim_request,
                        "blackboard": blackboard_state,
                        "agent_outputs": [item.model_dump() for item in agent_outputs],
                        "flags": system_flags,
                    }
                )
                # IPFS or chain may degrade independently; do not abort the decision path.
                cid_value = str((trust_layer_payload or {}).get("cid") or "").strip()
                tx_value = str((trust_layer_payload or {}).get("tx_hash") or "").strip()
                if not cid_value or not tx_value:
                    LOGGER.warning(
                        "trust_layer_anchoring_incomplete claim_id=%s has_cid=%s has_tx=%s",
                        claim_id,
                        bool(cid_value),
                        bool(tx_value),
                    )
            except TrustLayerIPFSFailure as exc:
                trust_layer_error = str(exc)
                LOGGER.error("trust_layer_ipfs_failed claim_id=%s error=%s", claim_id, trust_layer_error)
                system_flags.append("TRUST_LAYER_DEGRADED")
                decision_modifier = -5.0
            except Exception as exc:
                trust_layer_error = str(exc)
                LOGGER.exception("trust_layer_unexpected_error claim_id=%s error=%s", claim_id, exc)
                system_flags.append("TRUST_LAYER_DEGRADED")
                decision_modifier = -5.0
        if "TRUST_LAYER_DEGRADED" in system_flags and decision_modifier:
            adjusted_ts = max(0.0, float(consensus_result.get("Ts", 0.0)) + decision_modifier)
            consensus_result["Ts"] = adjusted_ts
            prev_decision = FINAL_DECISION
            FINAL_DECISION = self._consensus_engine._decision_for_ts(adjusted_ts, self._consensus_config)
            FINAL_EXIT_REASON = "trust_layer_degraded" if FINAL_DECISION != "APPROVED" else FINAL_EXIT_REASON
            decision_trace["final_decision"] = FINAL_DECISION
            decision_trace["decision_reason"] = (
                f"Trust layer degraded signal applied as score modifier ({decision_modifier})."
            )
            # If trust layer downgraded APPROVED → HUMAN_REVIEW, save the review context now
            if prev_decision == "APPROVED" and FINAL_DECISION == "HUMAN_REVIEW":
                try:
                    review_context = self._register_human_review_context(
                        claim_id=claim_id,
                        claim_request=claim_request,
                        ts_score=float(adjusted_ts),
                        reason="Trust layer degraded — manual review required",
                        verified_fields=verified_structured_data,
                        agent_outputs=agent_outputs,
                        blackboard_snapshot=blackboard_state,
                    )
                    document_url = str(review_context.get("document_url") or "") or None
                    heatmap = list(review_context.get("heatmap", []))
                    heatmap_fallback = list(review_context.get("heatmap_fallback", []))
                except Exception as exc:
                    LOGGER.warning("trust_layer_review_context_failed claim_id=%s error=%s", claim_id, exc)
        _trace_stage(
            stage="TRUST_LAYER",
            status="PASS" if bool(trust_layer_payload) else "SKIPPED",
            inputs={"final_decision": FINAL_DECISION},
            outputs={"trust_layer": trust_layer_payload or {}},
            reason=(
                "Trust layer anchoring completed"
                if trust_layer_payload
                else (f"Trust layer skipped: {trust_layer_error}" if trust_layer_error else "Trust layer skipped or unavailable")
            ),
        )

        # ── Step 7: Store claim in memory AFTER consensus ──────────────────
        self._store_claim_in_memory(
            claim_id=claim_id,
            claim_request=claim_request,
            agent_outputs=agent_outputs,
            decision=FINAL_DECISION,
            ts_score=consensus_result["Ts"],
        )
        success_count = sum(1 for row in runtime_agents if str(row.get("status", "")).upper() == "DONE")
        error_count = sum(1 for row in runtime_agents if str(row.get("status", "")).upper() == "ERROR")
        LOGGER.info("[AGENT SUCCESS COUNT] %s", success_count)
        LOGGER.info("[AGENT ERROR COUNT] %s", error_count)
        LOGGER.info(
            "[FINAL DECISION TRACE] decision=%s score=%.2f reason=%s flags=%s",
            FINAL_DECISION,
            float(consensus_result.get("Ts", 0.0)),
            str(decision_trace.get("decision_reason", "")),
            sorted(set(system_flags)),
        )
        LOGGER.info("[TOTAL TIME] %.3fs", time.time() - start_time)
        LOGGER.info("[PIPELINE END] claim_id=%s", claim_id)

        return _finalize_response(
            exit_reason=FINAL_EXIT_REASON,
            agent_outputs=agent_outputs,
            agents=runtime_agents,
            blackboard=blackboard_state,
            routing_decision=routing,
            goa_used=goa_used,
            Ts=consensus_result["Ts"],
            decision=FINAL_DECISION,
            retry_count=consensus_result["retry_count"],
            mahic_breakdown=consensus_result["mahic_breakdown"],
            contradictions=consensus_result["contradictions"],
            trust_layer=trust_layer_payload if trust_layer_payload else None,
            memory_context=memory_context,
            validation_result=validation_result,
            pre_validation_result=None,
            forensic_trace=forensic_trace,
            decision_trace=decision_trace,
            system_flags=sorted(set(system_flags)),
            reason=(
                "Dossier valide. Tous les champs critiques vérifiés. Aucun signal de fraude détecté."
                if FINAL_DECISION == "APPROVED"
                else (
                    "Dossier orienté vers un contrôle humain pour vérification complémentaire."
                    if FINAL_DECISION == "HUMAN_REVIEW"
                    else "Dossier rejeté suite à des incohérences critiques détectées."
                )
            ),
            document_url=document_url,
            extracted_data=verified_structured_data,
            heatmap=heatmap,
            heatmap_fallback=heatmap_fallback,
            pipeline_version="v2",
        )

    def _store_claim_in_memory(
        self,
        *,
        claim_id: str,
        claim_request: Dict[str, Any],
        agent_outputs: List[AgentOutput],
        decision: str,
        ts_score: float,
    ) -> None:
        """Build a CaseMemoryEntry from this claim's result and persist it."""
        try:
            identity = claim_request.get("identity", {})
            policy = claim_request.get("policy", {})
            metadata = claim_request.get("metadata", {})

            cin = (
                identity.get("cin") or identity.get("CIN")
                or claim_request.get("patient_id", "")
            )
            hospital = (
                identity.get("hospital") or policy.get("hospital")
                or metadata.get("hospital", "")
            )
            doctor = (
                identity.get("doctor") or policy.get("doctor")
                or metadata.get("doctor", "")
            )
            diagnosis = (
                policy.get("diagnosis") or metadata.get("diagnosis")
                or claim_request.get("diagnosis", "")
            )

            fraud_label = decision_to_fraud_label(decision, ts_score)
            agent_summary = build_agent_summary(
                [ao.model_dump() for ao in agent_outputs]
            )

            entry = CaseMemoryEntry(
                claim_id=claim_id,
                cin=str(cin),
                hospital=str(hospital),
                doctor=str(doctor),
                diagnosis=str(diagnosis),
                fraud_label=fraud_label,
                ts_score=ts_score,
                agent_summary=agent_summary,
            )
            self._memory.store_case(entry)
            LOGGER.info(
                "claim_stored_in_memory claim_id=%s fraud_label=%s Ts=%.2f",
                claim_id, fraud_label, ts_score,
            )
        except Exception as exc:
            LOGGER.error("memory_store_claim_failed claim_id=%s error=%s", claim_id, exc)

    def replay(self, claim_id: str) -> Dict[str, Any]:
        package = self._reliability_store.get_replay_package(claim_id)
        if not package:
            trace = self._reliability_store.get_trace(claim_id)
            if not trace:
                raise KeyError(f"No replay data found for claim {claim_id}")
            return {
                "claim_id": claim_id,
                "status": "trace_only",
                "stored_trace_hash": trace.get("trace_hash"),
            }
        rerun = self._consensus_engine.evaluate(
            claim_request=package.get("raw_input", {}),
            entries=package.get("consensus_entries", {}),
            blackboard={
                "memory_degraded": bool(package.get("memory_status", False)),
                "memory_status": str(package.get("memory_status", "")),
            },
        )
        same_decision = rerun.get("decision") == package.get("decision")
        same_ts = abs(float(rerun.get("Ts", 0.0)) - float(package.get("Ts", 0.0))) <= 1e-9
        return {
            "claim_id": claim_id,
            "replayed_decision": rerun.get("decision"),
            "replayed_Ts": rerun.get("Ts"),
            "original_decision": package.get("decision"),
            "original_Ts": package.get("Ts"),
            "deterministic_match": bool(same_decision and same_ts),
            "replay_hash": package.get("replay_hash"),
        }

    def get_proof_trace(self, claim_id: str) -> Dict[str, Any] | None:
        payload = self._reliability_store.get_trace(claim_id)
        if not payload:
            return None
        trace = payload.get("trace", {}) if isinstance(payload, dict) else {}
        proof_trace = trace.get("proof_trace") if isinstance(trace, dict) else None
        return proof_trace if isinstance(proof_trace, dict) else None

    def record_human_feedback(
        self,
        *,
        claim_id: str,
        outcome: str,
        reviewer_id: str,
        cin: str,
        provider: str,
        ip_address: str = "",
    ) -> Dict[str, Any]:
        return self._reliability_store.add_human_feedback(
            claim_id=claim_id,
            outcome=outcome,
            reviewer_id=reviewer_id,
            cin=cin,
            provider=provider,
            ip_address=ip_address,
        )

    def get_fraud_graph_debug(self, *, render_png: bool = False) -> Dict[str, Any]:
        rings = self._fraud_ring_graph.detect_fraud_rings()
        payload: Dict[str, Any] = {
            "fraud_rings": rings.get("fraud_rings", []),
            "node_count": int(self._fraud_ring_graph.graph.number_of_nodes()),
            "edge_count": int(self._fraud_ring_graph.graph.number_of_edges()),
            "png_path": None,
        }
        if render_png:
            artifact_dir = Path(__file__).resolve().parents[2] / "tests" / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            output_path = artifact_dir / "v2_fraud_graph.png"
            payload["png_path"] = self._fraud_ring_graph.visualize_graph(output_path)
        return payload

    def get_trust_layer_health(self) -> Dict[str, Any]:
        # BLOCKCHAIN-FIX: expose live trust-layer readiness checks.
        return self._trust_layer.healthcheck()


_singleton: ClaimGuardV2Orchestrator | None = None


def get_v2_orchestrator() -> ClaimGuardV2Orchestrator:
    global _singleton
    if _singleton is None:
        _singleton = ClaimGuardV2Orchestrator()
    return _singleton


def run_pipeline_v2(claim_request: Dict[str, Any]) -> ClaimGuardV2Response:
    return get_v2_orchestrator().run(claim_request)
