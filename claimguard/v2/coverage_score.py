"""Document-coverage scoring used to replace brittle document_type enum checks.

The orchestrator previously hard-rejected claims when the classifier emitted a
label it had not been configured for. That gate has been replaced with a
coverage-score model: each claim bundle is scored on how well the OCR/structured
content covers the three canonical medical-claim components (medical report,
invoice, prescription) plus auxiliary signals, and the pipeline never rejects
purely on label mismatch.

The score is informational for downstream stages; the only decision rule is the
very permissive ``score < MIN_COVERAGE_ACCEPT`` threshold, which lets a claim
continue even with low coverage but marks it for degraded confidence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from pydantic import BaseModel, ConfigDict, Field


MIN_COVERAGE_ACCEPT: float = float(os.getenv("COVERAGE_MIN_ACCEPT", "0.4"))
LOW_COVERAGE_WARN: float = float(os.getenv("COVERAGE_LOW_WARN", "0.55"))

COVERAGE_WEIGHTS: Dict[str, float] = {
    "medical_report_coverage": 0.35,
    "invoice_coverage": 0.35,
    "prescription_coverage": 0.15,
    "entity_reference_coverage": 0.10,
    "monetary_signal_coverage": 0.05,
}


MEDICAL_REPORT_TOKENS: tuple[str, ...] = (
    "medical report", "medical_report", "rapport medical", "rapport médical",
    "compte rendu", "diagnostic", "diagnosis", "hospitalisation",
    "consultation", "traitement", "certificat medical", "certificat médical",
    "clinical report", "examen", "symptome", "symptôme", "pathologie",
)
INVOICE_TOKENS: tuple[str, ...] = (
    "invoice", "facture", "total ttc", "montant ttc", "tva",
    "reçu", "recu", "note d'honoraires", "remboursement", "montant",
    "total", "honoraires", "receipt",
)
PRESCRIPTION_TOKENS: tuple[str, ...] = (
    "prescription", "ordonnance", "medicament", "médicament",
    "posologie", "traitement prescrit", "dose", "posology", "rx",
)
ENTITY_TOKENS: tuple[str, ...] = (
    "patient", "cin", "ipp", "provider", "hospital", "clinic",
    "clinique", "hopital", "hôpital", "medecin", "médecin", "doctor", "dr",
)


class CoverageScore(BaseModel):
    """Structured coverage metrics (all values in [0, 1])."""

    model_config = ConfigDict(extra="forbid")

    medical_report_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    invoice_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    prescription_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    entity_reference_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    monetary_signal_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    overall: float = Field(ge=0.0, le=1.0, default=0.0)
    warnings: List[str] = Field(default_factory=list)
    classifier_bundle: str = Field(default="unknown_bundle")
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)


def _keyword_coverage(text: str, tokens: Iterable[str]) -> tuple[float, List[str]]:
    lowered = (text or "").lower()
    hits: List[str] = [tok for tok in tokens if tok in lowered]
    if not tokens:
        return 0.0, []
    saturation = min(1.0, len(hits) / 3.0)  # three distinct hits = saturated
    return saturation, hits


def _classifier_bundle(tool_output: Mapping[str, Any] | None) -> str:
    if not isinstance(tool_output, Mapping):
        return "unknown_bundle"
    output = tool_output.get("output") if isinstance(tool_output.get("output"), Mapping) else tool_output
    raw = str(output.get("document_type") or "").strip().lower()
    if raw:
        return raw
    found = [str(x).lower() for x in (output.get("found_docs") or [])]
    missing = [str(x).lower() for x in (output.get("missing_docs") or [])]
    if found and not missing:
        return "medical_claim_bundle"
    if found and missing:
        return "hybrid_bundle"
    return "unknown_bundle"


def _classifier_hits(tool_output: Mapping[str, Any] | None, key: str) -> bool:
    if not isinstance(tool_output, Mapping):
        return False
    output = tool_output.get("output") if isinstance(tool_output.get("output"), Mapping) else tool_output
    found = [str(x).lower() for x in (output.get("found_docs") or [])]
    return key in found


def compute_coverage_score(
    *,
    extracted_text: str,
    structured_data: Mapping[str, Any] | None = None,
    ml_classification: Mapping[str, Any] | None = None,
    document_classifier_tool: Mapping[str, Any] | None = None,
) -> CoverageScore:
    """Compute a coverage score from OCR text + classifier tool output.

    Label outputs from the document classifier (medical_claim_bundle,
    hybrid_bundle, unknown_bundle) are treated as *informational clustering*
    only — they boost/damp the score slightly but never gate it.
    """
    structured_data = dict(structured_data or {})
    warnings: List[str] = []

    med_cov, med_hits = _keyword_coverage(extracted_text, MEDICAL_REPORT_TOKENS)
    inv_cov, inv_hits = _keyword_coverage(extracted_text, INVOICE_TOKENS)
    rx_cov, rx_hits = _keyword_coverage(extracted_text, PRESCRIPTION_TOKENS)
    entity_cov, entity_hits = _keyword_coverage(extracted_text, ENTITY_TOKENS)

    if _classifier_hits(document_classifier_tool, "medical_report"):
        med_cov = max(med_cov, 0.7)
    if _classifier_hits(document_classifier_tool, "invoice"):
        inv_cov = max(inv_cov, 0.7)
    if _classifier_hits(document_classifier_tool, "prescription"):
        rx_cov = max(rx_cov, 0.7)

    for field_name in ("cin", "ipp", "name"):
        if str(structured_data.get(field_name) or "").strip():
            entity_cov = min(1.0, entity_cov + 0.15)

    monetary_cov = 0.0
    if str(structured_data.get("amount") or "").strip():
        monetary_cov = 0.8
    if any(sym in (extracted_text or "").lower() for sym in ("mad", "dh", "dhs", "dirham", "eur", "usd", "€")):
        monetary_cov = max(monetary_cov, 0.6)

    bundle = _classifier_bundle(document_classifier_tool)
    ml_label = str((ml_classification or {}).get("label", "")).upper()
    ml_confidence = float((ml_classification or {}).get("confidence", 0.0)) / 100.0

    overall = (
        COVERAGE_WEIGHTS["medical_report_coverage"] * med_cov
        + COVERAGE_WEIGHTS["invoice_coverage"] * inv_cov
        + COVERAGE_WEIGHTS["prescription_coverage"] * rx_cov
        + COVERAGE_WEIGHTS["entity_reference_coverage"] * entity_cov
        + COVERAGE_WEIGHTS["monetary_signal_coverage"] * monetary_cov
    )

    if ml_label == "CLAIM":
        overall = min(1.0, overall + 0.05 * ml_confidence)
    elif ml_label == "NON_CLAIM":
        warnings.append("ml_classifier_flagged_non_claim")
    if bundle == "unknown_bundle":
        warnings.append("classifier_bundle_unknown")
    if med_cov < 0.2 and inv_cov < 0.2 and rx_cov < 0.2:
        warnings.append("all_three_component_types_absent")
    if overall < MIN_COVERAGE_ACCEPT:
        warnings.append(f"coverage_below_accept_threshold_{MIN_COVERAGE_ACCEPT:.2f}")
    elif overall < LOW_COVERAGE_WARN:
        warnings.append(f"coverage_below_warn_threshold_{LOW_COVERAGE_WARN:.2f}")

    tool_outputs: Dict[str, Any] = {
        "document_classifier_tool": dict(document_classifier_tool or {}),
        "ml_classification": dict(ml_classification or {}),
        "keyword_hits": {
            "medical_report": med_hits,
            "invoice": inv_hits,
            "prescription": rx_hits,
            "entity": entity_hits,
        },
        "structured_signals": {
            "has_amount": bool(str(structured_data.get("amount") or "").strip()),
            "has_cin": bool(str(structured_data.get("cin") or "").strip()),
            "has_ipp": bool(str(structured_data.get("ipp") or "").strip()),
        },
    }

    return CoverageScore(
        medical_report_coverage=round(med_cov, 4),
        invoice_coverage=round(inv_cov, 4),
        prescription_coverage=round(rx_cov, 4),
        entity_reference_coverage=round(entity_cov, 4),
        monetary_signal_coverage=round(monetary_cov, 4),
        overall=round(overall, 4),
        warnings=warnings,
        classifier_bundle=bundle,
        tool_outputs=tool_outputs,
    )


@dataclass
class DecisionExplanation:
    """Mandatory structured explanation returned on every pipeline exit."""

    summary: str
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
            "tool_outputs": dict(self.tool_outputs),
        }


def build_explanation(
    *,
    decision: str,
    score: float,
    coverage: CoverageScore | None = None,
    summary: str = "",
    reasons: Iterable[str] | None = None,
    signals: Mapping[str, Any] | None = None,
    tool_outputs: Mapping[str, Any] | None = None,
    debug: bool | None = None,
) -> Dict[str, Any]:
    """Build the mandatory {decision, score, explanation} envelope.

    ``DEBUG_EXPLANATION_MODE=1`` (env) or ``debug=True`` forces full reasoning
    output with no truncation.
    """
    if debug is None:
        debug = os.getenv("DEBUG_EXPLANATION_MODE", "1").strip() not in ("", "0", "false", "False")

    reason_list = [str(r) for r in (reasons or []) if str(r).strip()]
    signal_dict: Dict[str, Any] = dict(signals or {})
    tool_dict: Dict[str, Any] = dict(tool_outputs or {})

    if coverage is not None:
        signal_dict.setdefault("coverage", {
            "medical_report": coverage.medical_report_coverage,
            "invoice": coverage.invoice_coverage,
            "prescription": coverage.prescription_coverage,
            "entity_reference": coverage.entity_reference_coverage,
            "monetary_signal": coverage.monetary_signal_coverage,
            "overall": coverage.overall,
            "classifier_bundle": coverage.classifier_bundle,
        })
        reason_list.extend(coverage.warnings)
        for key, value in (coverage.tool_outputs or {}).items():
            tool_dict.setdefault(key, value)

    if not summary:
        if coverage is not None:
            summary = (
                f"Decision={decision} score={score:.2f} "
                f"coverage={coverage.overall:.2f} bundle={coverage.classifier_bundle}"
            )
        else:
            summary = f"Decision={decision} score={score:.2f}"

    explanation = {
        "summary": summary,
        "reasons": sorted(set(reason_list)),
        "signals": signal_dict,
        "tool_outputs": tool_dict,
    }
    envelope: Dict[str, Any] = {
        "decision": str(decision or "").strip().upper() or "REJECTED",
        "score": float(score),
        "explanation": explanation,
    }
    envelope["debug_mode"] = bool(debug)
    return envelope


def coverage_decision(coverage: CoverageScore) -> str:
    """Convert a coverage score to ACCEPTED/REJECTED (pipeline-continuation).

    This is only used for the *coverage gate*; final APPROVED/HUMAN_REVIEW/
    REJECTED is still derived from the consensus Ts score downstream. A low
    coverage score produces ``REJECTED`` here (meaning "don't bother running
    consensus") while acceptable coverage produces ``ACCEPTED`` (pipeline
    continues, possibly with warnings).
    """
    return "ACCEPTED" if coverage.overall >= MIN_COVERAGE_ACCEPT else "REJECTED"
