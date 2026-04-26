"""Coverage-score model tests for the four mandated scenarios.

These replace the brittle document_type enum tests. The pipeline must:
    CASE 1: accept a medical bundle (never hard-reject on label mismatch)
    CASE 2: accept an invoice-only document with low-coverage warning
    CASE 3: accept a mixed-document bundle with enriched explanation
    CASE 4: reject invalid/empty OCR *with* a structured explanation
"""
from __future__ import annotations

from claimguard.v2.coverage_score import (
    MIN_COVERAGE_ACCEPT,
    build_explanation,
    compute_coverage_score,
    coverage_decision,
)


MEDICAL_BUNDLE_TEXT = (
    "Rapport médical — consultation chez Dr. Bennani\n"
    "Diagnostic: pathologie respiratoire aiguë\n"
    "Ordonnance: amoxicilline 500mg, posologie 3x/jour\n"
    "Facture n° F-2024-018\n"
    "Montant TTC: 1 250,00 MAD\n"
    "Patient: Test User — CIN AB123456 — IPP 98765\n"
    "Clinique Al-Azhar, Casablanca\n"
)

INVOICE_ONLY_TEXT = (
    "Facture n° F-2024-042\n"
    "Montant TTC: 890,00 MAD\n"
    "TVA: 20%\n"
    "Provider: Pharmacie Centrale\n"
    "Patient: John Doe — CIN XY987654\n"
)

MIXED_BUNDLE_TEXT = (
    "Rapport médical\n"
    "Diagnostic: consultation routine\n"
    "Ordonnance: vitamine D\n"
    "Pharmacie: dose 1000 UI\n"
    "Autres documents: certificat employeur, attestation assurance CNSS\n"
    "Patient: CIN LM456789\n"
    "Montant: 450 MAD\n"
)

EMPTY_OCR_TEXT = "..."


def _structured(cin: str = "", amount: str = "") -> dict:
    return {"cin": cin, "amount": amount}


def test_case1_medical_bundle_not_rejected() -> None:
    """Medical bundle like the current production example must pass coverage."""
    coverage = compute_coverage_score(
        extracted_text=MEDICAL_BUNDLE_TEXT,
        structured_data=_structured(cin="AB123456", amount="1250"),
        ml_classification={"label": "CLAIM", "confidence": 92},
        document_classifier_tool={
            "output": {
                "document_type": "medical_claim_bundle",
                "missing_docs": [],
                "found_docs": ["medical_report", "invoice", "prescription"],
            }
        },
    )
    assert coverage.overall >= MIN_COVERAGE_ACCEPT, (
        f"medical bundle should pass coverage; got {coverage.overall}"
    )
    assert coverage_decision(coverage) == "ACCEPTED"
    assert coverage.classifier_bundle == "medical_claim_bundle"
    assert coverage.medical_report_coverage > 0.0
    assert coverage.invoice_coverage > 0.0
    assert coverage.prescription_coverage > 0.0


def test_case2_invoice_only_accepted_with_warning() -> None:
    coverage = compute_coverage_score(
        extracted_text=INVOICE_ONLY_TEXT,
        structured_data=_structured(cin="XY987654", amount="890"),
        ml_classification={"label": "CLAIM", "confidence": 75},
        document_classifier_tool={
            "output": {
                "document_type": "incomplete_claim_bundle",
                "missing_docs": ["medical_report", "prescription"],
                "found_docs": ["invoice"],
            }
        },
    )
    assert coverage_decision(coverage) == "ACCEPTED", (
        f"invoice-only with strong invoice coverage should accept; got {coverage.overall}"
    )
    assert coverage.invoice_coverage > coverage.medical_report_coverage
    assert coverage.invoice_coverage > coverage.prescription_coverage
    assert any("below_warn_threshold" in w or "component_types_absent" in w
               for w in coverage.warnings), (
        f"expected low-coverage warning, got {coverage.warnings}"
    )


def test_case3_mixed_documents_enriched_explanation() -> None:
    coverage = compute_coverage_score(
        extracted_text=MIXED_BUNDLE_TEXT,
        structured_data=_structured(cin="LM456789", amount="450"),
        ml_classification={"label": "CLAIM", "confidence": 80},
        document_classifier_tool={
            "output": {
                "document_type": "hybrid_bundle",
                "missing_docs": [],
                "found_docs": ["medical_report", "invoice", "prescription"],
            }
        },
    )
    assert coverage_decision(coverage) == "ACCEPTED"
    envelope = build_explanation(
        decision="ACCEPTED",
        score=coverage.overall * 100.0,
        coverage=coverage,
        reasons=["mixed_bundle_accepted"],
        debug=True,
    )
    assert envelope["decision"] == "ACCEPTED"
    assert envelope["explanation"]["summary"]
    assert "coverage" in envelope["explanation"]["signals"]
    assert envelope["explanation"]["tool_outputs"]
    # Enriched = classifier tool AND ml classifier present
    assert "document_classifier_tool" in envelope["explanation"]["tool_outputs"]
    assert "ml_classification" in envelope["explanation"]["tool_outputs"]


def test_case4_invalid_or_empty_ocr_rejected_with_explanation() -> None:
    coverage = compute_coverage_score(
        extracted_text=EMPTY_OCR_TEXT,
        structured_data={},
        ml_classification={"label": "NON_CLAIM", "confidence": 99},
        document_classifier_tool={"output": {"document_type": "unknown_bundle"}},
    )
    assert coverage_decision(coverage) == "REJECTED"
    envelope = build_explanation(
        decision=coverage_decision(coverage),
        score=coverage.overall * 100.0,
        coverage=coverage,
        reasons=["empty_or_unreadable_ocr"],
        debug=True,
    )
    assert envelope["decision"] == "REJECTED"
    assert envelope["explanation"]["summary"]
    assert envelope["explanation"]["reasons"], "rejection must list reasons"
    assert "coverage" in envelope["explanation"]["signals"]
    # Mandatory contract: explanation must include tool outputs on failure too
    assert envelope["explanation"]["tool_outputs"]


def test_explanation_envelope_contract_shape() -> None:
    """Every exit MUST return {decision, score, explanation:{summary,reasons,signals,tool_outputs}}."""
    coverage = compute_coverage_score(
        extracted_text=MEDICAL_BUNDLE_TEXT,
        structured_data=_structured(cin="AB123456", amount="1250"),
        ml_classification={"label": "CLAIM", "confidence": 92},
        document_classifier_tool={"output": {"document_type": "medical_claim_bundle"}},
    )
    envelope = build_explanation(
        decision="ACCEPTED", score=88.5, coverage=coverage, debug=True
    )
    assert set(envelope.keys()) >= {"decision", "score", "explanation"}
    assert set(envelope["explanation"].keys()) >= {
        "summary", "reasons", "signals", "tool_outputs"
    }
    assert isinstance(envelope["explanation"]["reasons"], list)
    assert isinstance(envelope["explanation"]["signals"], dict)
    assert isinstance(envelope["explanation"]["tool_outputs"], dict)


def test_classifier_bundle_is_informational_not_gating() -> None:
    """unknown_bundle output must not by itself drive rejection."""
    coverage = compute_coverage_score(
        extracted_text=MEDICAL_BUNDLE_TEXT,
        structured_data=_structured(cin="AB123456", amount="1250"),
        ml_classification={"label": "CLAIM", "confidence": 92},
        document_classifier_tool={"output": {"document_type": "unknown_bundle"}},
    )
    assert coverage_decision(coverage) == "ACCEPTED", (
        "bundle='unknown_bundle' must not gate; coverage still carries the decision"
    )
    assert coverage.classifier_bundle == "unknown_bundle"
    assert "classifier_bundle_unknown" in coverage.warnings
