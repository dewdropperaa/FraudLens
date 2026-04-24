from __future__ import annotations

from claimguard.agents.validation_agent import ClaimValidationAgent


def _base_claim() -> dict:
    return {
        "identity": {"cin": "AB123456", "name": "Sara Benali", "hospital": "Clinique Atlas"},
        "policy": {"diagnosis": "Consultation", "service_description": "Consultation generale"},
        "metadata": {"service_date": "2026-03-01", "amount": 450, "claim_id": "coverage-001"},
        "amount": 450,
    }


def _payload(result: dict) -> dict:
    return result.get("output", result)


def test_case_1_medical_bundle_not_rejected_by_type_mismatch() -> None:
    agent = ClaimValidationAgent()
    claim = _base_claim()
    claim["documents"] = ["bundle.pdf"]
    claim["document_extractions"] = [
        {
            "file_name": "bundle.pdf",
            "extracted_text": (
                "Ordonnance Dr. Alaoui\n"
                "Facture Clinique Atlas\n"
                "Patient Sara Benali CIN AB123456\n"
                "Date 01/03/2026 Total 450 MAD"
            ),
        }
    ]
    payload = _payload(agent.analyze(claim))
    assert payload["validation_score"] >= 40
    assert payload["explanation"]["summary"]
    assert "tool_outputs" in payload["explanation"]


def test_case_2_invoice_only_accepted_with_low_coverage_warning() -> None:
    agent = ClaimValidationAgent()
    claim = _base_claim()
    claim["documents"] = ["invoice.pdf"]
    claim["document_extractions"] = [
        {"file_name": "invoice.pdf", "extracted_text": "Facture clinique total 450 MAD"}
    ]
    payload = _payload(agent.analyze(claim))
    assert payload["validation_status"] == "VALID"
    assert payload["validation_score"] >= 40
    assert payload["details"]["coverage_metrics"]["coverage_score"] < 0.7
    assert payload["explanation"]["summary"]


def test_case_3_mixed_documents_accepted_with_enriched_explanation() -> None:
    agent = ClaimValidationAgent()
    claim = _base_claim()
    claim["documents"] = ["invoice.pdf", "lab.pdf", "ordonnance.pdf"]
    claim["document_extractions"] = [
        {"file_name": "invoice.pdf", "extracted_text": "Facture clinique total 450 MAD"},
        {"file_name": "lab.pdf", "extracted_text": "Laboratoire bilan sanguin patient Sara Benali"},
        {"file_name": "ordonnance.pdf", "extracted_text": "Ordonnance traitement prescrit"},
    ]
    payload = _payload(agent.analyze(claim))
    assert payload["validation_status"] == "VALID"
    assert payload["validation_score"] >= 40
    assert len(payload["explanation"]["reasons"]) >= 2
    assert "signals" in payload["explanation"]


def test_case_4_invalid_empty_ocr_rejected_with_explanation() -> None:
    agent = ClaimValidationAgent()
    claim = _base_claim()
    claim["documents"] = ["empty.pdf"]
    claim["document_extractions"] = [{"file_name": "empty.pdf", "extracted_text": ""}]
    payload = _payload(agent.analyze(claim))
    assert payload["validation_status"] == "INVALID"
    assert payload["validation_score"] < 40
    assert payload["explanation"]["summary"]
    assert "tool_outputs" in payload["explanation"]
