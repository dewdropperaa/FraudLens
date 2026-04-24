from __future__ import annotations

from claimguard.v2.orchestrator import _normalize_validation_document_type


def test_maps_medical_claim_bundle_to_supported_type() -> None:
    normalized, original = _normalize_validation_document_type("medical_claim_bundle")
    assert normalized == "hospital_bill"
    assert original is None


def test_maps_incomplete_bundle_to_unknown_without_crashing() -> None:
    normalized, original = _normalize_validation_document_type("incomplete_claim_bundle")
    assert normalized == "unknown"
    assert original is None


def test_unknown_type_is_downgraded_and_preserved_for_forensics() -> None:
    normalized, original = _normalize_validation_document_type("totally_new_type")
    assert normalized == "unknown"
    assert original == "totally_new_type"
