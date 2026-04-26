from __future__ import annotations

from claimguard.v2.orchestrator import _normalize_validation_document_type


def test_medical_claim_bundle_is_preserved_as_informational_cluster() -> None:
    normalized, original = _normalize_validation_document_type("medical_claim_bundle")
    assert normalized == "medical_claim_bundle"
    assert original is None


def test_incomplete_bundle_is_remapped_to_hybrid_bundle() -> None:
    normalized, original = _normalize_validation_document_type("incomplete_claim_bundle")
    assert normalized == "hybrid_bundle"
    assert original == "incomplete_claim_bundle"


def test_arbitrary_label_is_preserved_and_never_rejected() -> None:
    normalized, original = _normalize_validation_document_type("totally_new_type")
    assert normalized == "totally_new_type"
    assert original is None
